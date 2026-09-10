"""Stage 4.1 MuZero network + inference-contract tests.

Covers representation, initial/recurrent inference, action encoding, masking,
determinism, gradients, device handling, batch support, and multi-step latent
unrolling.  No environment, MCTS, or training code is exercised: the point is
that the *pure neural* interface is mathematically consistent.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from forexmind.muzero import (
    MuZeroConfig,
    MuZeroNetwork,
    NetworkOutput,
    apply_action_mask,
    build_muzero_network,
    observation_dim,
)
from forexmind.muzero.config import observation_dim as observation_dim_fn
from forexmind.observation.encoder import EncoderConfig

NUM_ACTIONS = 10


def make_config(**overrides) -> MuZeroConfig:
    base = dict(
        obs_dim=16,
        num_actions=NUM_ACTIONS,
        latent_dim=8,
        hidden_dim=16,
        action_embedding_dim=4,
        num_layers=2,
        use_support=True,
        value_support_size=11,
        reward_support_size=11,
    )
    base.update(overrides)
    return MuZeroConfig(**base)


@pytest.fixture
def model() -> MuZeroNetwork:
    torch.manual_seed(0)
    net = build_muzero_network(make_config())
    net.eval()
    return net


def _obs(batch: int, obs_dim: int) -> torch.Tensor:
    return torch.randn(batch, obs_dim)


# --------------------------------------------------------------------------- #
# Observation contract / config derivation
# --------------------------------------------------------------------------- #


def test_observation_dim_derived_from_encoder_not_hard_coded() -> None:
    expected = EncoderConfig().spec.encoded_shape[0]
    assert observation_dim() == expected
    assert observation_dim_fn() == expected
    # Current Phase-2 defaults: 64 * 5 market + 10 account + 14 time + 7 instrument.
    assert expected == 351


def test_config_from_encoder_config_sets_obs_dim() -> None:
    config = MuZeroConfig.from_encoder_config(EncoderConfig(context_length=8), latent_dim=4)
    assert config.obs_dim == 8 * 5 + 10 + 14 + 7


def test_config_validates_architecture_fields() -> None:
    with pytest.raises(ValueError):
        make_config(obs_dim=0)
    with pytest.raises(ValueError):
        make_config(activation="nope")
    with pytest.raises(ValueError):
        make_config(value_support_size=20)  # even
    with pytest.raises(ValueError):
        make_config(num_layers=0)
    with pytest.raises(ValueError):
        make_config(use_support=False, value_scale=0.0)


def test_config_output_dims_track_support_setting() -> None:
    assert make_config(use_support=True).value_output_dim == 11
    assert make_config(use_support=False).value_output_dim == 1
    assert make_config(use_support=False).scalar_head is True


# --------------------------------------------------------------------------- #
# Representation network
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("batch", [1, 4, 32])
def test_representation_shapes_and_finiteness(model: MuZeroNetwork, batch: int) -> None:
    obs = _obs(batch, model.config.obs_dim)
    with torch.no_grad():
        latent = model.representation(obs)
    assert latent.shape == (batch, model.config.latent_dim)
    assert torch.isfinite(latent).all()


def test_representation_rejects_wrong_obs_dim(model: MuZeroNetwork) -> None:
    with pytest.raises(ValueError):
        model.representation(torch.randn(2, model.config.obs_dim + 1))


# --------------------------------------------------------------------------- #
# Initial inference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("batch", [1, 4, 32])
def test_initial_inference_contract(model: MuZeroNetwork, batch: int) -> None:
    obs = _obs(batch, model.config.obs_dim)
    with torch.no_grad():
        out = model.initial_inference(obs)
    assert out.latent_state.shape == (batch, model.config.latent_dim)
    assert out.policy_logits.shape == (batch, NUM_ACTIONS)
    assert out.value.shape == (batch, 1)
    assert out.reward.shape == (batch, 1)
    assert out.value_logits is not None and out.value_logits.shape == (batch, 11)
    assert out.reward_logits is None
    assert torch.equal(out.reward, torch.zeros_like(out.reward))
    for tensor in (out.latent_state, out.policy_logits, out.value):
        assert torch.isfinite(tensor).all()


def test_initial_inference_accepts_single_observation(model: MuZeroNetwork) -> None:
    obs = torch.randn(model.config.obs_dim)
    with torch.no_grad():
        out = model.initial_inference(obs)
    assert out.batch_size == 1


def test_initial_inference_accepts_numpy(model: MuZeroNetwork) -> None:
    obs = np.random.randn(3, model.config.obs_dim).astype(np.float32)
    with torch.no_grad():
        out = model.initial_inference(obs)
    assert out.batch_size == 3


# --------------------------------------------------------------------------- #
# Recurrent inference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("action", list(range(NUM_ACTIONS)))
def test_recurrent_inference_all_actions_finite(model: MuZeroNetwork, action: int) -> None:
    latent = torch.randn(4, model.config.latent_dim)
    with torch.no_grad():
        out = model.recurrent_inference(latent, action)
    assert out.latent_state.shape == (4, model.config.latent_dim)
    assert out.policy_logits.shape == (4, NUM_ACTIONS)
    assert out.value.shape == (4, 1)
    assert out.reward.shape == (4, 1)
    assert out.reward_logits is not None and out.reward_logits.shape == (4, 11)
    for tensor in (out.latent_state, out.policy_logits, out.value, out.reward):
        assert torch.isfinite(tensor).all()


def test_recurrent_inference_is_pure_neural(model: MuZeroNetwork) -> None:
    """A latent state can be supplied directly: no environment is consulted."""
    latent = torch.full((2, model.config.latent_dim), 0.25)
    with torch.no_grad():
        out = model.recurrent_inference(latent, torch.tensor([0, 5]))
    assert out.batch_size == 2


def test_recurrent_inference_accepts_per_batch_actions(model: MuZeroNetwork) -> None:
    latent = torch.randn(3, model.config.latent_dim)
    with torch.no_grad():
        out = model.recurrent_inference(latent, torch.tensor([1, 2, 3]))
    assert out.batch_size == 3


def test_recurrent_inference_rejects_action_batch_mismatch(model: MuZeroNetwork) -> None:
    latent = torch.randn(3, model.config.latent_dim)
    with pytest.raises(ValueError):
        model.recurrent_inference(latent, torch.tensor([1, 2]))


@pytest.mark.parametrize("bad_action", [-1, 10, 11, 100])
def test_invalid_action_indices_fail_clearly(model: MuZeroNetwork, bad_action: int) -> None:
    latent = torch.randn(2, model.config.latent_dim)
    with pytest.raises(ValueError, match="out of range"):
        model.recurrent_inference(latent, bad_action)


def test_action_batch_range_validated_elementwise(model: MuZeroNetwork) -> None:
    latent = torch.randn(3, model.config.latent_dim)
    with pytest.raises(ValueError, match="out of range"):
        model.recurrent_inference(latent, torch.tensor([0, 1, 12]))


def test_non_integral_float_action_rejected(model: MuZeroNetwork) -> None:
    latent = torch.randn(2, model.config.latent_dim)
    with pytest.raises(ValueError):
        model.recurrent_inference(latent, torch.tensor([0.5, 1.0]))


# --------------------------------------------------------------------------- #
# Action encoding
# --------------------------------------------------------------------------- #


def test_action_embeddings_are_distinct(model: MuZeroNetwork) -> None:
    embedding = model.dynamics.action_embedding.weight.detach()
    assert embedding.shape == (NUM_ACTIONS, model.config.action_embedding_dim)
    for i in range(NUM_ACTIONS):
        for j in range(i + 1, NUM_ACTIONS):
            assert not torch.allclose(embedding[i], embedding[j])


def test_dynamics_uses_embedding_not_raw_index(model: MuZeroNetwork) -> None:
    latent = torch.zeros(1, model.config.latent_dim)
    with torch.no_grad():
        out_a = model.dynamics(latent, torch.tensor([0]))
        out_b = model.dynamics(latent, torch.tensor([9]))
    assert not torch.allclose(out_a[0], out_b[0])


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_inference_is_deterministic(model: MuZeroNetwork) -> None:
    obs = _obs(4, model.config.obs_dim)
    latent = torch.randn(4, model.config.latent_dim)
    with torch.no_grad():
        first = model.initial_inference(obs)
        second = model.initial_inference(obs)
        child_a = model.recurrent_inference(latent, torch.tensor([0, 1, 2, 3]))
        child_b = model.recurrent_inference(latent, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(first.policy_logits, second.policy_logits)
    assert torch.equal(first.value, second.value)
    assert torch.equal(child_a.latent_state, child_b.latent_state)
    assert torch.equal(child_a.reward, child_b.reward)


def test_no_sampling_inside_network(model: MuZeroNetwork) -> None:
    """Repeated calls with a fixed seed leave outputs unchanged (no randomness)."""
    obs = _obs(2, model.config.obs_dim)
    with torch.no_grad():
        a = model.initial_inference(obs).policy_logits
        torch.manual_seed(999)
        b = model.initial_inference(obs).policy_logits
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# Action masks
# --------------------------------------------------------------------------- #


def test_mask_zeroes_invalid_action_probability(model: MuZeroNetwork) -> None:
    obs = _obs(3, model.config.obs_dim)
    mask = torch.ones(3, NUM_ACTIONS, dtype=torch.bool)
    mask[:, 1] = False
    mask[:, 5] = False
    with torch.no_grad():
        out = model.initial_inference(obs, mask)
    probs = torch.softmax(out.policy_logits, dim=-1)
    assert torch.all(probs[:, 1] == 0.0)
    assert torch.all(probs[:, 5] == 0.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_mask_preserves_hold_and_at_least_one_action(model: MuZeroNetwork) -> None:
    obs = _obs(1, model.config.obs_dim)
    mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)
    mask[:, 1:] = False  # only HOLD valid
    with torch.no_grad():
        out = model.initial_inference(obs, mask)
    probs = torch.softmax(out.policy_logits, dim=-1)
    assert probs[0, 0].item() == pytest.approx(1.0)


def test_mask_rejects_hold_removed() -> None:
    logits = torch.randn(2, NUM_ACTIONS)
    mask = torch.ones(2, NUM_ACTIONS, dtype=torch.bool)
    mask[:, 0] = False
    with pytest.raises(ValueError, match="HOLD"):
        apply_action_mask(logits, mask)


def test_mask_rejects_all_false_row() -> None:
    logits = torch.randn(2, NUM_ACTIONS)
    mask = torch.ones(2, NUM_ACTIONS, dtype=torch.bool)
    mask[1, :] = False
    with pytest.raises(ValueError, match="at least one valid action"):
        apply_action_mask(logits, mask)


def test_mask_shape_must_match_logits() -> None:
    logits = torch.randn(2, NUM_ACTIONS)
    with pytest.raises(ValueError, match="action_mask must have shape"):
        apply_action_mask(logits, torch.ones(2, NUM_ACTIONS + 1, dtype=torch.bool))


def test_one_dimensional_mask_is_broadcast() -> None:
    logits = torch.randn(4, NUM_ACTIONS)
    mask = torch.ones(NUM_ACTIONS, dtype=torch.bool)
    mask[2] = False
    masked = apply_action_mask(logits, mask)
    probs = torch.softmax(masked, dim=-1)
    assert torch.all(probs[:, 2] == 0.0)


def test_apply_action_mask_is_pure_function() -> None:
    logits = torch.randn(2, NUM_ACTIONS)
    original = logits.clone()
    mask = torch.ones(2, NUM_ACTIONS, dtype=torch.bool)
    mask[:, 3] = False
    _ = apply_action_mask(logits, mask)
    assert torch.equal(logits, original)  # raw logits untouched


def test_unmasked_logits_are_raw(model: MuZeroNetwork) -> None:
    obs = _obs(2, model.config.obs_dim)
    with torch.no_grad():
        out = model.initial_inference(obs)
        raw = model.prediction(model.representation(obs))[0]
    assert torch.allclose(out.policy_logits, raw)


# --------------------------------------------------------------------------- #
# Batch / device
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("batch", [1, 4, 32])
def test_batch_sizes_supported(model: MuZeroNetwork, batch: int) -> None:
    obs = _obs(batch, model.config.obs_dim)
    with torch.no_grad():
        out = model.initial_inference(obs)
        child = model.recurrent_inference(out.latent_state, torch.zeros(batch, dtype=torch.long))
    assert out.batch_size == batch
    assert child.batch_size == batch


def test_outputs_stay_on_model_device(model: MuZeroNetwork) -> None:
    obs = _obs(2, model.config.obs_dim)
    with torch.no_grad():
        out = model.initial_inference(obs)
    assert out.policy_logits.device == model.device
    assert out.value.device == model.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device_supported() -> None:  # pragma: no cover - GPU only
    torch.manual_seed(0)
    net = build_muzero_network(make_config()).to("cuda").eval()
    obs = torch.randn(2, net.config.obs_dim, device="cuda")
    with torch.no_grad():
        out = net.initial_inference(obs)
    assert out.latent_state.device.type == "cuda"
    assert torch.isfinite(out.value).all()


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #


def test_synthetic_loss_produces_finite_gradients(model: MuZeroNetwork) -> None:
    torch.manual_seed(0)
    obs = _obs(4, model.config.obs_dim)
    root = model.initial_inference(obs)
    child = model.recurrent_inference(root.latent_state, torch.tensor([0, 1, 2, 3]))
    target = torch.tensor([0, 4, 2, 9])
    loss = (
        F.cross_entropy(root.policy_logits, target)
        + F.cross_entropy(child.policy_logits, target)
        + root.value.pow(2).mean()
        + child.value.pow(2).mean()
        + child.reward.pow(2).mean()
    )
    assert torch.isfinite(loss)
    model.zero_grad(set_to_none=True)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
    assert float(model.dynamics.action_embedding.weight.grad.abs().sum()) > 0.0
    assert float(model.dynamics.reward_head.weight.grad.abs().sum()) > 0.0


def test_reward_head_receives_gradient(model: MuZeroNetwork) -> None:
    root = model.initial_inference(_obs(3, model.config.obs_dim))
    child = model.recurrent_inference(root.latent_state, torch.tensor([0, 1, 2]))
    model.zero_grad(set_to_none=True)
    child.reward.sum().backward()
    grad = model.dynamics.reward_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0.0


# --------------------------------------------------------------------------- #
# Multi-step latent unroll
# --------------------------------------------------------------------------- #


def test_five_step_latent_unroll_is_stable(model: MuZeroNetwork) -> None:
    torch.manual_seed(1)
    obs = _obs(2, model.config.obs_dim)
    sequence = [0, 3, 7, 9, 4]
    with torch.no_grad():
        out = model.initial_inference(obs)
        latents = [out.latent_state]
        for action in sequence:
            out = model.recurrent_inference(out.latent_state, action)
            latents.append(out.latent_state)
            assert out.latent_state.shape == (2, model.config.latent_dim)
            assert torch.isfinite(out.latent_state).all()
            assert torch.isfinite(out.reward).all()
            assert torch.isfinite(out.value).all()
            assert torch.isfinite(out.policy_logits).all()
    assert len(latents) == 6


def test_different_action_sequences_diverge(model: MuZeroNetwork) -> None:
    torch.manual_seed(2)
    obs = _obs(1, model.config.obs_dim)
    with torch.no_grad():
        root = model.initial_inference(obs)
        seq_a = root.latent_state
        seq_b = root.latent_state
        for action in [2, 2, 2, 2, 2]:
            seq_a = model.recurrent_inference(seq_a, action).latent_state
        for action in [9, 9, 9, 9, 9]:
            seq_b = model.recurrent_inference(seq_b, action).latent_state
    assert not torch.allclose(seq_a, seq_b, atol=1e-6)


def test_action_order_matters(model: MuZeroNetwork) -> None:
    torch.manual_seed(3)
    obs = _obs(1, model.config.obs_dim)
    with torch.no_grad():
        root = model.initial_inference(obs)
        ab = root.latent_state
        for action in (1, 8):
            ab = model.recurrent_inference(ab, action).latent_state
        ba = root.latent_state
        for action in (8, 1):
            ba = model.recurrent_inference(ba, action).latent_state
    assert not torch.allclose(ab, ba, atol=1e-6)


# --------------------------------------------------------------------------- #
# Reporting + scalar head variant
# --------------------------------------------------------------------------- #


def test_parameter_report_sums(model: MuZeroNetwork) -> None:
    report = model.parameter_report()
    assert report["representation"] > 0
    assert report["dynamics"] > 0
    assert report["prediction"] > 0
    assert report["total"] == (report["representation"] + report["dynamics"] + report["prediction"])


def test_scalar_head_variant_has_no_support_logits() -> None:
    torch.manual_seed(0)
    net = build_muzero_network(make_config(use_support=False))
    net.eval()
    with torch.no_grad():
        root = net.initial_inference(_obs(2, net.config.obs_dim))
        child = net.recurrent_inference(root.latent_state, torch.tensor([0, 1]))
    assert root.value_logits is None
    assert child.value_logits is None
    assert child.reward_logits is None
    assert root.value.shape == (2, 1)
    assert child.reward.shape == (2, 1)
    assert torch.isfinite(child.reward).all()


def test_scalar_and_support_share_public_api() -> None:
    for use_support in (True, False):
        torch.manual_seed(0)
        net = build_muzero_network(make_config(use_support=use_support))
        net.eval()
        out = net.initial_inference(_obs(1, net.config.obs_dim))
        assert isinstance(out, NetworkOutput)
        assert out.policy_logits.shape == (1, NUM_ACTIONS)


# --------------------------------------------------------------------------- #
# NetworkOutput validation
# --------------------------------------------------------------------------- #


def test_network_output_validates_shapes() -> None:
    with pytest.raises(ValueError, match="value"):
        NetworkOutput(
            latent_state=torch.zeros(2, 8),
            policy_logits=torch.zeros(2, NUM_ACTIONS),
            value=torch.zeros(3, 1),
            reward=torch.zeros(2, 1),
        )
    with pytest.raises(ValueError, match="policy_logits"):
        NetworkOutput(
            latent_state=torch.zeros(2, 8),
            policy_logits=torch.zeros(3, NUM_ACTIONS),
            value=torch.zeros(2, 1),
            reward=torch.zeros(2, 1),
        )


def test_network_output_rejects_bad_logit_rank() -> None:
    with pytest.raises(ValueError, match="value_logits"):
        NetworkOutput(
            latent_state=torch.zeros(2, 8),
            policy_logits=torch.zeros(2, NUM_ACTIONS),
            value=torch.zeros(2, 1),
            reward=torch.zeros(2, 1),
            value_logits=torch.zeros(2),
        )
