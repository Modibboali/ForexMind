"""MuZero loss tests (Stage 4.4 §4-§13, §21, §22).

Exactness is checked by hand: with only one valid target position, the reported
loss must equal a directly computed cross-entropy for that position.  That is
the off-by-one regression guard for every head.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from forexmind.muzero.losses import (
    LossConfig,
    MuZeroPrediction,
    muzero_losses,
)
from forexmind.muzero.support import scalar_to_support
from forexmind.muzero.targets import TargetConfig, build_unroll_sample, collate_samples

from tests.muzero_synthetic import make_trajectory, one_hot

ACTIONS = 6
OBS_DIM = 8
SUPPORT = 31
VALUE_SCALE = 0.5
REWARD_SCALE = 0.2


def _batch(
    *,
    rewards: list[float],
    actions: list[int] | None = None,
    root_policies: list | None = None,
    root_values: list[float] | None = None,
    terminated: list[bool] | None = None,
    truncated: list[bool] | None = None,
    boundary_value: float = 0.0,
    unroll: int = 5,
    td_steps: int = 3,
    split: str = "train",
    initial_is_flat: bool = False,
):
    steps = len(rewards)
    actions = actions if actions is not None else [0] * steps
    trajectory = make_trajectory(
        actions=list(actions),
        rewards=list(rewards),
        root_policies=root_policies,
        root_values=root_values,
        terminated=terminated,
        truncated=truncated,
        boundary_value=boundary_value,
        obs_dim=OBS_DIM,
        split=split,
        # Start non-flat with zero exposure so every action is legal by default:
        # loss tests want an all-valid mask unless they ask for a flat account.
        initial_exposure=0.0,
        initial_is_flat=initial_is_flat,
    )
    sample = build_unroll_sample(
        trajectory, 0, TargetConfig(num_unroll_steps=unroll, td_steps=td_steps, discount=0.99)
    )
    return collate_samples([sample])


def _prediction(
    batch,
    *,
    policy_logits: torch.Tensor | None = None,
    value_logits: torch.Tensor | None = None,
    reward_logits: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    reward: torch.Tensor | None = None,
) -> MuZeroPrediction:
    rows, positions, _ = batch.target_policies.shape
    reward_positions = positions - 1
    policy_logits = (
        torch.zeros(rows, positions, ACTIONS) if policy_logits is None else policy_logits
    )
    value_logits = torch.zeros(rows, positions, SUPPORT) if value_logits is None else value_logits
    reward_logits = (
        torch.zeros(rows, reward_positions, SUPPORT) if reward_logits is None else reward_logits
    )
    value = torch.zeros(rows, positions) if value is None else value
    reward = torch.zeros(rows, reward_positions) if reward is None else reward
    return MuZeroPrediction(
        policy_logits=policy_logits,
        value_logits=value_logits,
        reward_logits=reward_logits,
        value=value,
        reward=reward,
        latents=[],
    )


def _config(**overrides) -> LossConfig:
    base = dict(value_scale=VALUE_SCALE, reward_scale=REWARD_SCALE)
    base.update(overrides)
    return LossConfig(**base)


def _loss(prediction, batch, config=None, *, use_support: bool = True):
    return muzero_losses(
        prediction,
        batch,
        config or _config(),
        use_support=use_support,
        value_support_size=SUPPORT,
        reward_support_size=SUPPORT,
    )


def _ce(prediction: torch.Tensor, target_distribution: torch.Tensor) -> float:
    return float(-(target_distribution * torch.log_softmax(prediction, dim=-1)).sum(-1).mean())


# --------------------------------------------------------------------------- #
# Alignment: one valid position at a time
# --------------------------------------------------------------------------- #


def test_reward_loss_aligns_with_its_own_unroll_step() -> None:
    # Steps 2 and 3 land on very different support bins, so the alignment check
    # is genuinely discriminating.
    batch = _batch(rewards=[0.0, 0.0, 0.0, 0.34, 0.0], unroll=5)
    batch.reward_masks.zero_()
    batch.reward_masks[0, 2] = 1.0
    batch.policy_masks.zero_()
    batch.value_masks.zero_()

    reward_logits = torch.zeros(1, 5, SUPPORT)
    reward_logits[:, :, SUPPORT // 2] = 6.0  # strong prediction of "near zero"
    result = _loss(_prediction(batch, reward_logits=reward_logits), batch)

    expected = _ce(
        reward_logits[0, 2],
        scalar_to_support(batch.target_rewards[0, 2].reshape(1), SUPPORT, scale=REWARD_SCALE),
    )
    other = _ce(
        reward_logits[0, 2],
        scalar_to_support(batch.target_rewards[0, 3].reshape(1), SUPPORT, scale=REWARD_SCALE),
    )
    assert abs(expected - other) > 1e-3  # the test can actually tell them apart
    assert float(result.reward) == pytest.approx(expected, rel=1e-5)


def test_policy_loss_aligns_with_its_own_unroll_step() -> None:
    policies = [one_hot(0), one_hot(2), one_hot(4), one_hot(1), one_hot(5)]
    batch = _batch(rewards=[0.0] * 5, root_policies=policies, unroll=4)
    batch.policy_masks.zero_()
    batch.policy_masks[0, 3] = 1.0
    batch.value_masks.zero_()
    batch.reward_masks.zero_()

    policy_logits = torch.zeros(1, 5, ACTIONS)
    policy_logits[0, 3, 0] = 5.0
    result = _loss(_prediction(batch, policy_logits=policy_logits), batch)

    expected = float(
        F.cross_entropy(policy_logits[0, 3].unsqueeze(0), batch.target_policies[0, 3].unsqueeze(0))
    )
    assert float(result.policy) == pytest.approx(expected, rel=1e-5)


def test_value_loss_aligns_with_its_own_unroll_step() -> None:
    batch = _batch(rewards=[0.1, 0.2, 0.3, 0.4, 0.5], unroll=4)
    batch.policy_masks.zero_()
    batch.reward_masks.zero_()
    batch.value_masks.zero_()
    batch.value_masks[0, 2] = 1.0

    value_logits = torch.zeros(1, 5, SUPPORT)
    value_logits[0, 2, -1] = 8.0
    result = _loss(_prediction(batch, value_logits=value_logits), batch)

    expected = _ce(
        value_logits[0, 2],
        scalar_to_support(batch.target_values[0, 2].reshape(1), SUPPORT, scale=VALUE_SCALE),
    )
    assert float(result.value) == pytest.approx(expected, rel=1e-5)


def test_no_reward_target_at_the_initial_state() -> None:
    """``target_rewards`` has length K while values/policies have K+1."""
    batch = _batch(rewards=[0.0] * 6, unroll=5)
    assert batch.actions.shape == (1, 5)
    assert batch.target_rewards.shape == (1, 5)
    assert batch.target_values.shape == (1, 6)
    assert batch.target_policies.shape == (1, 6, ACTIONS)
    assert batch.reward_masks.shape == (1, 5)


# --------------------------------------------------------------------------- #
# Loss values
# --------------------------------------------------------------------------- #


def test_perfect_policy_prediction_has_zero_loss() -> None:
    policies = [one_hot(0), one_hot(3), one_hot(5)]
    batch = _batch(rewards=[0.0] * 3, root_policies=policies, unroll=2)
    batch.value_masks.zero_()
    batch.reward_masks.zero_()
    logits = torch.log(batch.target_policies.clamp_min(1e-12))
    result = _loss(_prediction(batch, policy_logits=logits), batch)
    assert float(result.policy) == pytest.approx(0.0, abs=1e-5)
    assert result.diagnostics["policy_kl"] == pytest.approx(0.0, abs=1e-5)
    assert result.diagnostics["policy_top1_agreement"] == pytest.approx(1.0)


def test_perfect_support_prediction_reaches_the_target_entropy() -> None:
    """Cross-entropy is minimized (not zeroed) when the prediction equals the target."""
    batch = _batch(rewards=[0.1, -0.05, 0.02], unroll=2)
    value_dist = scalar_to_support(batch.target_values, SUPPORT, scale=VALUE_SCALE)
    reward_dist = scalar_to_support(batch.target_rewards, SUPPORT, scale=REWARD_SCALE)
    prediction = _prediction(
        batch,
        value_logits=torch.log(value_dist.clamp_min(1e-12)),
        reward_logits=torch.log(reward_dist.clamp_min(1e-12)),
    )
    batch.policy_masks.zero_()
    result = _loss(prediction, batch, _config(policy_loss_weight=0.0))
    value_entropy = float(-(value_dist * torch.log(value_dist.clamp_min(1e-12))).sum(-1).mean())
    reward_entropy = float(-(reward_dist * torch.log(reward_dist.clamp_min(1e-12))).sum(-1).mean())
    assert value_entropy > 0.0
    assert float(result.value) == pytest.approx(value_entropy, rel=1e-4)
    assert float(result.reward) == pytest.approx(reward_entropy, rel=1e-4)


def test_loss_weights_compose_the_total() -> None:
    batch = _batch(rewards=[0.1, -0.2, 0.05, 0.3], unroll=3)
    prediction = _prediction(batch)
    config = _config(policy_loss_weight=2.0, value_loss_weight=3.0, reward_loss_weight=4.0)
    result = _loss(prediction, batch, config)
    expected = 2.0 * float(result.policy) + 3.0 * float(result.value) + 4.0 * float(result.reward)
    assert float(result.total) == pytest.approx(expected, rel=1e-6)


def test_scalar_representation_path_trains_huber_regression() -> None:
    batch = _batch(rewards=[0.1, -0.2, 0.05, 0.3], unroll=3)
    prediction = _prediction(
        batch,
        value=torch.zeros_like(batch.target_values),
        reward=torch.zeros_like(batch.target_rewards),
    )
    config = _config(scalar_loss="huber", huber_delta=1.0)
    result = _loss(prediction, batch, config, use_support=False)
    assert torch.isfinite(result.total)
    assert float(result.value) > 0.0
    assert float(result.reward) > 0.0
    # Huber with beta=1 is 0.5 * e^2 while |e| < 1
    expected = float(0.5 * batch.target_rewards.pow(2).mean())
    assert float(result.reward) == pytest.approx(expected, rel=1e-4)


def test_mse_scalar_loss_matches_manual_computation() -> None:
    batch = _batch(rewards=[0.1, -0.2], unroll=1)
    prediction = _prediction(
        batch,
        value=torch.full_like(batch.target_values, 0.25),
        reward=torch.zeros_like(batch.target_rewards),
    )
    result = _loss(prediction, batch, _config(scalar_loss="mse"), use_support=False)
    expected = float((batch.target_values - 0.25).pow(2).mean())
    assert float(result.value) == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------------------- #
# §13 padding / mask normalization
# --------------------------------------------------------------------------- #


def test_padding_does_not_dilute_the_loss() -> None:
    """A trailing all-padded row must not change the mean loss."""
    short = _batch(rewards=[0.1, 0.2], unroll=5)
    long = _batch(rewards=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], unroll=5)
    assert float(short.reward_masks.sum()) == 2.0
    assert float(long.reward_masks.sum()) == 5.0

    prediction_short = _prediction(short)
    prediction_long = _prediction(long)
    result_short = _loss(prediction_short, short)
    result_long = _loss(prediction_long, long)

    assert result_short.valid_reward_targets == 2
    assert result_long.valid_reward_targets == 5
    # uniform logits give log(SUPPORT) per valid reward either way
    assert float(result_short.reward) == pytest.approx(math.log(SUPPORT), rel=1e-4)
    assert float(result_long.reward) == pytest.approx(math.log(SUPPORT), rel=1e-4)


def test_valid_target_counts_are_reported() -> None:
    batch = _batch(rewards=[0.1, 0.2, 0.3], unroll=5)
    result = _loss(_prediction(batch), batch)
    # T = 3: three reward targets, four existing states, three of which have a
    # stored search policy (the final observation does not).
    assert result.valid_reward_targets == 3
    assert result.valid_policy_targets == 3
    assert result.valid_value_targets == 4


def test_fully_masked_terms_contribute_zero_not_nan() -> None:
    batch = _batch(rewards=[0.1, 0.2], unroll=2)
    batch.reward_masks.zero_()
    result = _loss(_prediction(batch), batch)
    assert result.valid_reward_targets == 0
    assert float(result.reward) == pytest.approx(0.0)
    assert torch.isfinite(result.total)


# --------------------------------------------------------------------------- #
# §6 action-mask handling
# --------------------------------------------------------------------------- #


def test_masked_logits_ignore_invalid_actions() -> None:
    # A flat account makes FLAT illegal everywhere, and the MCTS target already
    # excludes it, so the target is uniform over the 5 legal actions.
    batch = _batch(rewards=[0.0] * 3, unroll=2, initial_is_flat=True)
    assert not bool(batch.action_masks[0, 0, 1])
    policy_logits = torch.zeros(1, 3, ACTIONS)
    policy_logits[:, :, 1] = 50.0  # huge logit on the illegal action
    result = _loss(
        _prediction(batch, policy_logits=policy_logits), batch, _config(mask_policy_logits=True)
    )
    # the masked softmax is uniform over the 5 legal actions, exactly like the target
    assert float(result.policy) == pytest.approx(math.log(ACTIONS - 1), rel=1e-4)


def test_unmasked_logits_expose_the_invalid_action() -> None:
    batch = _batch(rewards=[0.0] * 3, unroll=2, initial_is_flat=True)
    policy_logits = torch.zeros(1, 3, ACTIONS)
    policy_logits[:, :, 1] = 50.0
    result = _loss(
        _prediction(batch, policy_logits=policy_logits), batch, _config(mask_policy_logits=False)
    )
    # leaking mass onto an illegal action makes the legal probabilities tiny
    assert float(result.policy) > math.log(ACTIONS)


def test_corrupted_policy_target_is_rejected() -> None:
    batch = _batch(rewards=[0.0] * 3, unroll=2, initial_is_flat=True)
    batch.target_policies[:, :, 1] += 0.25  # illegal mass on FLAT
    with pytest.raises(ValueError, match="invalid action"):
        _loss(_prediction(batch), batch)


def test_target_validation_can_be_disabled() -> None:
    batch = _batch(rewards=[0.0] * 3, unroll=2, initial_is_flat=True)
    batch.target_policies[:, :, 1] += 0.25
    result = _loss(
        _prediction(batch),
        batch,
        _config(validate_targets=False, mask_policy_logits=False),
    )
    assert torch.isfinite(result.total)


# --------------------------------------------------------------------------- #
# §21 / §22 diagnostics
# --------------------------------------------------------------------------- #


def test_diagnostics_cover_policy_value_reward_and_hold() -> None:
    batch = _batch(rewards=[0.1, -0.2, 0.05], unroll=3)
    result = _loss(_prediction(batch), batch)
    for key in (
        "policy_entropy",
        "target_policy_entropy",
        "policy_kl",
        "policy_top1_agreement",
        "value_pred_mean",
        "value_target_mean",
        "value_mae",
        "value_rmse",
        "reward_pred_mean",
        "reward_target_mean",
        "reward_mae",
        "reward_rmse",
        "pred_hold_prob",
        "target_hold_prob",
        "pred_argmax_hold_fraction",
        "target_argmax_hold_fraction",
        "pred_group_hold",
        "pred_group_flat",
        "pred_group_short",
        "pred_group_long",
        "target_group_hold",
        "target_group_flat",
        "target_group_short",
        "target_group_long",
    ):
        assert key in result.diagnostics, key
        assert math.isfinite(result.diagnostics[key]), key


def test_latent_diagnostics_are_reported_per_unroll_step() -> None:
    batch = _batch(rewards=[0.0] * 3, unroll=2)
    prediction = _prediction(batch)
    prediction.latents = [torch.randn(1, 8) for _ in range(3)]
    result = _loss(prediction, batch)
    for index in range(3):
        for suffix in ("mean", "std", "min", "max", "absmax", "norm", "finite"):
            key = f"latent_k{index}_{suffix}"
            assert key in result.diagnostics, key
    assert result.diagnostics["latent_k0_finite"] == pytest.approx(1.0)


def test_target_hold_diagnostics_reflect_the_targets() -> None:
    policies = [one_hot(0), one_hot(0), one_hot(4)]
    batch = _batch(rewards=[0.0] * 3, root_policies=policies, unroll=2)
    result = _loss(_prediction(batch), batch)
    assert result.valid_policy_targets == 3
    assert result.diagnostics["target_argmax_hold_fraction"] == pytest.approx(2 / 3)
    assert result.diagnostics["target_group_hold"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_loss_config_validation() -> None:
    with pytest.raises(ValueError):
        LossConfig(policy_loss_weight=-1.0)
    with pytest.raises(ValueError):
        LossConfig(value_scale=0.0)
    with pytest.raises(ValueError):
        LossConfig(scalar_loss="nope")
    with pytest.raises(ValueError):
        LossConfig(huber_delta=0.0)
    with pytest.raises(ValueError):
        LossConfig(support_epsilon=-1.0)


def test_for_model_copies_the_decoding_scales() -> None:
    from forexmind.muzero import MuZeroConfig

    model_config = MuZeroConfig(
        obs_dim=8, value_scale=0.25, reward_scale=0.125, value_epsilon=0.0, reward_epsilon=0.0
    )
    config = LossConfig.for_model(model_config, policy_loss_weight=2.0)
    assert config.value_scale == pytest.approx(0.25)
    assert config.reward_scale == pytest.approx(0.125)
    assert config.policy_loss_weight == pytest.approx(2.0)


def test_for_model_rejects_mismatched_epsilons() -> None:
    from forexmind.muzero import MuZeroConfig

    model_config = MuZeroConfig(obs_dim=8, value_epsilon=0.0, reward_epsilon=0.001)
    with pytest.raises(ValueError, match="epsilon"):
        LossConfig.for_model(model_config)
