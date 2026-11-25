"""MuZero learner tests (Stage 4.4 §14-§38).

Covers the joint optimizer, multi-step gradient flow (including the anti-detach
regression), overfitting on controlled data, the reward/value/policy sanity
tasks, split guarding, numerical safety, and checkpoint/resume.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from forexmind.muzero import (
    LearnerConfig,
    MuZeroConfig,
    MuZeroLearner,
    OptimizerConfig,
    ReplayConfig,
    TargetConfig,
    TrajectoryReplayBuffer,
    build_muzero_network,
)
from forexmind.muzero.losses import LossConfig, muzero_losses
from forexmind.muzero.targets import build_unroll_sample, collate_samples
from forexmind.training.numerics import FiniteError

from tests.muzero_synthetic import make_trajectory, one_hot

OBS_DIM = 8
LATENT = 8
SUPPORT = 31
VALUE_SCALE = 0.5
REWARD_SCALE = 0.2
UNROLL = 5
TD_STEPS = 3

TARGET_CONFIG = TargetConfig(num_unroll_steps=UNROLL, td_steps=TD_STEPS, discount=0.99)

#: The long-running overfit/sanity tests use a shorter unroll and a larger
#: learning rate so they converge within a few hundred CPU updates.
SANITY_UNROLL = 3
SANITY_TARGET_CONFIG = TargetConfig(
    num_unroll_steps=SANITY_UNROLL, td_steps=TD_STEPS, discount=0.99
)
SANITY_OPTIMIZER = OptimizerConfig(learning_rate=1e-2, max_grad_norm=10.0)


def _model(*, use_support: bool = True, seed: int = 0, **overrides):
    torch.manual_seed(seed)
    base: dict = dict(
        obs_dim=OBS_DIM,
        latent_dim=LATENT,
        hidden_dim=16,
        num_layers=1,
        action_embedding_dim=4,
        use_support=use_support,
        value_support_size=SUPPORT,
        reward_support_size=SUPPORT,
        value_scale=VALUE_SCALE,
        reward_scale=REWARD_SCALE,
    )
    base.update(overrides)
    model = build_muzero_network(MuZeroConfig(**base))
    model.train()
    return model


def _learner(model=None, *, use_support: bool = True, sanity: bool = False, **overrides):
    model = model if model is not None else _model(use_support=use_support)
    if sanity:
        overrides.setdefault("optimizer", SANITY_OPTIMIZER)
    config = LearnerConfig.for_model(model.config, **overrides)
    return MuZeroLearner(model, config)


def _replay(trajectories, *, seed: int = 0) -> TrajectoryReplayBuffer:
    buffer = TrajectoryReplayBuffer(ReplayConfig(max_trajectories=64, seed=seed))
    for trajectory in trajectories:
        buffer.add(trajectory)
    return buffer


def _sample(buffer, batch_size=4, *, seed=0, target_config=None):
    return buffer.sample(
        batch_size, target_config=target_config or TARGET_CONFIG, rng=np.random.default_rng(seed)
    )


def _sample_sanity(buffer, batch_size=4, *, seed=0):
    return _sample(buffer, batch_size, seed=seed, target_config=SANITY_TARGET_CONFIG)


def _controlled_batch(trajectories, *, unroll: int = SANITY_UNROLL):
    """One row per trajectory, taken at position 0.

    The sanity tests need to know exactly which (trajectory, position) pairs
    they train on, so they bypass replay sampling here (replay sampling is
    covered by the overfit tests and the Stage 4.3 suite).
    """
    config = TargetConfig(num_unroll_steps=unroll, td_steps=TD_STEPS, discount=0.99)
    samples = [build_unroll_sample(trajectory, 0, config) for trajectory in trajectories]
    return collate_samples(samples)


def _varying_trajectories(count: int = 8, *, steps: int = 8):
    """Trajectories with distinct observation tags, actions, rewards and policies."""
    trajectories = []
    for index in range(count):
        sign = 1.0 if index % 2 == 0 else -1.0
        rewards = [sign * (0.05 + 0.01 * index)] * steps
        policies = [one_hot(index % 6)] * steps
        trajectories.append(
            make_trajectory(
                actions=[0] * steps,
                rewards=rewards,
                root_policies=policies,
                root_values=[sign * (0.1 + 0.01 * index)] * steps,
                obs_dim=OBS_DIM,
                trajectory_id=index,
                observation_tag=float(index),
                initial_is_flat=False,
            )
        )
    return trajectories


# --------------------------------------------------------------------------- #
# §16-§19 optimizer structure
# --------------------------------------------------------------------------- #


def test_learner_is_one_joint_optimizer_over_the_whole_model() -> None:
    learner = _learner()
    assert len(learner.optimizer.param_groups) == 1
    optimized = {id(p) for group in learner.optimizer.param_groups for p in group["params"]}
    for module in (learner.model.representation, learner.model.dynamics, learner.model.prediction):
        for parameter in module.parameters():
            assert id(parameter) in optimized
    assert isinstance(learner.optimizer, torch.optim.AdamW)


def test_learner_configures_the_optimizer_from_config() -> None:
    optimizer = OptimizerConfig(
        name="adam", learning_rate=1e-4, weight_decay=0.01, betas=(0.8, 0.9), eps=1e-6
    )
    learner = _learner(optimizer=optimizer)
    group = learner.optimizer.param_groups[0]
    assert isinstance(learner.optimizer, torch.optim.Adam)
    assert group["lr"] == pytest.approx(1e-4)
    assert group["weight_decay"] == pytest.approx(0.01)
    assert group["betas"] == (0.8, 0.9)
    assert group["eps"] == pytest.approx(1e-6)


def test_no_target_network_is_created() -> None:
    learner = _learner()
    names = [name for name, _ in learner.model.named_modules()]
    assert not any("target" in name for name in names)


def test_optimizer_config_validation() -> None:
    with pytest.raises(ValueError):
        OptimizerConfig(name="sgd")
    with pytest.raises(ValueError):
        OptimizerConfig(learning_rate=0.0)
    with pytest.raises(ValueError):
        OptimizerConfig(betas=(1.0, 0.9))
    with pytest.raises(ValueError):
        OptimizerConfig(schedule="cosine")
    with pytest.raises(ValueError):
        OptimizerConfig(schedule="step", lr_gamma=1.0)


# --------------------------------------------------------------------------- #
# §20 / §21 / §37 metrics
# --------------------------------------------------------------------------- #


def test_train_step_returns_the_required_metrics() -> None:
    learner = _learner()
    batch = _sample(_replay(_varying_trajectories(4)))
    metrics = learner.train_step(batch)
    required = {
        "total_loss",
        "policy_loss",
        "value_loss",
        "reward_loss",
        "gradient_norm",
        "learning_rate",
        "batch_size",
        "valid_policy_targets",
        "valid_value_targets",
        "valid_reward_targets",
        "policy_kl",
        "policy_entropy",
        "target_policy_entropy",
        "value_mae",
        "reward_mae",
        "pred_hold_prob",
        "target_hold_prob",
        "pred_argmax_hold_fraction",
        "target_argmax_hold_fraction",
        "latent_k0_norm",
        "latent_k5_norm",
        "representation",
    }
    assert required <= set(metrics)
    for key in required - {"representation"}:
        assert np.isfinite(metrics[key]), key
    assert metrics["batch_size"] == 4
    assert metrics["unroll_steps"] == UNROLL
    assert metrics["representation"] == "support"
    assert learner.update_count == 1


def test_gradient_norm_is_measured_before_clipping() -> None:
    learner = _learner(optimizer=OptimizerConfig(max_grad_norm=1e-6))
    batch = _sample(_replay(_varying_trajectories(4)))
    metrics = learner.train_step(batch)
    assert metrics["gradient_norm"] > 1e-6


def test_metrics_summary_reports_trailing_means() -> None:
    learner = _leaner_with_history()
    summary = learner.metrics_summary()
    assert summary["updates"] == 3
    assert "mean_total_loss" in summary and "last_total_loss" in summary


def _leaner_with_history() -> MuZeroLearner:
    learner = _learner()
    batch = _sample(_replay(_varying_trajectories(4)))
    for _ in range(3):
        learner.train_step(batch)
    return learner


# --------------------------------------------------------------------------- #
# §14 / §31 multi-step gradients
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gate", [1.0, 0.5])
def test_multi_step_unroll_gives_every_component_a_finite_gradient(gate: float) -> None:
    learner = _learner(latent_gradient_scale=gate)
    batch = _sample(_replay(_varying_trajectories(4)))
    learner.train_step(batch)
    expected = {
        "representation": learner.model.representation,
        "dynamics": learner.model.dynamics,
        "prediction": learner.model.prediction,
        "action_embedding": learner.model.dynamics.action_embedding,
        "reward_head": learner.model.dynamics.reward_head,
    }
    for name, module in expected.items():
        for parameter_name, parameter in module.named_parameters():
            assert parameter.grad is not None, f"{name}.{parameter_name} has no gradient"
            assert torch.isfinite(parameter.grad).all(), f"{name}.{parameter_name}"
            assert float(parameter.grad.abs().sum()) > 0.0, f"{name}.{parameter_name} is zero"


def _late_step_only_batch():
    batch = _sample(_replay(_varying_trajectories(4)))
    batch.policy_masks.zero_()
    batch.policy_masks[:, UNROLL] = 1.0  # only the final recurrent step trains
    batch.value_masks.zero_()
    batch.reward_masks.zero_()
    return batch


@pytest.mark.parametrize("gate,expect_gradient", [(1.0, True), (0.0, False)])
def test_late_step_gradient_reaches_the_representation_only_when_not_detached(
    gate: float, expect_gradient: bool
) -> None:
    """A loss that depends only on step K must still reach h_theta."""
    learner = _learner(latent_gradient_scale=gate)
    batch = _late_step_only_batch()
    prediction = learner.unroll(batch)
    loss = muzero_losses(
        prediction,
        batch,
        LossConfig.for_model(learner.model.config),
        use_support=True,
        value_support_size=SUPPORT,
        reward_support_size=SUPPORT,
    )
    learner.model.zero_grad(set_to_none=True)
    loss.total.backward()
    grad = learner.model.representation.proj.weight.grad
    assert grad is not None
    norm = float(grad.abs().sum())
    if expect_gradient:
        assert norm > 0.0
    else:
        assert norm == 0.0


def test_latent_gradient_scaling_reduces_the_gradient_norm() -> None:
    norms = {}
    for gate in (1.0, 0.0):
        learner = _learner(latent_gradient_scale=gate)
        batch = _sample(_replay(_varying_trajectories(4)))
        metrics = learner.train_step(batch)
        norms[gate] = metrics["gradient_norm"]
    assert norms[0.0] < norms[1.0]


def test_latent_gradient_scale_is_validated() -> None:
    with pytest.raises(ValueError):
        LearnerConfig(latent_gradient_scale=1.5)
    with pytest.raises(ValueError):
        LearnerConfig(latent_gradient_scale=-0.1)


def test_latents_are_finite_across_the_whole_unroll() -> None:
    learner = _learner()
    batch = _sample(_replay(_varying_trajectories(4)))
    metrics = learner.train_step(batch)
    for index in range(UNROLL + 1):
        assert metrics[f"latent_k{index}_finite"] == pytest.approx(1.0)
        assert np.isfinite(metrics[f"latent_k{index}_norm"])
        assert np.isfinite(metrics[f"latent_k{index}_min"])
        assert np.isfinite(metrics[f"latent_k{index}_max"])


# --------------------------------------------------------------------------- #
# §27 one-batch overfit
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("use_support", [True, False])
def test_single_batch_overfit(use_support: bool) -> None:
    learner = _learner(use_support=use_support, sanity=True)
    batch = _sample_sanity(_replay(_varying_trajectories(6)), batch_size=6)
    first = learner.train_step(batch)
    for _ in range(149):
        last = learner.train_step(batch)

    assert np.isfinite(last["total_loss"])
    assert last["total_loss"] < first["total_loss"]
    assert last["policy_kl"] < first["policy_kl"]
    assert last["reward_mae"] < first["reward_mae"]
    assert last["value_mae"] < first["value_mae"]
    assert last["policy_loss"] < first["policy_loss"] * 0.5


# --------------------------------------------------------------------------- #
# §26 tiny synthetic replay overfit
# --------------------------------------------------------------------------- #


def test_overfits_a_tiny_synthetic_replay() -> None:
    replay = _replay(_varying_trajectories(8))
    learner = _learner_for_replay(replay)
    results = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        batch = replay.sample(8, target_config=SANITY_TARGET_CONFIG, rng=rng)
        results.append(learner.train_step(batch))
    first, last = results[0], results[-1]
    assert last["total_loss"] < first["total_loss"] * 0.5
    assert last["policy_loss"] < first["policy_loss"] * 0.5
    assert last["value_mae"] < first["value_mae"]
    assert last["reward_mae"] < first["reward_mae"]
    assert all(np.isfinite(row["total_loss"]) for row in results)


def _learner_for_replay(replay) -> MuZeroLearner:
    learner = _learner(sanity=True)
    learner.replay = replay
    return learner


# --------------------------------------------------------------------------- #
# §28 reward model sanity
# --------------------------------------------------------------------------- #

ACTION_REWARDS = {0: 0.0, 1: -0.1, 2: -0.2, 3: -0.05, 4: 0.1, 5: 0.2}


def _action_reward_batch():
    trajectories = []
    for action, reward in ACTION_REWARDS.items():
        trajectories.append(
            make_trajectory(
                actions=[action] + [0] * 7,
                rewards=[reward] * 8,
                root_values=[reward] * 8,
                obs_dim=OBS_DIM,
                trajectory_id=action,
                initial_is_flat=False,
            )
        )
    batch = _controlled_batch(trajectories)
    # only the first recurrent step carries a reward target
    batch.reward_masks[:, 1:] = 0.0
    assert batch.actions.shape == (6, SANITY_UNROLL)
    assert sorted(batch.actions[:, 0].tolist()) == [0, 1, 2, 3, 4, 5]
    return batch


@pytest.mark.parametrize("use_support", [True, False])
def test_reward_model_learns_action_dependent_rewards(use_support: bool) -> None:
    learner = _learner(use_support=use_support, sanity=True)
    batch = _action_reward_batch()
    first = learner.train_step(batch)
    for _ in range(199):
        last = learner.train_step(batch)

    assert np.isfinite(last["reward_mae"])
    assert last["reward_mae"] < first["reward_mae"] * 0.25

    prediction = learner.unroll(batch)
    predicted = prediction.reward[:, 0].detach().numpy()
    targets = batch.target_rewards[:, 0].numpy()
    assert np.corrcoef(predicted, targets)[0, 1] > 0.9
    order = np.argsort(predicted)
    assert list(order) == list(np.argsort(targets))


# --------------------------------------------------------------------------- #
# §29 value model sanity
# --------------------------------------------------------------------------- #


def _value_ordering_batch():
    good = make_trajectory(
        actions=[0] * 8,
        rewards=[0.2] * 8,
        obs_dim=OBS_DIM,
        trajectory_id=0,
        observation_tag=1.0,
        initial_is_flat=False,
    )
    bad = make_trajectory(
        actions=[0] * 8,
        rewards=[-0.2] * 8,
        obs_dim=OBS_DIM,
        trajectory_id=1,
        observation_tag=-1.0,
        initial_is_flat=False,
    )
    batch = _controlled_batch([good, bad])
    return batch


def test_value_model_learns_the_target_ordering() -> None:
    learner = _learner(sanity=True)
    batch = _value_ordering_batch()
    initial = learner.train_step(batch)
    for _ in range(199):
        last = learner.train_step(batch)

    assert last["value_mae"] < initial["value_mae"] * 0.5
    prediction = learner.unroll(batch)
    values = prediction.value[:, 0].detach().numpy()
    targets = batch.target_values[:, 0].numpy()
    assert targets[0] > targets[1]
    assert values[0] > values[1]
    assert np.corrcoef(values, targets)[0, 1] > 0.99


# --------------------------------------------------------------------------- #
# §30 policy model sanity
# --------------------------------------------------------------------------- #


def _policy_family_batch():
    policies = [
        one_hot(0),  # one-hot
        np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # multimodal
        np.full(6, 1.0 / 6.0, dtype=np.float32),  # maximum entropy
        np.array([0.94, 0.01, 0.01, 0.01, 0.01, 0.02], dtype=np.float32),  # low entropy
    ]
    trajectories = []
    for index, policy in enumerate(policies):
        trajectories.append(
            make_trajectory(
                actions=[0] * 8,
                rewards=[0.0] * 8,
                root_policies=[policy] * 8,
                obs_dim=OBS_DIM,
                trajectory_id=index,
                observation_tag=float(index + 1),
                initial_is_flat=False,
            )
        )
    batch = _controlled_batch(trajectories)
    batch.policy_masks[:, 1:] = 0.0
    batch.value_masks.zero_()
    batch.reward_masks.zero_()
    return batch


@pytest.mark.parametrize("use_support", [True, False])
def test_policy_model_fits_one_hot_multimodal_and_soft_targets(use_support: bool) -> None:
    learner = _learner(use_support=use_support, sanity=True)
    batch = _policy_family_batch()
    first = learner.train_step(batch)
    for _ in range(199):
        last = learner.train_step(batch)
    assert last["policy_kl"] < first["policy_kl"] * 0.25
    assert last["policy_kl"] < 0.1

    prediction = learner.unroll(batch)
    predicted_probs = torch.softmax(prediction.policy_logits[:, 0].detach(), dim=-1)
    largest_gap = (predicted_probs - batch.target_policies[:, 0]).abs().max().item()
    # the head fits one-hot, multimodal, maximum-entropy and low-entropy targets
    assert largest_gap < 0.15


# --------------------------------------------------------------------------- #
# §25 TRAIN-split enforcement
# --------------------------------------------------------------------------- #


def test_learner_refuses_non_train_batches() -> None:
    learner = _learner()
    batch = _sample(_replay(_varying_trajectories(2)))
    assert learner.config.require_train_split is True
    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="non-train"):
            learner.train_step(replace(batch, splits=(split,)))


def test_learner_accepts_explicit_train_splits_argument() -> None:
    learner = _learner()
    batch = _sample(_replay(_varying_trajectories(2)))
    metrics = learner.train_step(batch, splits=("train",) * batch.batch_size)
    assert metrics["batch_size"] == batch.batch_size


def test_split_guard_can_be_disabled_explicitly() -> None:
    learner = _learner(require_train_split=False)
    batch = _sample(_replay(_varying_trajectories(2)))
    metrics = learner.train_step(replace(batch, splits=("validation",)))
    assert np.isfinite(metrics["total_loss"])


# --------------------------------------------------------------------------- #
# §36 numerical safety
# --------------------------------------------------------------------------- #


def test_nan_target_raises_in_finite_check_mode() -> None:
    learner = _learner(finite_check=True)
    batch = _sample(_replay(_varying_trajectories(2)))
    batch.target_values[0, 0] = float("nan")
    with pytest.raises(FiniteError, match="target_values"):
        learner.train_step(batch)


def test_infinite_observation_raises_in_finite_check_mode() -> None:
    learner = _learner(finite_check=True)
    batch = _sample(_replay(_varying_trajectories(2)))
    batch.observation[0, 0] = float("inf")
    with pytest.raises(FiniteError, match="observation"):
        learner.train_step(batch)


def test_finite_check_can_be_disabled() -> None:
    learner = _learner(finite_check=False)
    batch = _sample(_replay(_varying_trajectories(2)))
    metrics = learner.train_step(batch)
    assert np.isfinite(metrics["total_loss"])


# --------------------------------------------------------------------------- #
# §17 learning-rate schedule
# --------------------------------------------------------------------------- #


def test_constant_schedule_keeps_the_learning_rate() -> None:
    learner = _learner(optimizer=OptimizerConfig(learning_rate=1e-3, schedule="constant"))
    batch = _sample(_replay(_varying_trajectories(2)))
    for _ in range(5):
        metrics = learner.train_step(batch)
    assert metrics["learning_rate"] == pytest.approx(1e-3)


def test_step_schedule_decays_the_learning_rate() -> None:
    optimizer = OptimizerConfig(learning_rate=1e-3, schedule="step", lr_step_size=2, lr_gamma=0.5)
    learner = _learner(optimizer=optimizer)
    batch = _sample(_replay(_varying_trajectories(2)))
    rates = [learner.train_step(batch)["learning_rate"] for _ in range(5)]
    assert rates[0] == pytest.approx(1e-3)
    assert rates[2] == pytest.approx(5e-4)
    assert rates[4] == pytest.approx(2.5e-4)


# --------------------------------------------------------------------------- #
# §34 / §35 checkpoint + resume
# --------------------------------------------------------------------------- #


def test_checkpoint_round_trip_and_resume(tmp_path) -> None:
    learner = _learner()
    batch = _sample(_replay(_varying_trajectories(4)))
    for _ in range(3):
        learner.train_step(batch)
    learner.env_steps = 123
    path = learner.save_checkpoint(tmp_path / "muzero.pt", target_config=TARGET_CONFIG)

    assert path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format"] == "forexmind.muzero.learner"
    assert payload["architecture"]["num_actions"] == 6
    assert payload["architecture"]["latent_dim"] == LATENT
    assert payload["architecture"]["reward_value_representation"] == "categorical_support"
    assert payload["target_config"]["num_unroll_steps"] == UNROLL
    assert payload["target_config"]["td_steps"] == TD_STEPS
    assert payload["update_count"] == 3
    assert payload["optimizer_state"] is not None
    assert payload["model_state"] is not None
    assert "torch" in payload["rng"] and "numpy" in payload["rng"]

    resumed = _learner()
    resumed.load_checkpoint(path)
    assert resumed.update_count == 3
    assert resumed.env_steps == 123
    for (name_a, param_a), (name_b, param_b) in zip(
        learner.model.named_parameters(), resumed.model.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.allclose(param_a, param_b)

    # resuming produces the same next update as continuing the original run
    continued = learner.train_step(batch)
    restarted = resumed.train_step(batch)
    assert restarted["total_loss"] == pytest.approx(continued["total_loss"], rel=1e-6)
    assert resumed.update_count == 4


def test_checkpoint_rejects_a_mismatched_architecture(tmp_path) -> None:
    learner = _learner()
    path = learner.save_checkpoint(tmp_path / "muzero.pt")
    other = _learner(_model(seed=1, latent_dim=LATENT * 2))
    with pytest.raises(ValueError, match="architecture"):
        other.load_checkpoint(path)


def test_checkpoint_rejects_a_foreign_file(tmp_path) -> None:
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"hello": "world"}, path)
    with pytest.raises(ValueError, match="not a ForexMind"):
        _learner().load_checkpoint(path)


def test_checkpoint_does_not_embed_the_replay_buffer(tmp_path) -> None:
    replay = _replay(_varying_trajectories(8))
    learner = _learner_for_replay(replay)
    batch = _sample(replay, batch_size=4)
    learner.train_step(batch)
    path = learner.save_checkpoint(tmp_path / "muzero.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["replay_metadata"]["num_trajectories"] == 8
    assert "trajectories" not in payload
    assert "replay" not in payload


# --------------------------------------------------------------------------- #
# §2 contract assertions
# --------------------------------------------------------------------------- #


def test_unroll_rejects_a_malformed_action_tensor() -> None:
    learner = _learner()
    batch = _sample(_replay(_varying_trajectories(2)))
    with pytest.raises(ValueError, match=r"\[B, K\]"):
        learner.unroll(replace(batch, actions=batch.actions.reshape(-1)))


def test_unroll_rejects_a_broadcastable_but_wrong_policy_target() -> None:
    """A [B, A] policy target would broadcast silently against [B, K+1, A]."""
    learner = _learner()
    batch = _sample(_replay(_varying_trajectories(2)))
    malformed = replace(batch, target_policies=batch.target_policies[:, 0, :])
    with pytest.raises(ValueError, match="target_policies shape"):
        learner.unroll(malformed)


def test_unroll_rejects_mismatched_mask_and_target_shapes() -> None:
    learner = _learner()
    batch = _sample(_replay(_varying_trajectories(2)))
    with pytest.raises(ValueError, match="policy_masks shape"):
        learner.unroll(replace(batch, policy_masks=batch.policy_masks[:, :1]))
    with pytest.raises(ValueError, match="target_rewards shape"):
        learner.unroll(replace(batch, target_rewards=batch.target_rewards[:, :1]))
    with pytest.raises(ValueError, match="action_masks shape"):
        learner.unroll(replace(batch, action_masks=batch.action_masks[:, :-1, :]))


def test_unroll_rejects_a_row_count_mismatch_and_float_actions() -> None:
    learner = _learner()
    batch = _sample(_replay(_varying_trajectories(2)))
    with pytest.raises(ValueError, match=r"rows but batch\.observation has"):
        learner.unroll(replace(batch, actions=batch.actions[:-1]))
    with pytest.raises(ValueError, match="integer tensor"):
        learner.unroll(replace(batch, actions=batch.actions.to(torch.float32)))


def test_train_step_validates_the_batch_contract_before_optimizing() -> None:
    """The §2 assertions run before any inference, backward or optimizer step."""
    learner = _learner()
    batch = _sample(_replay(_varying_trajectories(2)))
    before = [p.detach().clone() for p in learner.model.parameters()]
    with pytest.raises(ValueError, match="value_masks shape"):
        learner.train_step(replace(batch, value_masks=batch.value_masks[:, :1]))
    assert learner.update_count == 0
    assert all(
        torch.equal(a, b.detach()) for a, b in zip(before, learner.model.parameters(), strict=True)
    )


def test_scalar_representation_is_reported() -> None:
    learner = _learner(use_support=False)
    batch = _sample(_replay(_varying_trajectories(2)))
    metrics = learner.train_step(batch)
    assert metrics["representation"] == "scalar"
    assert np.isfinite(metrics["total_loss"])


def test_describe_reports_learner_state() -> None:
    learner = _learner()
    text = learner.describe()
    assert "MuZeroLearner" in text
    assert "updates=0" in text
    assert "support" in text
