"""Stage 4.3 collector tests: real-environment trajectory generation + batches."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
import torch
from forexmind.config import (
    EnvironmentConfig,
    ExecutionConfig,
    MarginConfig,
    PositionSizingConfig,
)
from forexmind.muzero import (
    MUZERO_NUM_ACTIONS,
    CollectorConfig,
    MuZeroCollector,
    MuZeroConfig,
    ReplayConfig,
    TargetConfig,
    TrajectoryReplayBuffer,
    build_muzero_network,
)
from forexmind.observation.encoder import EncoderConfig

from tests.synthetic import make_instrument, make_split_dataset, timeline_m5

DATES = [
    "2020-01-06",
    "2020-03-02",
    "2020-06-01",
    "2020-09-07",
    "2020-12-07",
    "2021-03-01",
    "2021-06-07",
    "2021-09-06",
    "2021-12-06",
]


def _dataset():
    return make_split_dataset({"EURUSD": make_instrument("EURUSD", timeline_m5(DATES, per_day=40))})


def _env_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        execution=ExecutionConfig(spread_mode="fixed", spread_value=0.0),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")),
        sizing=PositionSizingConfig(mode="equity_fraction"),
    )


def _encoder_config() -> EncoderConfig:
    return EncoderConfig(context_length=8)


def _model(encoder_config: EncoderConfig, seed: int = 0):
    torch.manual_seed(seed)
    model = build_muzero_network(
        MuZeroConfig(
            obs_dim=encoder_config.spec.encoded_shape[0],
            latent_dim=8,
            hidden_dim=16,
            num_layers=1,
            action_embedding_dim=4,
        )
    )
    model.eval()
    return model


def _collector(
    model=None, *, split: str = "train", dataset=None, seed: int = 0, **overrides
) -> MuZeroCollector:
    encoder_config = _encoder_config()
    config = CollectorConfig(split=split, horizon=6, num_simulations=4, seed=seed, **overrides)
    return MuZeroCollector(
        dataset if dataset is not None else _dataset(),
        _env_config(),
        encoder_config,
        model if model is not None else _model(encoder_config),
        config,
    )


# --------------------------------------------------------------------------- #
# §40 real-environment collection smoke test
# --------------------------------------------------------------------------- #


def test_collects_valid_trajectories_from_the_real_environment() -> None:
    collector = _collector()
    trajectories = collector.collect(2)

    assert len(trajectories) == 2
    for index, trajectory in enumerate(trajectories):
        trajectory.validate()
        assert trajectory.metadata.trajectory_id == index
        assert trajectory.metadata.split == "train"
        assert trajectory.metadata.instrument == "EURUSD"
        assert trajectory.observations.shape == (len(trajectory) + 1, 71)
        assert trajectory.action_masks.shape == (len(trajectory), MUZERO_NUM_ACTIONS)
        assert np.allclose(trajectory.root_policies.sum(axis=1), 1.0, atol=1e-5)
        assert trajectory.action_masks[:, 0].all()
        assert len(trajectory) == 6  # horizon reached
        assert bool(trajectory.truncated[-1]) or bool(trajectory.terminated[-1])

    stats = collector.stats.to_dict()
    assert stats["trajectories"] == 2
    assert stats["env_steps"] == sum(len(t) for t in trajectories)
    assert sum(stats["action_counts"]) == stats["env_steps"]
    assert stats["searches"] >= stats["env_steps"]
    assert stats["recurrent_inference_calls"] == stats["searches"] * 4
    assert np.isfinite(stats["mean_reward"])


def test_collection_stats_account_for_boundary_searches() -> None:
    collector = _collector()
    trajectories = collector.collect(2)
    boundary_searches = sum(
        1 for t in trajectories if bool(t.truncated[-1]) and not bool(t.terminated[-1])
    )
    assert collector.stats.searches == collector.stats.env_steps + boundary_searches
    for trajectory in trajectories:
        if bool(trajectory.terminated[-1]):
            assert trajectory.boundary_value == 0.0
        elif bool(trajectory.truncated[-1]):
            assert np.isfinite(trajectory.boundary_value)


def test_action_frequencies_and_masks_are_internally_consistent() -> None:
    collector = _collector()
    trajectory = collector.collect(1)[0]
    assert trajectory.action_frequencies().sum() == len(trajectory)
    for t in range(len(trajectory)):
        mask = trajectory.action_masks[t]
        assert mask.shape == (MUZERO_NUM_ACTIONS,)
        assert mask[int(trajectory.actions[t])]
        assert trajectory.root_policies[t][~mask].sum() == pytest.approx(0.0)


def test_planning_metadata_matches_the_environment_account() -> None:
    collector = _collector()
    trajectory = collector.collect(1)[0]
    # planning states came from the live account, so masks agree by construction
    for t in range(len(trajectory)):
        assert np.array_equal(
            trajectory.planning_state(t).action_mask(), trajectory.action_masks[t]
        )
    assert "planning_chain_disagreements" in trajectory.extra
    assert trajectory.extra["planning_chain_disagreements"] >= 0


# --------------------------------------------------------------------------- #
# §28 reproducibility
# --------------------------------------------------------------------------- #


def test_collection_is_reproducible_for_fixed_seeds() -> None:
    encoder_config = _encoder_config()
    model = _model(encoder_config)
    first = _collector(model).collect(2)
    second = _collector(model).collect(2)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.actions, b.actions)
        assert np.array_equal(a.rewards, b.rewards)
        assert np.array_equal(a.observations, b.observations)
        assert np.allclose(a.root_policies, b.root_policies)
        assert np.allclose(a.root_values, b.root_values)
        assert a.boundary_value == b.boundary_value


def test_different_seeds_produce_different_trajectories() -> None:
    encoder_config = _encoder_config()
    model = _model(encoder_config)
    first = _collector(model, seed=1).collect(1)[0]
    second = _collector(model, seed=999).collect(1)[0]
    assert not np.array_equal(first.actions, second.actions) or not np.allclose(
        first.root_policies, second.root_policies
    )


def test_evaluation_collection_disables_noise_and_temperature() -> None:
    collector = _collector(training=False, temperature=None)
    assert collector.config.effective_temperature == 0.0
    assert collector.search_config.add_root_noise is False
    first = collector.collect(1)[0]
    second = _collector(training=False).collect(1)[0]
    assert np.array_equal(first.actions, second.actions)
    assert first.metadata.temperature == 0.0
    assert first.metadata.training is False


def test_training_collection_enables_noise_and_temperature() -> None:
    collector = _collector(training=True)
    assert collector.config.effective_temperature == 1.0
    assert collector.search_config.add_root_noise is True
    trajectory = collector.collect(1)[0]
    assert trajectory.metadata.training is True
    assert trajectory.metadata.temperature == 1.0
    assert trajectory.metadata.num_simulations == 4
    assert trajectory.metadata.discount == 0.99
    assert len(trajectory.metadata.model_version) == 16


# --------------------------------------------------------------------------- #
# §30 leakage protection
# --------------------------------------------------------------------------- #


def test_collector_refuses_an_unknown_split() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        CollectorConfig(split="magic")


def test_validation_trajectories_cannot_enter_training_replay() -> None:
    collector = _collector(split="validation")
    trajectories = collector.collect(1)
    assert trajectories[0].metadata.split == "validation"

    replay = TrajectoryReplayBuffer(ReplayConfig())
    with pytest.raises(ValueError, match="leakage"):
        replay.add(trajectories[0])
    assert len(replay) == 0


def test_train_trajectories_enter_training_replay() -> None:
    collector = _collector(split="train")
    replay = TrajectoryReplayBuffer(ReplayConfig(max_trajectories=4))
    collector.collect_into(replay, 2)
    assert replay.num_trajectories == 2
    assert all(t.metadata.split == "train" for t in replay.trajectories)


# --------------------------------------------------------------------------- #
# §41 sample and inspect training batches
# --------------------------------------------------------------------------- #


def test_collected_replay_produces_correctly_shaped_batches() -> None:
    collector = _collector()
    replay = TrajectoryReplayBuffer(ReplayConfig(max_trajectories=4))
    collector.collect_into(replay, 2)

    target_config = TargetConfig(num_unroll_steps=5, td_steps=3, discount=0.99)
    batch = replay.sample(8, target_config=target_config, rng=np.random.default_rng(0))

    assert batch.batch_size == 8
    assert batch.unroll_steps == 5
    assert batch.observation.shape == (8, 71)
    assert batch.actions.shape == (8, 5)
    assert batch.target_rewards.shape == (8, 5)
    assert batch.target_values.shape == (8, 6)
    assert batch.target_policies.shape == (8, 6, MUZERO_NUM_ACTIONS)
    assert batch.policy_masks.shape == (8, 6)
    assert batch.value_masks.shape == (8, 6)
    assert batch.reward_masks.shape == (8, 5)
    assert batch.action_masks.shape == (8, 6, MUZERO_NUM_ACTIONS)

    for tensor in (
        batch.observation,
        batch.target_rewards,
        batch.target_values,
        batch.target_policies,
    ):
        assert torch.isfinite(tensor).all()
    assert batch.observation.dtype == torch.float32
    assert batch.actions.dtype == torch.int64
    assert batch.action_masks.dtype == torch.bool

    # policy targets are normalized and respect their masks
    policies = batch.target_policies.numpy()
    masks = batch.action_masks.numpy()
    policy_masks = batch.policy_masks.numpy().astype(bool)
    assert np.allclose(policies[policy_masks].sum(axis=-1), 1.0, atol=1e-5)
    assert policies[~masks].sum() == pytest.approx(0.0)
    # reward targets are aligned with the unrolled actions
    assert (batch.reward_masks.sum(dim=1) <= batch.unroll_steps).all()
    # batch indices point at real trajectory positions
    for row in range(batch.batch_size):
        trajectory_id = int(batch.trajectory_ids[row])
        position = int(batch.positions[row])
        trajectory = replay.trajectories[trajectory_id]
        assert 0 <= position < len(trajectory)
        if float(batch.reward_masks[row, 0]) == 1.0:
            assert int(batch.actions[row, 0]) == int(trajectory.actions[position])
            assert float(batch.target_rewards[row, 0]) == pytest.approx(
                float(trajectory.rewards[position])
            )


def test_replay_memory_report_for_collected_data() -> None:
    collector = _collector()
    replay = TrajectoryReplayBuffer(ReplayConfig(max_trajectories=4))
    collector.collect_into(replay, 2)
    report = replay.memory_report()
    assert report["num_trajectories"] == 2
    assert report["num_transitions"] == 12
    assert report["bytes_per_transition"] > 71 * 4  # observations dominate
    assert report["total_mb"] < 1.0


def test_collector_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError):
        CollectorConfig(horizon=0)
    with pytest.raises(ValueError):
        CollectorConfig(num_simulations=0)
    with pytest.raises(ValueError):
        CollectorConfig(discount=0.0)
    with pytest.raises(ValueError):
        CollectorConfig(temperature=-1.0)
