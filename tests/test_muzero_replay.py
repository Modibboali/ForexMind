"""Stage 4.3 replay-buffer tests: sampling, boundaries, masks, capacity, leakage."""

from __future__ import annotations

import numpy as np
import pytest
from forexmind.muzero import (
    MUZERO_NUM_ACTIONS,
    ReplayConfig,
    TargetConfig,
    TrajectoryReplayBuffer,
)
from forexmind.muzero.actions import FLAT, HOLD, LONG_50
from forexmind.muzero.replay import (
    action_distribution_diagnostics,
    event_distribution_diagnostics,
)

from tests.muzero_synthetic import make_trajectory

K = 4
TARGET_CONFIG = TargetConfig(num_unroll_steps=K, td_steps=3, discount=0.99)


def _two_exchange_trajectories():
    """Two trajectories with distinct reward signatures so leakage is detectable."""
    first = make_trajectory(
        actions=[HOLD] * 6, rewards=[1.0] * 6, root_values=[0.1] * 6, trajectory_id=1
    )
    second = make_trajectory(
        actions=[HOLD] * 4, rewards=[-2.0] * 4, root_values=[0.2] * 4, trajectory_id=2
    )
    return first, second


def _buffer(*trajectories, **overrides) -> TrajectoryReplayBuffer:
    config = ReplayConfig(**{"max_trajectories": 16, **overrides})
    buffer = TrajectoryReplayBuffer(config)
    for trajectory in trajectories:
        buffer.add(trajectory)
    return buffer


def assert_batch_matches_buffer(batch, buffer: TrajectoryReplayBuffer) -> None:
    """Every real value in the batch must come from its own trajectory."""
    by_id = {trajectory.metadata.trajectory_id: trajectory for trajectory in buffer.trajectories}
    for row in range(batch.batch_size):
        trajectory = by_id[int(batch.trajectory_ids[row])]
        position = int(batch.positions[row])
        assert 0 <= position < len(trajectory)
        for k in range(batch.unroll_steps):
            if float(batch.reward_masks[row, k]) == 1.0:
                assert int(batch.actions[row, k]) == int(trajectory.actions[position + k])
                assert float(batch.target_rewards[row, k]) == pytest.approx(
                    float(trajectory.rewards[position + k])
                )
                assert position + k < len(trajectory)
            else:
                assert int(batch.actions[row, k]) == 0
        for k in range(batch.unroll_steps + 1):
            if float(batch.policy_masks[row, k]) == 1.0:
                assert position + k < len(trajectory)
                assert np.allclose(
                    batch.target_policies[row, k].numpy(),
                    trajectory.root_policies[position + k],
                )


# --------------------------------------------------------------------------- #
# §37 replay boundaries
# --------------------------------------------------------------------------- #


def test_samples_never_cross_into_another_trajectory() -> None:
    first, second = _two_exchange_trajectories()
    buffer = _buffer(first, second)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        batch = buffer.sample(32, target_config=TARGET_CONFIG, rng=rng)
        assert_batch_matches_buffer(batch, buffer)
        # the two reward signatures never mix inside one row
        rewards = batch.target_rewards.numpy()
        masks = batch.reward_masks.numpy().astype(bool)
        for row in range(batch.batch_size):
            values = set(np.round(rewards[row][masks[row]], 5).tolist())
            assert values <= {1.0} or values <= {-2.0}


def test_positions_are_bounded_by_their_own_trajectory() -> None:
    first, second = _two_exchange_trajectories()
    buffer = _buffer(first, second)
    rng = np.random.default_rng(0)
    batch = buffer.sample(64, target_config=TARGET_CONFIG, rng=rng)
    lengths = {1: len(first), 2: len(second)}
    for row in range(batch.batch_size):
        trajectory_id = int(batch.trajectory_ids[row])
        position = int(batch.positions[row])
        assert 0 <= position < lengths[trajectory_id]


# --------------------------------------------------------------------------- #
# §38 action-mask integrity
# --------------------------------------------------------------------------- #


def test_sampled_batches_respect_stored_action_masks() -> None:
    trajectory = make_trajectory(
        actions=[LONG_50, HOLD, HOLD, FLAT, HOLD, LONG_50], rewards=[0.0] * 6
    )
    buffer = _buffer(trajectory)
    batch = buffer.sample(16, target_config=TARGET_CONFIG, rng=np.random.default_rng(3))
    masks = batch.action_masks.numpy()
    assert masks.shape[1:] == (K + 1, MUZERO_NUM_ACTIONS)
    assert masks[:, :, 0].all()  # HOLD always valid
    assert masks.any(axis=-1).all()  # at least one valid action
    for row in range(batch.batch_size):
        for k in range(K + 1):
            if float(batch.policy_masks[row, k]) == 1.0:
                policies = batch.target_policies[row, k].numpy()
                assert policies[~masks[row, k]].sum() == pytest.approx(0.0)
                assert policies.sum() == pytest.approx(1.0)


def test_every_stored_policy_respects_its_mask() -> None:
    trajectory = make_trajectory(actions=[LONG_50, HOLD, FLAT, LONG_50], rewards=[0.0] * 4)
    trajectory.validate()  # already enforced by the contract
    for t in range(len(trajectory)):
        policy = trajectory.root_policies[t]
        mask = trajectory.action_masks[t]
        assert policy[~mask].sum() == pytest.approx(0.0)
        assert mask[int(trajectory.actions[t])]


# --------------------------------------------------------------------------- #
# Uniform sampling behaviour
# --------------------------------------------------------------------------- #


def test_uniform_sampling_is_seeded_reproducible() -> None:
    first, second = _two_exchange_trajectories()
    buffer = _buffer(first, second)
    batch_a = buffer.sample(16, target_config=TARGET_CONFIG, rng=np.random.default_rng(7))
    batch_b = buffer.sample(16, target_config=TARGET_CONFIG, rng=np.random.default_rng(7))
    assert batch_a.trajectory_ids.tolist() == batch_b.trajectory_ids.tolist()
    assert batch_a.positions.tolist() == batch_b.positions.tolist()
    assert np.array_equal(batch_a.target_rewards.numpy(), batch_b.target_rewards.numpy())


def test_uniform_sampling_covers_the_buffer() -> None:
    trajectory = make_trajectory(actions=[HOLD] * 12, rewards=[0.0] * 12)
    buffer = _buffer(trajectory)
    batch = buffer.sample(400, target_config=TARGET_CONFIG, rng=np.random.default_rng(0))
    positions = set(batch.positions.tolist())
    assert positions == set(range(12))


def test_sampling_requires_a_nonempty_buffer() -> None:
    buffer = TrajectoryReplayBuffer(ReplayConfig())
    with pytest.raises(ValueError, match="empty"):
        buffer.sample(1, target_config=TARGET_CONFIG)


def test_sampling_validates_batch_size() -> None:
    trajectory = make_trajectory(actions=[HOLD] * 3, rewards=[0.0] * 3)
    buffer = _buffer(trajectory)
    with pytest.raises(ValueError, match="batch_size"):
        buffer.sample(0, target_config=TARGET_CONFIG)


def test_unknown_sampling_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="sampling strategy"):
        ReplayConfig(sampling="magic")
    trajectory = make_trajectory(actions=[HOLD] * 3, rewards=[0.0] * 3)
    buffer = _buffer(trajectory)
    with pytest.raises(ValueError, match="sampling strategy"):
        buffer.sample(2, target_config=TARGET_CONFIG, strategy="nope")


def test_decision_rich_weight_zero_matches_uniform() -> None:
    trajectory = make_trajectory(actions=[LONG_50, HOLD, HOLD, HOLD, HOLD, FLAT], rewards=[0.0] * 6)
    uniform = _buffer(trajectory, sampling="uniform")
    rich = _buffer(trajectory, sampling="decision_rich", decision_rich_weight=0.0)
    batch_a = uniform.sample(24, target_config=TARGET_CONFIG, rng=np.random.default_rng(1))
    batch_b = rich.sample(24, target_config=TARGET_CONFIG, rng=np.random.default_rng(1))
    assert batch_a.positions.tolist() == batch_b.positions.tolist()


def test_decision_rich_sampling_still_produces_valid_samples() -> None:
    trajectory = make_trajectory(actions=[LONG_50, HOLD, HOLD, HOLD, HOLD, FLAT], rewards=[0.0] * 6)
    buffer = _buffer(trajectory, sampling="decision_rich", decision_rich_weight=0.5)
    batch = buffer.sample(16, target_config=TARGET_CONFIG, rng=np.random.default_rng(2))
    assert_batch_matches_buffer(batch, buffer)
    assert batch.batch_size == 16


# --------------------------------------------------------------------------- #
# §39 HOLD-heavy trajectories
# --------------------------------------------------------------------------- #


def _hold_heavy():
    return make_trajectory(
        actions=[LONG_50, HOLD, HOLD, HOLD, HOLD, FLAT],
        rewards=[0.0] * 6,
        root_values=[0.0] * 6,
    )


def test_hold_heavy_trajectory_preserves_every_hold_state() -> None:
    trajectory = _hold_heavy()
    assert int(trajectory.action_frequencies()[HOLD]) == 4
    buffer = _buffer(trajectory)
    assert buffer.num_transitions == 6
    batch = buffer.sample(200, target_config=TARGET_CONFIG, rng=np.random.default_rng(0))
    # every stored position is sampleable, including the four HOLD positions
    assert set(batch.positions.tolist()) == set(range(6))
    hold_rows = (batch.actions[:, 0] == HOLD).sum().item()
    assert hold_rows > 0


def test_sampling_diagnostics_report_hold_heavy_trajectory() -> None:
    buffer = _buffer(_hold_heavy())
    diagnostics = buffer.sampling_diagnostics()
    actions = diagnostics["actions"]
    assert actions["pct_hold"] == pytest.approx(4 / 6)
    assert actions["pct_long"] == pytest.approx(1 / 6)
    assert actions["pct_flat"] == pytest.approx(1 / 6)
    events = diagnostics["events"]
    assert events["pct_entry"] == pytest.approx(1 / 6)
    assert events["pct_exit"] == pytest.approx(1 / 6)
    assert events["pct_resize"] == pytest.approx(0.0)
    assert diagnostics["sampling_strategy"] == "uniform"
    assert diagnostics["decision_rich_weight"] == 0.0


def test_action_distribution_diagnostics_are_normalized() -> None:
    actions = np.array([HOLD, HOLD, FLAT, LONG_50, 2, 4], dtype=np.int64)
    stats = action_distribution_diagnostics(actions)
    assert stats["pct_hold"] == pytest.approx(2 / 6)
    assert stats["pct_flat"] == pytest.approx(1 / 6)
    assert stats["pct_long"] == pytest.approx(2 / 6)
    assert stats["pct_short"] == pytest.approx(1 / 6)
    assert (
        stats["pct_hold"] + stats["pct_flat"] + stats["pct_long"] + stats["pct_short"]
    ) == pytest.approx(1.0)


def test_event_diagnostics_handles_a_subset_of_positions() -> None:
    trajectory = _hold_heavy()
    subset = event_distribution_diagnostics(trajectory, positions=[0, 5])
    assert subset["pct_entry"] == pytest.approx(0.5)
    assert subset["pct_exit"] == pytest.approx(0.5)
    assert subset["pct_hold"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Capacity + memory
# --------------------------------------------------------------------------- #


def test_fifo_eviction_by_trajectory_count() -> None:
    buffer = _buffer(max_trajectories=2)
    for index in range(4):
        buffer.add(
            make_trajectory(actions=[HOLD] * 3, rewards=[float(index)] * 3, trajectory_id=index),
            validate=False,
        )
    assert buffer.num_trajectories == 2
    assert [t.metadata.trajectory_id for t in buffer.trajectories] == [2, 3]
    batch = buffer.sample(8, target_config=TARGET_CONFIG, rng=np.random.default_rng(0))
    assert set(batch.trajectory_ids.tolist()) <= {2, 3}


def test_fifo_eviction_by_transition_count() -> None:
    buffer = _buffer(max_trajectories=64, max_transitions=7)
    for index in range(4):
        buffer.add(
            make_trajectory(actions=[HOLD] * 3, rewards=[0.0] * 3, trajectory_id=index),
            validate=False,
        )
    assert buffer.num_transitions <= 7
    assert buffer.num_trajectories == 2  # 3 + 3 = 6 <= 7, a third would reach 9


def test_clear_empties_the_buffer() -> None:
    buffer = _buffer(_hold_heavy())
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.num_transitions == 0
    assert buffer.total_positions == 0
    with pytest.raises(ValueError, match="empty"):
        buffer.sample(1, target_config=TARGET_CONFIG)


def test_memory_report_accounts_for_storage() -> None:
    buffer = _buffer(_hold_heavy())
    report = buffer.memory_report()
    assert report["num_trajectories"] == 1
    assert report["num_transitions"] == 6
    assert report["total_bytes"] > 0
    assert report["bytes_per_transition"] == pytest.approx(report["total_bytes"] / 6)
    assert report["bytes_per_trajectory"] == pytest.approx(report["total_bytes"])
    assert report["max_trajectories"] == 16
    assert 0.0 < report["capacity_used_fraction"] <= 1.0


def test_memory_scales_with_observation_dimension() -> None:
    """Observations dominate replay size -- widening them must show up."""
    small = _buffer(make_trajectory(actions=[HOLD] * 4, rewards=[0.0] * 4, obs_dim=4))
    large = _buffer(make_trajectory(actions=[HOLD] * 4, rewards=[0.0] * 4, obs_dim=400))
    assert large.memory_report()["total_bytes"] > small.memory_report()["total_bytes"] * 5


# --------------------------------------------------------------------------- #
# Leakage protection
# --------------------------------------------------------------------------- #


def test_replay_refuses_non_train_trajectories() -> None:
    for split in ("validation", "test"):
        trajectory = make_trajectory(
            actions=[HOLD] * 3, rewards=[0.0] * 3, split=split, trajectory_id=0
        )
        buffer = TrajectoryReplayBuffer(ReplayConfig())
        with pytest.raises(ValueError, match="leakage"):
            buffer.add(trajectory)
        assert len(buffer) == 0


def test_replay_can_accept_other_splits_explicitly() -> None:
    trajectory = make_trajectory(
        actions=[HOLD] * 3, rewards=[0.0] * 3, split="validation", trajectory_id=0
    )
    buffer = TrajectoryReplayBuffer(ReplayConfig())
    buffer.add(trajectory, require_split="validation")
    assert buffer.num_trajectories == 1


def test_replay_validates_trajectories_on_add() -> None:
    trajectory = make_trajectory(actions=[HOLD] * 3, rewards=[0.0] * 3)
    trajectory.rewards[0] = np.nan
    buffer = TrajectoryReplayBuffer(ReplayConfig())
    with pytest.raises(ValueError, match="non-finite"):
        buffer.add(trajectory)


def test_replay_rejects_empty_trajectories() -> None:
    buffer = TrajectoryReplayBuffer(ReplayConfig())
    with pytest.raises(ValueError, match="empty"):
        buffer.add(make_trajectory(actions=[], rewards=[]))
