"""Stage 4.7 tests: vectorized replay target construction (brief S5-S13).

The packed/vectorized sampler must reproduce the Stage 4.3 per-sample reference
*exactly* — every tensor of the batch, including padding masks, terminal
handling and truncation bootstrapping — and it must never read across a
trajectory boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
from forexmind.muzero.packed_replay import (
    PackedTrajectories,
    build_vectorized_batch,
    sample_reference,
)
from forexmind.muzero.replay import ReplayConfig, TrajectoryReplayBuffer
from forexmind.muzero.targets import TargetConfig, value_target

from tests.muzero_synthetic import make_trajectory

CONFIG = TargetConfig(num_unroll_steps=5, td_steps=5, discount=0.99)
BATCH_FIELDS = (
    "observation",
    "actions",
    "target_rewards",
    "target_values",
    "target_policies",
    "policy_masks",
    "value_masks",
    "reward_masks",
    "action_masks",
    "trajectory_ids",
    "positions",
)


def _mixed_trajectories() -> list:
    """Trajectories covering truncation, true termination, short and long episodes."""
    long_truncated = make_trajectory(
        actions=[4, 0, 0, 0, 0, 0, 0],
        rewards=[0.001 * (k + 1) for k in range(7)],
        root_values=[0.01 * (k + 1) for k in range(7)],
        truncated=[False] * 6 + [True],
        boundary_value=0.123,
        trajectory_id=100,
        observation_tag=1.0,
    )
    # A true terminal: only the final transition may terminate, and a terminal
    # episode must carry boundary_value == 0.
    terminal = make_trajectory(
        actions=[4, 0, 0, 0],
        rewards=[0.002, -0.003, 0.004, 0.005],
        root_values=[0.5, 0.4, 0.3, 0.2],
        terminated=[False, False, False, True],
        truncated=[False, False, False, False],
        boundary_value=0.0,
        trajectory_id=200,
        observation_tag=2.0,
    )
    short = make_trajectory(
        actions=[4, 1],
        rewards=[-0.007, 0.008],
        root_values=[-0.02, 0.03],
        truncated=[False, True],
        boundary_value=-0.456,
        trajectory_id=300,
        observation_tag=3.0,
    )
    medium = make_trajectory(
        actions=[5, 0, 3, 0],
        rewards=[0.011, 0.012, 0.013, 0.014],
        root_values=[0.11, 0.12, 0.13, 0.14],
        truncated=[False, False, False, True],
        boundary_value=0.789,
        trajectory_id=400,
        observation_tag=4.0,
    )
    return [long_truncated, terminal, short, medium]


def _all_pairs(trajectories: list) -> tuple[np.ndarray, np.ndarray]:
    """Every (trajectory, position) pair, so no edge case is skipped."""
    trajectory_index: list[int] = []
    positions: list[int] = []
    for index, trajectory in enumerate(trajectories):
        for position in range(len(trajectory)):
            trajectory_index.append(index)
            positions.append(position)
    return np.asarray(trajectory_index, dtype=np.int64), np.asarray(positions, dtype=np.int64)


def test_vectorized_matches_reference_on_every_position() -> None:
    """Every tensor must agree exactly, for every position of every trajectory."""
    trajectories = _mixed_trajectories()
    packed = PackedTrajectories.from_trajectories(trajectories)
    trajectory_index, positions = _all_pairs(trajectories)
    reference = sample_reference(trajectories, trajectory_index, positions, CONFIG)
    vectorized = build_vectorized_batch(packed, trajectory_index, positions, CONFIG)
    for field in BATCH_FIELDS:
        left = getattr(reference, field)
        right = getattr(vectorized, field)
        assert right.shape == left.shape, field
        assert np.array_equal(right.numpy(), left.numpy()), field
    assert vectorized.splits == reference.splits


@pytest.mark.parametrize("batch_size", [1, 2, 32, 64])
def test_buffer_sampling_backends_agree(batch_size: int) -> None:
    buffer = TrajectoryReplayBuffer(ReplayConfig(max_trajectories=8, seed=0))
    for trajectory in _mixed_trajectories():
        buffer.add(trajectory)
    reference = buffer.sample(
        batch_size, target_config=CONFIG, rng=np.random.default_rng(7), batch_backend="reference"
    )
    vectorized = buffer.sample(
        batch_size, target_config=CONFIG, rng=np.random.default_rng(7), batch_backend="vectorized"
    )
    for field in BATCH_FIELDS:
        assert np.array_equal(
            getattr(vectorized, field).numpy(), getattr(reference, field).numpy()
        ), field


def test_padding_never_crosses_a_trajectory_boundary() -> None:
    """Near-end unrolls must be padded, never continued into the next trajectory."""
    trajectories = _mixed_trajectories()
    packed = PackedTrajectories.from_trajectories(trajectories)
    trajectory_index, positions = _all_pairs(trajectories)
    batch = build_vectorized_batch(packed, trajectory_index, positions, CONFIG)
    lengths = np.asarray([len(t) for t in trajectories], dtype=np.int64)
    k = CONFIG.num_unroll_steps
    for row, (index, position) in enumerate(zip(trajectory_index, positions, strict=True)):
        length = int(lengths[index])
        for offset in range(k):
            local = position + offset
            if local >= length:
                assert int(batch.actions[row, offset]) == 0
                assert float(batch.target_rewards[row, offset]) == 0.0
                assert float(batch.reward_masks[row, offset]) == 0.0
        for offset in range(k + 1):
            local = position + offset
            if local > length:
                assert float(batch.value_masks[row, offset]) == 0.0
                assert float(batch.target_values[row, offset]) == 0.0
                assert float(batch.policy_masks[row, offset]) == 0.0
                assert np.array_equal(
                    batch.action_masks[row, offset].numpy(),
                    np.array([True, False, False, False, False, False]),
                )
            if position + offset == length:
                # The final stored state: a target exists but no policy is trained.
                assert float(batch.value_masks[row, offset]) == 1.0
                assert float(batch.policy_masks[row, offset]) == 0.0


def test_observation_rows_come_from_the_own_trajectory() -> None:
    trajectories = _mixed_trajectories()
    packed = PackedTrajectories.from_trajectories(trajectories)
    trajectory_index, positions = _all_pairs(trajectories)
    batch = build_vectorized_batch(packed, trajectory_index, positions, CONFIG)
    for row, (index, position) in enumerate(zip(trajectory_index, positions, strict=True)):
        expected = trajectories[index].observations[position]
        assert np.array_equal(batch.observation[row].numpy(), expected)
        # The per-trajectory marker (observation column 1) must match too.
        assert batch.observation[row, 1].item() == pytest.approx(float(index + 1))


def test_value_targets_match_the_scalar_reference_semantics() -> None:
    """Terminal and truncation bootstrap rules must survive vectorization."""
    trajectories = _mixed_trajectories()
    packed = PackedTrajectories.from_trajectories(trajectories)
    for index, trajectory in enumerate(trajectories):
        positions = np.arange(len(trajectory) + 1, dtype=np.int64)
        states = np.full(positions.shape, index, dtype=np.int64)
        targets, valid = _state_targets(packed, states, positions)
        for row, position in enumerate(positions.tolist()):
            if position > len(trajectory):
                assert not bool(valid[row])
                continue
            expected = value_target(
                trajectory,
                position,
                td_steps=CONFIG.td_steps,
                discount=CONFIG.discount,
                use_boundary_value=CONFIG.use_boundary_value,
            )
            if position == len(trajectory):
                # Past the last decision: the reference uses the boundary value.
                assert trajectory.truncated[-1] or trajectory.boundary_value == 0.0
            assert targets[row] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def _state_targets(packed: PackedTrajectories, states: np.ndarray, positions: np.ndarray):
    from forexmind.muzero.packed_replay import _value_targets_vectorized

    return _value_targets_vectorized(packed, states, positions, CONFIG)


def test_use_boundary_value_false_zeroes_the_tail_bootstrap() -> None:
    trajectories = _mixed_trajectories()
    packed = PackedTrajectories.from_trajectories(trajectories)
    config = TargetConfig(
        num_unroll_steps=2, td_steps=5, discount=0.99, use_boundary_value=False
    )
    reference = sample_reference(
        trajectories, *_all_pairs(trajectories), config
    )
    vectorized = build_vectorized_batch(
        packed, *_all_pairs(trajectories), config
    )
    assert np.array_equal(
        vectorized.target_values.numpy(), reference.target_values.numpy()
    )


def test_decision_rich_strategy_uses_the_same_pairs() -> None:
    buffer = TrajectoryReplayBuffer(
        ReplayConfig(max_trajectories=8, sampling="decision_rich", decision_rich_weight=0.5, seed=3)
    )
    for trajectory in _mixed_trajectories():
        buffer.add(trajectory)
    reference = buffer.sample(
        16, target_config=CONFIG, rng=np.random.default_rng(11), batch_backend="reference"
    )
    vectorized = buffer.sample(
        16, target_config=CONFIG, rng=np.random.default_rng(11), batch_backend="vectorized"
    )
    assert np.array_equal(
        vectorized.positions.numpy(), reference.positions.numpy()
    )
    assert np.array_equal(
        vectorized.trajectory_ids.numpy(), reference.trajectory_ids.numpy()
    )
    assert np.array_equal(
        vectorized.target_values.numpy(), reference.target_values.numpy()
    )


def test_packed_view_is_rebuilt_after_mutation() -> None:
    buffer = TrajectoryReplayBuffer(ReplayConfig(max_trajectories=2, seed=0))
    trajectories = _mixed_trajectories()
    buffer.add(trajectories[0])
    first = buffer.packed()
    assert first.num_trajectories == 1
    assert first.total_positions == len(trajectories[0])
    buffer.add(trajectories[1])
    second = buffer.packed()
    assert second is not first  # rebuilt after mutation
    assert second.num_trajectories == 2
    buffer.add(trajectories[2])  # capacity 2 -> FIFO eviction
    third = buffer.packed()
    assert third.num_trajectories == 2
    assert third.trajectory_ids.tolist() == [
        trajectories[1].metadata.trajectory_id,
        trajectories[2].metadata.trajectory_id,
    ]
    buffer.clear()
    with pytest.raises(ValueError):
        buffer.packed()
    assert buffer.memory_report()["packed"] is False


def test_reference_backend_is_still_available_and_default_is_vectorized() -> None:
    assert ReplayConfig().batch_backend == "vectorized"
    with pytest.raises(ValueError, match="batch_backend"):
        ReplayConfig(batch_backend="magic")
    buffer = TrajectoryReplayBuffer(ReplayConfig(max_trajectories=4))
    for trajectory in _mixed_trajectories():
        buffer.add(trajectory)
    batch = buffer.sample(8, target_config=CONFIG, rng=np.random.default_rng(0))
    assert batch.batch_size == 8
    assert buffer.memory_report()["packed"] is True
