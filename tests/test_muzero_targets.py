"""Stage 4.3 target-construction tests.

Covers the indexing contract, the n-step return equation, terminal and
truncation bootstrap semantics, policy/reward alignment (the mandatory
off-by-one regression), padding masks, and batch collation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from forexmind.muzero import MUZERO_NUM_ACTIONS, TargetConfig
from forexmind.muzero.actions import FLAT, HOLD, LONG_50, SHORT_50
from forexmind.muzero.targets import (
    PAD_ACTION,
    build_unroll_sample,
    collate_samples,
    state_is_terminal,
    value_target,
)

from tests.muzero_synthetic import make_trajectory, one_hot

# --------------------------------------------------------------------------- #
# §31 indexing
# --------------------------------------------------------------------------- #


def test_unroll_alignment_matches_the_documented_diagram() -> None:
    traj = make_trajectory(
        actions=[HOLD, SHORT_50, FLAT],
        rewards=[1.0, 2.0, 3.0],
        root_policies=[one_hot(0), one_hot(1), one_hot(2)],
        root_values=[10.0, 20.0, 30.0],
        initial_exposure=0.5,
        initial_is_flat=False,
    )
    sample = build_unroll_sample(traj, 0, TargetConfig(num_unroll_steps=3, td_steps=1))
    assert sample.position == 0
    assert sample.actions.tolist() == [HOLD, SHORT_50, FLAT]
    assert sample.target_rewards.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert sample.reward_masks.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert sample.target_values.shape == (4,)
    assert sample.target_policies.shape == (4, MUZERO_NUM_ACTIONS)
    # pi_0 at offset 0, pi_1 after a_0, pi_2 after a_1 -- no shift.
    assert np.argmax(sample.target_policies[0]) == 0
    assert np.argmax(sample.target_policies[1]) == 1
    assert np.argmax(sample.target_policies[2]) == 2
    # The observation is the state at t, not t+1.
    assert sample.observation[0] == 0.0


def test_sample_shapes_for_unroll_length() -> None:
    traj = make_trajectory(actions=[HOLD] * 8, rewards=[0.0] * 8)
    sample = build_unroll_sample(traj, 0, TargetConfig(num_unroll_steps=5))
    assert sample.actions.shape == (5,)
    assert sample.target_rewards.shape == (5,)
    assert sample.reward_masks.shape == (5,)
    assert sample.target_values.shape == (6,)
    assert sample.target_policies.shape == (6, MUZERO_NUM_ACTIONS)
    assert sample.policy_masks.shape == (6,)
    assert sample.value_masks.shape == (6,)
    assert sample.action_masks.shape == (6, MUZERO_NUM_ACTIONS)
    sample.validate()


# --------------------------------------------------------------------------- #
# §32 n-step return
# --------------------------------------------------------------------------- #


def test_n_step_value_target_matches_hand_computation() -> None:
    """gamma = 0.9, r = [1, 2, 3], V3 = 4, td_steps = 3 -> z0 = 8.146."""
    traj = make_trajectory(
        actions=[HOLD, HOLD, HOLD],
        rewards=[1.0, 2.0, 3.0],
        boundary_value=4.0,
    )
    expected = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0 + 0.9**3 * 4.0
    assert expected == pytest.approx(8.146)
    z0 = value_target(traj, 0, td_steps=3, discount=0.9)
    assert z0 == pytest.approx(expected)


def test_n_step_target_uses_stored_search_value_inside_the_trajectory() -> None:
    traj = make_trajectory(
        actions=[HOLD] * 4,
        rewards=[1.0, 2.0, 3.0, 4.0],
        root_values=[0.0, 0.0, 4.0, 0.0],
    )
    # bootstrap index 2 is inside the trajectory, so its stored search value is used
    z0 = value_target(traj, 0, td_steps=2, discount=0.9)
    assert z0 == pytest.approx(1.0 + 0.9 * 2.0 + 0.81 * 4.0)


def test_td_steps_shorter_than_horizon() -> None:
    traj = make_trajectory(
        actions=[HOLD] * 5,
        rewards=[1.0, 2.0, 3.0, 4.0, 5.0],
        root_values=[0.0, 0.0, 7.0, 0.0, 0.0],
    )
    # n = 2 -> r1 + gamma*r2 + gamma^2 * V(2)
    z0 = value_target(traj, 0, td_steps=2, discount=0.5)
    assert z0 == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 7.0)


def test_discounting_is_single_agent_no_sign_flipping() -> None:
    traj = make_trajectory(
        actions=[HOLD] * 3,
        rewards=[1.0, -2.0, 0.0],
        root_values=[0.0, 0.0, 3.0],
    )
    # +gamma at every step: 1 + 0.5*(-2) + 0.25*3
    z0 = value_target(traj, 0, td_steps=2, discount=0.5)
    assert z0 == pytest.approx(1.0 - 1.0 + 0.75)
    z1 = value_target(traj, 1, td_steps=1, discount=0.5)
    assert z1 == pytest.approx(-2.0 + 0.5 * 3.0)


def test_value_target_rejects_out_of_range_position() -> None:
    traj = make_trajectory(actions=[HOLD], rewards=[0.0])
    with pytest.raises(IndexError):
        value_target(traj, 5, td_steps=1, discount=0.99)
    with pytest.raises(IndexError):
        build_unroll_sample(traj, 1, TargetConfig())


# --------------------------------------------------------------------------- #
# §33 true terminal
# --------------------------------------------------------------------------- #


def test_true_terminal_stops_the_sum_and_bootstraps_zero() -> None:
    """o0 -> o1 -> terminal: rewards beyond the terminal are never included."""
    traj = make_trajectory(
        actions=[HOLD, HOLD],
        rewards=[1.0, 2.0],
        terminated=[False, True],
    )
    assert state_is_terminal(traj, 2)
    assert not state_is_terminal(traj, 1)
    # Large td_steps: must stop at the terminal with bootstrap 0.
    z0 = value_target(traj, 0, td_steps=10, discount=0.9)
    assert z0 == pytest.approx(1.0 + 0.9 * 2.0)
    # The terminal state itself is worth exactly 0.
    assert value_target(traj, 2, td_steps=10, discount=0.9) == pytest.approx(0.0)


def test_terminal_policy_target_is_masked_out() -> None:
    traj = make_trajectory(actions=[HOLD, HOLD], rewards=[0.0, 0.0], terminated=[False, True])
    sample = build_unroll_sample(traj, 0, TargetConfig(num_unroll_steps=2))
    # offsets: state 0 (mask 1), state 1 (mask 1), state 2 = terminal (mask 0)
    assert sample.policy_masks.tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert sample.value_masks.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert sample.target_values[2] == pytest.approx(0.0)
    assert sample.target_policies[2].tolist() == pytest.approx([0.0] * MUZERO_NUM_ACTIONS)


# --------------------------------------------------------------------------- #
# §34 truncation
# --------------------------------------------------------------------------- #


def test_truncation_uses_the_recorded_boundary_value_not_zero() -> None:
    traj = make_trajectory(
        actions=[HOLD, HOLD],
        rewards=[1.0, 2.0],
        truncated=[False, True],
        boundary_value=5.0,
    )
    z0 = value_target(traj, 0, td_steps=2, discount=0.5)
    assert z0 == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 5.0)
    # Truncation is not a terminal state.
    assert not state_is_terminal(traj, 2)


def test_boundary_value_can_be_disabled_explicitly() -> None:
    traj = make_trajectory(
        actions=[HOLD, HOLD],
        rewards=[1.0, 2.0],
        truncated=[False, True],
        boundary_value=5.0,
    )
    z0 = value_target(traj, 0, td_steps=2, discount=0.5, use_boundary_value=False)
    assert z0 == pytest.approx(1.0 + 0.5 * 2.0)
    assert TargetConfig(use_boundary_value=False).use_boundary_value is False


# --------------------------------------------------------------------------- #
# §35 policy alignment
# --------------------------------------------------------------------------- #


def test_policy_targets_are_not_shifted() -> None:
    pi0 = one_hot(HOLD)
    pi1 = one_hot(FLAT)
    pi2 = one_hot(SHORT_50)
    traj = make_trajectory(
        actions=[LONG_50, HOLD, HOLD],
        rewards=[0.0, 0.0, 0.0],
        root_policies=[pi0, pi1, pi2],
    )
    sample = build_unroll_sample(traj, 0, TargetConfig(num_unroll_steps=3))
    assert np.array_equal(sample.target_policies[0], pi0)
    assert np.array_equal(sample.target_policies[1], pi1)
    assert np.array_equal(sample.target_policies[2], pi2)


def test_policy_targets_start_from_the_sampled_position() -> None:
    traj = make_trajectory(
        actions=[HOLD, HOLD, HOLD],
        rewards=[0.0, 0.0, 0.0],
        root_policies=[one_hot(0), one_hot(2), one_hot(3)],
    )
    sample = build_unroll_sample(traj, 1, TargetConfig(num_unroll_steps=2))
    assert np.argmax(sample.target_policies[0]) == 2  # pi_1 at offset 0
    assert np.argmax(sample.target_policies[1]) == 3  # pi_2 at offset 1


# --------------------------------------------------------------------------- #
# §36 reward alignment (mandatory regression)
# --------------------------------------------------------------------------- #


def test_reward_targets_align_to_the_recurrent_step() -> None:
    """r1 = 11, r2 = 22, r3 = 33 -> recurrent(a0)=11, recurrent(a1)=22, recurrent(a2)=33."""
    traj = make_trajectory(
        actions=[HOLD, LONG_50, FLAT],
        rewards=[11.0, 22.0, 33.0],
    )
    sample = build_unroll_sample(traj, 0, TargetConfig(num_unroll_steps=3))
    assert sample.actions.tolist() == [HOLD, LONG_50, FLAT]
    assert sample.target_rewards.tolist() == pytest.approx([11.0, 22.0, 33.0])


def test_reward_targets_from_a_nonzero_position() -> None:
    traj = make_trajectory(actions=[HOLD] * 4, rewards=[1.0, 2.0, 3.0, 4.0])
    sample = build_unroll_sample(traj, 1, TargetConfig(num_unroll_steps=3))
    assert sample.target_rewards.tolist() == pytest.approx([2.0, 3.0, 4.0])
    assert sample.actions.tolist() == [0, 0, 0]


# --------------------------------------------------------------------------- #
# §22 padding near the end
# --------------------------------------------------------------------------- #


def test_padding_near_the_end_masks_losses_without_crossing_trajectories() -> None:
    traj = make_trajectory(actions=[HOLD, LONG_50], rewards=[1.0, 2.0])
    sample = build_unroll_sample(traj, 1, TargetConfig(num_unroll_steps=4))
    assert sample.actions.tolist() == [LONG_50, PAD_ACTION, PAD_ACTION, PAD_ACTION]
    assert sample.reward_masks.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert sample.target_rewards.tolist() == pytest.approx([2.0, 0.0, 0.0, 0.0])
    # states 1 and 2 exist; the final observation has no root policy and every
    # later position is masked out of every loss
    assert sample.policy_masks.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])
    assert sample.value_masks.tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0])
    # padded positions still carry a legal (HOLD-only) mask
    assert sample.action_masks[2][0] and not sample.action_masks[2][1:].any()


def test_last_position_sample_is_allowed_and_padded() -> None:
    traj = make_trajectory(actions=[HOLD, HOLD, HOLD], rewards=[0.0, 0.0, 0.0])
    sample = build_unroll_sample(traj, 2, TargetConfig(num_unroll_steps=5))
    assert sample.reward_masks.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])
    assert sample.policy_masks.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert sample.value_masks.tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])


def test_padding_values_are_zero_not_invented() -> None:
    traj = make_trajectory(actions=[HOLD], rewards=[5.0], root_values=[3.0])
    sample = build_unroll_sample(traj, 0, TargetConfig(num_unroll_steps=3))
    assert sample.target_rewards[1:].tolist() == [0.0, 0.0]
    assert sample.target_policies[2:].tolist() == [[0.0] * MUZERO_NUM_ACTIONS] * 2


# --------------------------------------------------------------------------- #
# Config + collation
# --------------------------------------------------------------------------- #


def test_target_config_validates() -> None:
    with pytest.raises(ValueError):
        TargetConfig(num_unroll_steps=0)
    with pytest.raises(ValueError):
        TargetConfig(td_steps=0)
    with pytest.raises(ValueError):
        TargetConfig(discount=0.0)


def test_collate_builds_documented_batch_shapes() -> None:
    traj = make_trajectory(actions=[HOLD] * 6, rewards=[0.1] * 6, obs_dim=5)
    config = TargetConfig(num_unroll_steps=4, td_steps=3)
    samples = [build_unroll_sample(traj, t, config) for t in (0, 1, 2)]
    batch = collate_samples(samples)
    assert batch.batch_size == 3
    assert batch.unroll_steps == 4
    assert batch.observation.shape == (3, 5)
    assert batch.actions.shape == (3, 4)
    assert batch.target_rewards.shape == (3, 4)
    assert batch.target_values.shape == (3, 5)
    assert batch.target_policies.shape == (3, 5, MUZERO_NUM_ACTIONS)
    assert batch.policy_masks.shape == (3, 5)
    assert batch.value_masks.shape == (3, 5)
    assert batch.reward_masks.shape == (3, 4)
    assert batch.action_masks.shape == (3, 5, MUZERO_NUM_ACTIONS)
    assert batch.actions.dtype == torch.int64
    assert batch.action_masks.dtype == torch.bool
    assert torch.isfinite(batch.target_values).all()
    assert torch.isfinite(batch.target_policies).all()


def test_collate_requires_a_uniform_unroll_length() -> None:
    traj = make_trajectory(actions=[HOLD] * 4, rewards=[0.0] * 4)
    short = build_unroll_sample(traj, 0, TargetConfig(num_unroll_steps=2))
    long = build_unroll_sample(traj, 0, TargetConfig(num_unroll_steps=3))
    with pytest.raises(ValueError, match="unroll length"):
        collate_samples([short, long])


def test_collate_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_samples([])


def test_batch_to_moves_device() -> None:
    traj = make_trajectory(actions=[HOLD] * 3, rewards=[0.0] * 3, obs_dim=2)
    batch = collate_samples([build_unroll_sample(traj, 0, TargetConfig())])
    moved = batch.to("cpu")
    assert moved.observation.device.type == "cpu"
