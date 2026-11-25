"""Stage 4.3 trajectory-contract tests: indexing, invariants, validation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from forexmind.muzero import MUZERO_NUM_ACTIONS, MuZeroConfig, build_muzero_network
from forexmind.muzero.actions import FLAT, HOLD, LONG_50
from forexmind.muzero.trajectory import model_version

from tests.muzero_synthetic import make_trajectory, one_hot


def test_trajectory_indexing_invariants() -> None:
    """o0 --a0/r1--> o1 --a1/r2--> o2 --a2/r3--> o3"""
    traj = make_trajectory(
        actions=[HOLD, LONG_50, FLAT],
        rewards=[1.0, 2.0, 3.0],
        root_values=[0.1, 0.2, 0.3],
    )
    assert traj.observations.shape[0] == 4  # T + 1
    assert len(traj) == 3
    assert traj.actions.tolist() == [HOLD, LONG_50, FLAT]
    assert traj.rewards.tolist() == [1.0, 2.0, 3.0]
    assert traj.root_values.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert traj.root_policies.shape == (3, MUZERO_NUM_ACTIONS)
    assert traj.action_masks.shape == (3, MUZERO_NUM_ACTIONS)
    assert traj.terminated.shape == (3,)
    assert traj.truncated.shape == (3,)
    assert traj.planning_exposure.shape == (4,)
    traj.validate()


def test_stored_rewards_are_the_real_next_rewards() -> None:
    """``rewards[t]`` must be ``r_{t+1}``, not ``r_t``."""
    traj = make_trajectory(actions=[HOLD, HOLD], rewards=[11.0, 22.0])
    assert float(traj.rewards[0]) == 11.0
    assert float(traj.rewards[1]) == 22.0


def test_observation_zero_is_the_first_state() -> None:
    traj = make_trajectory(actions=[HOLD], rewards=[1.0], obs_dim=3)
    assert traj.observations[0].tolist() == [0.0, 0.0, 0.0]
    assert traj.observations[1][0] == 1.0


def test_empty_trajectory_is_valid_but_has_no_transitions() -> None:
    traj = make_trajectory(actions=[], rewards=[], obs_dim=2)
    assert len(traj) == 0
    traj.validate()


def test_planning_states_are_environment_ground_truth() -> None:
    traj = make_trajectory(actions=[LONG_50, HOLD, FLAT], rewards=[0.0, 0.0, 0.0])
    assert traj.planning_state(0).is_flat
    assert traj.planning_state(1).exposure == pytest.approx(0.5)
    assert traj.planning_state(2).exposure == pytest.approx(0.5)
    assert traj.planning_state(3).is_flat
    assert traj.planning_chain_disagreements() == []


def test_planning_state_matches_stored_masks() -> None:
    traj = make_trajectory(actions=[LONG_50, HOLD], rewards=[0.0, 0.0])
    for t in range(len(traj)):
        assert np.array_equal(traj.planning_state(t).action_mask(), traj.action_masks[t])


def test_action_frequencies_and_event_counts() -> None:
    traj = make_trajectory(
        actions=[LONG_50, HOLD, HOLD, HOLD, HOLD, FLAT],
        rewards=[0.0] * 6,
    )
    counts = traj.action_frequencies()
    assert counts.sum() == 6
    assert counts[HOLD] == 4
    assert counts[LONG_50] == 1
    assert counts[FLAT] == 1
    events = traj.event_counts()
    assert events == {"hold": 4, "flat": 1, "entry": 1, "exit": 1, "resize": 0}


def test_memory_breakdown_sums_to_nbytes() -> None:
    traj = make_trajectory(actions=[HOLD, HOLD], rewards=[0.0, 0.0], obs_dim=8)
    breakdown = traj.memory_breakdown()
    assert sum(breakdown.values()) == traj.nbytes()
    assert breakdown["observations"] == 3 * 8 * 4  # (T+1) * obs_dim * float32


def test_to_dict_is_json_friendly() -> None:
    import json

    traj = make_trajectory(actions=[HOLD, FLAT], rewards=[0.1, -0.2], trajectory_id=7)
    payload = json.loads(json.dumps(traj.to_dict()))
    assert payload["num_transitions"] == 2
    assert payload["metadata"]["trajectory_id"] == 7
    assert payload["metadata"]["split"] == "train"


# --------------------------------------------------------------------------- #
# Validation must fail loudly
# --------------------------------------------------------------------------- #


def test_validate_rejects_policy_that_does_not_sum_to_one() -> None:
    bad = one_hot(HOLD) * 0.5
    traj = make_trajectory(actions=[HOLD], rewards=[0.0], root_policies=[bad])
    with pytest.raises(ValueError, match="sum to 1"):
        traj.validate()


def test_validate_rejects_policy_mass_on_invalid_actions() -> None:
    policy = one_hot(HOLD)
    policy[FLAT] = 0.5
    policy[HOLD] = 0.5
    # FLAT is invalid because the account starts flat.
    traj = make_trajectory(actions=[HOLD], rewards=[0.0], root_policies=[policy])
    with pytest.raises(ValueError, match="invalid actions"):
        traj.validate()


def test_validate_rejects_selected_action_that_is_invalid() -> None:
    traj = make_trajectory(actions=[FLAT], rewards=[0.0])
    with pytest.raises(ValueError, match="not valid under"):
        traj.validate()


def test_validate_rejects_hold_being_invalid() -> None:
    traj = make_trajectory(actions=[HOLD], rewards=[0.0])
    traj.action_masks[0, HOLD] = False
    with pytest.raises(ValueError, match="HOLD"):
        traj.validate()


def test_validate_rejects_non_finite_values() -> None:
    traj = make_trajectory(actions=[HOLD], rewards=[np.nan])
    with pytest.raises(ValueError, match="rewards contain non-finite"):
        traj.validate()


def test_validate_rejects_mass_before_a_terminal() -> None:
    traj = make_trajectory(actions=[HOLD, HOLD], rewards=[0.0, 0.0], terminated=[True, False])
    with pytest.raises(ValueError, match="final stored transition"):
        traj.validate()


def test_validate_rejects_terminal_with_nonzero_boundary_value() -> None:
    traj = make_trajectory(actions=[HOLD], rewards=[0.0], terminated=[True], boundary_value=1.0)
    with pytest.raises(ValueError, match="boundary_value"):
        traj.validate()


def test_validate_rejects_both_terminated_and_truncated() -> None:
    traj = make_trajectory(actions=[HOLD], rewards=[0.0], terminated=[True], truncated=[True])
    with pytest.raises(ValueError, match="both"):
        traj.validate()


def test_validate_rejects_metadata_length_mismatch() -> None:
    traj = make_trajectory(actions=[HOLD], rewards=[0.0])
    object.__setattr__(traj.metadata, "num_steps", 99)
    with pytest.raises(ValueError, match="num_steps"):
        traj.validate()


def test_validate_rejects_mask_dimension_error() -> None:
    traj = make_trajectory(actions=[HOLD], rewards=[0.0])
    traj.action_masks = np.ones((1, MUZERO_NUM_ACTIONS + 1), dtype=bool)
    with pytest.raises(ValueError):
        traj.validate()


# --------------------------------------------------------------------------- #
# model_version
# --------------------------------------------------------------------------- #


def test_model_version_tracks_weights() -> None:
    config = MuZeroConfig(obs_dim=4, latent_dim=4, hidden_dim=8, num_layers=1)
    torch.manual_seed(0)
    first = build_muzero_network(config)
    torch.manual_seed(1)
    second = build_muzero_network(config)
    assert model_version(first) != model_version(second)
    assert model_version(first) == model_version(first)
    assert len(model_version(first)) == 16
