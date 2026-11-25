"""Stage 4.2 tests for MuZero's six-action space and planning state.

Covers the frozen six-action definition, the projection from the shared
ten-action environment mask, and the deterministic planning-state transitions
that produce imagined action masks.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from forexmind.config import PositionSizingConfig
from forexmind.environment import ForexEnvironment
from forexmind.environment.actions import ACTION_NAMES, TARGET_EXPOSURES, valid_action_mask
from forexmind.muzero.actions import (
    FLAT,
    HOLD,
    LONG_50,
    LONG_100,
    MUZERO_ACTION_NAMES,
    MUZERO_ENV_ACTION_INDICES,
    MUZERO_NUM_ACTIONS,
    MUZERO_TARGET_EXPOSURES,
    SHORT_50,
    SHORT_100,
    PlanningState,
    env_action_index,
    mu_zero_action_index,
    project_action_mask,
)

from tests.test_environment import _config, _dataset


def _make_env(pair: str = "EURUSD", price: float = 1.1) -> ForexEnvironment:
    config = replace(
        _config(close_at_episode_end=False), sizing=PositionSizingConfig(mode="equity_fraction")
    )
    env = ForexEnvironment(_dataset(instrument=pair, price=price), config)
    env.reset(start_index=0, horizon=8)
    return env


# --------------------------------------------------------------------------- #
# Frozen six-action space
# --------------------------------------------------------------------------- #


def test_six_action_space_matches_the_frozen_definition() -> None:
    assert MUZERO_NUM_ACTIONS == 6
    assert MUZERO_ACTION_NAMES == ("HOLD", "FLAT", "SHORT_100", "SHORT_50", "LONG_50", "LONG_100")
    assert MUZERO_TARGET_EXPOSURES == (None, 0.0, -1.0, -0.5, 0.5, 1.0)


def test_plus_minus_25_and_75_are_not_reachable() -> None:
    assert "SHORT_75" not in MUZERO_ACTION_NAMES
    assert "LONG_75" not in MUZERO_ACTION_NAMES
    assert "SHORT_25" not in MUZERO_ACTION_NAMES
    assert "LONG_25" not in MUZERO_ACTION_NAMES
    for exposure in MUZERO_TARGET_EXPOSURES:
        assert exposure not in (-0.75, -0.25, 0.25, 0.75)


def test_action_names_and_exposures_derive_from_the_environment() -> None:
    for position, env_index in enumerate(MUZERO_ENV_ACTION_INDICES):
        assert MUZERO_ACTION_NAMES[position] == ACTION_NAMES[env_index]
        assert MUZERO_TARGET_EXPOSURES[position] == TARGET_EXPOSURES[env_index]


def test_environment_index_mapping_is_the_expected_subset() -> None:
    assert MUZERO_ENV_ACTION_INDICES == (0, 1, 2, 4, 7, 9)
    assert [env_action_index(i) for i in range(MUZERO_NUM_ACTIONS)] == list(
        MUZERO_ENV_ACTION_INDICES
    )
    assert [mu_zero_action_index(i) for i in MUZERO_ENV_ACTION_INDICES] == list(
        range(MUZERO_NUM_ACTIONS)
    )
    assert mu_zero_action_index(3) is None  # SHORT_75
    assert mu_zero_action_index(6) is None  # LONG_25


def test_index_helpers_validate_range() -> None:
    with pytest.raises(ValueError):
        env_action_index(6)
    with pytest.raises(ValueError):
        env_action_index(-1)


def test_named_constants_line_up() -> None:
    assert (HOLD, FLAT, SHORT_100, SHORT_50, LONG_50, LONG_100) == (0, 1, 2, 3, 4, 5)


# --------------------------------------------------------------------------- #
# Environment mask projection
# --------------------------------------------------------------------------- #


def test_project_accepts_ten_wide_and_six_wide_masks() -> None:
    env_mask = np.ones(10, dtype=bool)
    env_mask[4] = False  # SHORT_50 in the environment space
    projected = project_action_mask(env_mask)
    assert projected.shape == (6,)
    assert not projected[SHORT_50]
    assert projected.sum() == 5

    already = np.ones(6, dtype=bool)
    already[FLAT] = False
    assert np.array_equal(project_action_mask(already), already)


def test_project_ignores_reductions_it_does_not_represent() -> None:
    env_mask = np.ones(10, dtype=bool)
    env_mask[3] = False  # SHORT_75 has no MuZero equivalent
    projected = project_action_mask(env_mask)
    assert projected.all()


def test_project_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="action mask must have"):
        project_action_mask(np.ones(7, dtype=bool))


# --------------------------------------------------------------------------- #
# Planning state
# --------------------------------------------------------------------------- #


def test_flat_state_masks_flat_and_keeps_hold() -> None:
    mask = PlanningState.flat().action_mask()
    assert not mask[FLAT]
    assert mask[HOLD]
    assert mask.sum() == 5


def test_hold_preserves_planning_state() -> None:
    state = PlanningState(exposure=0.5, is_flat=False)
    assert state.after(HOLD) == state
    assert state.after(HOLD).action_mask().tolist() == state.action_mask().tolist()


def test_flat_action_zeroes_the_planning_state() -> None:
    state = PlanningState(exposure=-1.0, is_flat=False)
    after = state.after(FLAT)
    assert after == PlanningState.flat()
    assert not after.action_mask()[FLAT]


def test_exposure_actions_set_the_target_exposure() -> None:
    cases = {
        SHORT_100: -1.0,
        SHORT_50: -0.5,
        LONG_50: 0.5,
        LONG_100: 1.0,
    }
    for action, target in cases.items():
        state = PlanningState.flat().after(action)
        assert state.exposure == pytest.approx(target)
        assert state.is_flat is False
        assert not state.action_mask()[action]  # its own target is now redundant
        assert state.action_mask()[HOLD]


def test_planning_state_after_rejects_bad_action() -> None:
    with pytest.raises(ValueError):
        PlanningState.flat().after(MUZERO_NUM_ACTIONS)


def test_planning_state_records_position_changes_monotonically() -> None:
    state = PlanningState.flat()
    state = state.after(LONG_100)
    assert state.action_mask().tolist() == [True, True, True, True, True, False]
    state = state.after(LONG_50)
    assert state.action_mask().tolist() == [True, True, True, True, False, True]
    state = state.after(FLAT)
    assert state.action_mask().tolist() == [True, False, True, True, True, True]


@pytest.mark.parametrize("exposure", [0.0, -1.0, -0.5, 0.5, 1.0, 0.37, -0.62])
def test_from_action_mask_round_trips(exposure: float) -> None:
    state = PlanningState.from_exposure(exposure)
    mask = state.action_mask()
    assert np.array_equal(PlanningState.from_action_mask(mask).action_mask(), mask)


def test_from_action_mask_requires_hold() -> None:
    mask = np.ones(6, dtype=bool)
    mask[HOLD] = False
    with pytest.raises(ValueError, match="HOLD"):
        PlanningState.from_action_mask(mask)


def test_from_action_mask_rejects_multiple_redundant_targets() -> None:
    mask = np.ones(6, dtype=bool)
    mask[SHORT_50] = False
    mask[LONG_50] = False
    with pytest.raises(ValueError, match="more than one"):
        PlanningState.from_action_mask(mask)


def test_valid_actions_lists_only_valid_indices() -> None:
    assert PlanningState.flat().valid_actions() == [HOLD, SHORT_100, SHORT_50, LONG_50, LONG_100]


# --------------------------------------------------------------------------- #
# Parity with the real environment
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "steps",
    [
        (),
        (HOLD,),
        (FLAT,),
        (SHORT_100,),
        (SHORT_50,),
        (LONG_50,),
        (LONG_100,),
        (SHORT_100, 0),
        (0, 0),
    ],
)
def test_planning_state_mask_matches_the_environment_mask(steps: tuple[int, ...]) -> None:
    env = _make_env()
    for mu_zero_action in steps:
        env.step(env_action_index(mu_zero_action))
    env_mask = np.asarray(env.action_masks(), dtype=bool)
    assert np.array_equal(project_action_mask(env_mask), PlanningState.from_env(env).action_mask())


def test_planning_state_from_env_mirrors_the_environment_rule() -> None:
    env = _make_env()
    env.step(env_action_index(LONG_100))
    state = PlanningState.from_env(env)
    assert state.exposure == pytest.approx(1.0)
    assert state.is_flat is False
    snapshot = env.portfolio.snapshot()
    assert np.array_equal(
        state.action_mask(), project_action_mask(valid_action_mask(1.0, is_flat=False))
    )
    assert snapshot.position.units > 0


def test_planning_state_from_env_requires_a_reset_environment() -> None:
    config = replace(
        _config(close_at_episode_end=False), sizing=PositionSizingConfig(mode="equity_fraction")
    )
    env = ForexEnvironment(_dataset(instrument="EURUSD", price=1.1), config)
    with pytest.raises(ValueError, match="reset"):
        PlanningState.from_env(env)
