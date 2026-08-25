"""Tests for the target-exposure action model."""

from __future__ import annotations

import pytest
from forexmind.environment.actions import (
    DISCRETE_ACTION_SIZE,
    TARGET_EXPOSURES,
    Action,
    ActionError,
    exposure_from_index,
    index_from_exposure,
    resolve_action,
)


def test_target_exposures_definition() -> None:
    assert TARGET_EXPOSURES == (-1.0, -0.5, 0.0, 0.5, 1.0)
    assert DISCRETE_ACTION_SIZE == 5


def test_exposure_from_index() -> None:
    assert exposure_from_index(0) == -1.0
    assert exposure_from_index(1) == -0.5
    assert exposure_from_index(2) == 0.0
    assert exposure_from_index(3) == 0.5
    assert exposure_from_index(4) == 1.0


def test_exposure_from_index_out_of_range() -> None:
    with pytest.raises(ActionError):
        exposure_from_index(5)
    with pytest.raises(ActionError):
        exposure_from_index(-1)


def test_index_from_exposure() -> None:
    assert index_from_exposure(-1.0) == 0
    assert index_from_exposure(0.0) == 2
    assert index_from_exposure(1.0) == 4
    assert index_from_exposure(0.6) == 3  # nearest to +0.5
    assert index_from_exposure(0.9) == 4  # nearest to +1.0


def test_index_from_exposure_out_of_range() -> None:
    with pytest.raises(ActionError):
        index_from_exposure(1.5)


def test_resolve_action() -> None:
    assert resolve_action(4).target_exposure == 1.0
    assert resolve_action(0).target_exposure == -1.0
    assert resolve_action(0.25).target_exposure == 0.25
    assert resolve_action(-0.5).target_exposure == -0.5


def test_action_validation() -> None:
    Action(1.0)
    with pytest.raises(ActionError):
        Action(1.01)
    with pytest.raises(ActionError):
        Action(-1.01)


def test_resolve_action_rejects_other_types() -> None:
    with pytest.raises(ActionError):
        resolve_action("long")  # type: ignore[arg-type]
    with pytest.raises(ActionError):
        resolve_action(True)  # type: ignore[arg-type]
