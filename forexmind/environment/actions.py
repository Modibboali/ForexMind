"""Explicit decisions. Exposure equivalence tolerance is 0.1 percentage point.

Only the nearest exposure within 0.001 is masked. FLAT is redundant only at
zero units, so residual positions can always be closed. HOLD preserves units.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import cast

import numpy as np

ACTION_NAMES = (
    "HOLD",
    "FLAT",
    "SHORT_100",
    "SHORT_75",
    "SHORT_50",
    "SHORT_25",
    "LONG_25",
    "LONG_50",
    "LONG_75",
    "LONG_100",
)
TARGET_EXPOSURES = (None, 0.0, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)
DISCRETE_ACTION_SIZE = len(ACTION_NAMES)
EXPOSURE_TOLERANCE = 0.001


class ActionError(ValueError):
    """Raised for invalid actions."""


@dataclass(frozen=True, slots=True)
class Action:
    """None preserves units exactly; a float means target exposure."""

    target_exposure: float | None

    def __post_init__(self) -> None:
        if self.target_exposure is not None and not -1.0 <= self.target_exposure <= 1.0:
            raise ActionError(f"target exposure out of range: {self.target_exposure}")

    @property
    def is_hold(self) -> bool:
        return self.target_exposure is None


def exposure_from_index(index: int) -> float | None:
    if not isinstance(index, Integral) or isinstance(index, (bool, np.bool_)):
        raise ActionError(f"discrete action must be an int, got {index!r}")
    if not 0 <= index < DISCRETE_ACTION_SIZE:
        raise ActionError(f"discrete action index {index} out of range [0, 10)")
    return TARGET_EXPOSURES[index]


def index_from_exposure(exposure: float) -> int:
    if not -1.0 <= exposure <= 1.0:
        raise ActionError(f"exposure {exposure} out of range")
    return min(
        range(1, DISCRETE_ACTION_SIZE),
        key=lambda i: abs(cast(float, TARGET_EXPOSURES[i]) - exposure),
    )


def resolve_action(action: Action | int | float) -> Action:
    if isinstance(action, Action):
        return action
    if isinstance(action, Integral) and not isinstance(action, (bool, np.bool_)):
        return Action(exposure_from_index(action))
    if isinstance(action, float):
        return Action(action)
    raise ActionError(f"unsupported action type: {type(action).__name__} ({action!r})")


def valid_action_mask(exposure: float, *, is_flat: bool) -> np.ndarray:
    """Causal mask from signed account-currency exposure / current equity."""
    mask = np.ones(DISCRETE_ACTION_SIZE, dtype=bool)
    if is_flat:
        mask[1] = False
    elif np.isfinite(exposure):
        nearest = min(
            range(2, DISCRETE_ACTION_SIZE),
            key=lambda i: abs(cast(float, TARGET_EXPOSURES[i]) - exposure),
        )
        if abs(cast(float, TARGET_EXPOSURES[nearest]) - exposure) <= EXPOSURE_TOLERANCE:
            mask[nearest] = False
    return mask
