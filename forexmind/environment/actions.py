"""Target-exposure action model.

The first action space is *target exposure* (not BUY/SELL/HOLD)::

    -1.0  fully short
    -0.5  half short
     0.0  flat
    +0.5  half long
    +1.0  fully long

The action means "adjust the portfolio to the requested target exposure".
This formulation maps naturally to discrete MuZero actions later and can be
extended to continuous target exposures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Discrete target-exposure set (index order matters: it defines the action id).
TARGET_EXPOSURES: Final[tuple[float, ...]] = (-1.0, -0.5, 0.0, 0.5, 1.0)
DISCRETE_ACTION_SIZE: Final[int] = len(TARGET_EXPOSURES)

_MIN_EXPOSURE: Final[float] = -1.0
_MAX_EXPOSURE: Final[float] = 1.0


class ActionError(ValueError):
    """Raised for invalid actions."""


@dataclass(frozen=True, slots=True)
class Action:
    """A target-exposure action."""

    target_exposure: float

    def __post_init__(self) -> None:
        if not (_MIN_EXPOSURE <= self.target_exposure <= _MAX_EXPOSURE):
            raise ActionError(
                f"target exposure must be in [{_MIN_EXPOSURE}, {_MAX_EXPOSURE}], "
                f"got {self.target_exposure}"
            )


def exposure_from_index(index: int) -> float:
    """Map a discrete action id to a target exposure."""
    if not isinstance(index, int) or isinstance(index, bool):
        raise ActionError(f"discrete action must be an int, got {index!r}")
    if not (0 <= index < DISCRETE_ACTION_SIZE):
        raise ActionError(f"discrete action index {index} out of range [0, {DISCRETE_ACTION_SIZE})")
    return TARGET_EXPOSURES[index]


def index_from_exposure(exposure: float) -> int:
    """Map a target exposure back to the nearest discrete action id."""
    if not (_MIN_EXPOSURE <= exposure <= _MAX_EXPOSURE):
        raise ActionError(f"exposure {exposure} out of range")
    return min(range(DISCRETE_ACTION_SIZE), key=lambda i: abs(TARGET_EXPOSURES[i] - exposure))


def resolve_action(action: int | float) -> Action:
    """Resolve either a discrete index (``int``) or a raw target (``float``)."""
    if isinstance(action, int) and not isinstance(action, bool):
        return Action(exposure_from_index(action))
    if isinstance(action, float):
        return Action(action)
    raise ActionError(f"unsupported action type: {type(action).__name__} ({action!r})")
