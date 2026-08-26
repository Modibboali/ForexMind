"""Action-space adapters (Phase 2).

The environment already accepts both discrete indices (0..4) and continuous
target exposures in [-1, +1].  These adapters make the mapping explicit so RL
agents can consume either representation later (MuZero: discrete; SAC:
continuous) without changing the environment or portfolio logic.
"""

from __future__ import annotations

import numpy as np

from forexmind.environment.actions import (
    TARGET_EXPOSURES,
    Action,
    exposure_from_index,
    index_from_exposure,
)


class DiscreteActionAdapter:
    """Maps a discrete action index in {0..4} to a target exposure.

    0 -> -1.0, 1 -> -0.5, 2 -> 0.0, 3 -> +0.5, 4 -> +1.0
    """

    n_actions: int = len(TARGET_EXPOSURES)

    def decode(self, index: int) -> Action:
        return Action(exposure_from_index(index))

    def encode(self, exposure: float) -> int:
        return index_from_exposure(exposure)

    @property
    def action_values(self) -> np.ndarray:
        return np.asarray(TARGET_EXPOSURES, dtype=np.float32)


class ContinuousActionAdapter:
    """Maps a continuous target exposure in [-1, +1] to an :class:`Action`."""

    def decode(self, exposure: float) -> Action:
        return Action(float(exposure))
