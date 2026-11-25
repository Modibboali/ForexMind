"""MuZero's own six-action space (Stage 4.2).

MuZero operates on a reduced, frozen six-action space::

    0 HOLD   1 FLAT   2 SHORT_100   3 SHORT_50   4 LONG_50   5 LONG_100

The shared Forex environment still exposes its full ten-action categorical
space (it also carries ``SHORT_75``/``SHORT_25``/``LONG_25``/``LONG_75`` and is
used by PPO, the evaluator and the existing baselines).  MuZero therefore
*projects* the real environment mask onto its six actions through
:func:`project_action_mask` instead of redefining the environment contract.

Both :data:`MUZERO_ACTION_NAMES` and :data:`MUZERO_TARGET_EXPOSURES` are derived
from :mod:`forexmind.environment.actions`, so the six-action space can never
silently drift from the environment's definitions.

The module also provides :class:`PlanningState`, the deterministic
"current discrete exposure" state that lets search compute imagined action
masks without asking the neural network to rediscover a rule it cannot observe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from forexmind.environment.actions import (
    ACTION_NAMES,
    DISCRETE_ACTION_SIZE,
    TARGET_EXPOSURES,
    valid_action_mask,
)

#: Environment action indices backing MuZero's six actions.
MUZERO_ENV_ACTION_INDICES: tuple[int, ...] = tuple(
    ACTION_NAMES.index(name)
    for name in ("HOLD", "FLAT", "SHORT_100", "SHORT_50", "LONG_50", "LONG_100")
)

#: MuZero action names (derived from the environment definitions, never invented).
MUZERO_ACTION_NAMES: tuple[str, ...] = tuple(ACTION_NAMES[i] for i in MUZERO_ENV_ACTION_INDICES)

#: MuZero target exposures; ``None`` is the HOLD sentinel (preserve units).
MUZERO_TARGET_EXPOSURES: tuple[float | None, ...] = tuple(
    TARGET_EXPOSURES[i] for i in MUZERO_ENV_ACTION_INDICES
)

#: Number of MuZero actions (``num_actions = 6``).
MUZERO_NUM_ACTIONS: int = len(MUZERO_ACTION_NAMES)

HOLD = 0
FLAT = 1
SHORT_100 = 2
SHORT_50 = 3
LONG_50 = 4
LONG_100 = 5


def env_action_index(muzero_action: int) -> int:
    """Map a MuZero action index onto the equivalent environment action index."""
    if not 0 <= muzero_action < MUZERO_NUM_ACTIONS:
        raise ValueError(f"MuZero action {muzero_action} out of range [0, {MUZERO_NUM_ACTIONS})")
    return MUZERO_ENV_ACTION_INDICES[muzero_action]


def mu_zero_action_index(env_index: int) -> int | None:
    """Map an environment action index to MuZero's, or ``None`` if unsupported."""
    try:
        return MUZERO_ENV_ACTION_INDICES.index(int(env_index))
    except ValueError:
        return None


def project_action_mask(action_mask: np.ndarray) -> np.ndarray:
    """Project a ten-wide environment mask onto MuZero's six actions.

    Accepts either the full ``[10]`` environment mask (normal case) or an
    already-projected ``[6]`` mask, which is returned unchanged.  HOLD stays
    valid because it is always valid in the environment contract.
    """
    mask = np.asarray(action_mask, dtype=bool).reshape(-1)
    if mask.shape[0] == MUZERO_NUM_ACTIONS:
        return mask
    if mask.shape[0] != DISCRETE_ACTION_SIZE:
        raise ValueError(
            f"action mask must have {DISCRETE_ACTION_SIZE} (environment) or "
            f"{MUZERO_NUM_ACTIONS} (MuZero) entries, got {mask.shape[0]}"
        )
    return mask[list(MUZERO_ENV_ACTION_INDICES)]


@dataclass(frozen=True, slots=True)
class PlanningState:
    """Deterministic planning state carried alongside the neural latent state.

    The neural latent cannot expose the account's future action-validity state,
    so search tracks it explicitly and deterministically.  This state is *not*
    learned: HOLD preserves it, FLAT zeroes it, and an exposure action sets the
    corresponding target exposure, exactly mirroring the environment rule in
    :func:`forexmind.environment.actions.valid_action_mask`.

    ``exposure`` is the signed account-currency exposure (gross exposure divided
    by equity); ``is_flat`` mirrors the environment's ``units == 0`` flag, which
    masks FLAT.
    """

    exposure: float = 0.0
    is_flat: bool = True

    # -- constructors ---------------------------------------------------------

    @classmethod
    def flat(cls) -> PlanningState:
        """An account with no open position (FLAT redundant)."""
        return cls(exposure=0.0, is_flat=True)

    @classmethod
    def from_exposure(cls, exposure: float) -> PlanningState:
        """Build a state from a signed exposure; zero exposure means flat."""
        return cls(exposure=float(exposure), is_flat=abs(float(exposure)) < 1e-12)

    @classmethod
    def from_env(cls, env: Any) -> PlanningState:
        """Read the live planning state from a reset environment.

        Mirrors ``ForexEnvironment.action_masks`` exactly so the derived mask
        matches the real environment mask.
        """
        portfolio = getattr(env, "portfolio", None)
        if portfolio is None:
            raise ValueError("environment has no portfolio; call env.reset() first")
        snapshot = portfolio.snapshot()
        units = snapshot.position.units
        exposure = float(snapshot.gross_exposure / snapshot.equity) if snapshot.equity > 0 else 0.0
        if units < 0:
            exposure = -exposure
        return cls(exposure=exposure, is_flat=units == 0)

    @classmethod
    def from_action_mask(cls, action_mask: np.ndarray) -> PlanningState:
        """Reconstruct the state that reproduces a six-wide action mask.

        Only two things influence the MuZero mask: whether the account is flat,
        and whether the current exposure already equals a MuZero target within
        tolerance.  Both are recoverable from the mask, so the reconstruction is
        faithful for every state reachable by MuZero actions.
        """
        mask = project_action_mask(action_mask)
        if not mask[HOLD]:
            raise ValueError("action mask must always keep HOLD valid")
        if not mask[FLAT]:
            return cls.flat()
        invalid = [i for i in range(2, MUZERO_NUM_ACTIONS) if not mask[i]]
        if len(invalid) == 1:
            target = MUZERO_TARGET_EXPOSURES[invalid[0]]
            assert target is not None  # indices >= 2 are exposure actions
            return cls(exposure=float(target), is_flat=False)
        if len(invalid) > 1:
            raise ValueError("action mask cannot invalidate more than one exposure target")
        # No redundant exposure target: pick a state that masks nothing.
        return cls(exposure=0.0, is_flat=False)

    # -- transitions ----------------------------------------------------------

    def action_mask(self) -> np.ndarray:
        """Six-wide boolean mask for the current planning state."""
        return project_action_mask(valid_action_mask(self.exposure, is_flat=self.is_flat))

    def after(self, action: int) -> PlanningState:
        """Deterministically apply a MuZero action to the planning state."""
        if not 0 <= action < MUZERO_NUM_ACTIONS:
            raise ValueError(f"MuZero action {action} out of range [0, {MUZERO_NUM_ACTIONS})")
        target = MUZERO_TARGET_EXPOSURES[action]
        if target is None:  # HOLD preserves the account exactly
            return self
        if target == 0.0:  # FLAT closes to zero units
            return PlanningState(exposure=0.0, is_flat=True)
        return PlanningState(exposure=float(target), is_flat=False)

    def valid_actions(self) -> list[int]:
        """Indices of the actions valid in this planning state."""
        return [i for i, ok in enumerate(self.action_mask()) if ok]
