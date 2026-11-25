"""MuZero trajectory contract (Stage 4.3).

A trajectory records one *real* Forex episode driven by MCTS::

    o_0 --a_0/r_1--> o_1 --a_1/r_2--> ... --a_{T-1}/r_T--> o_T

Indexing (this is the single source of truth for the whole stage):

* ``observations[t]`` is the observation **before** decision ``t`` (state ``s_t``).
* ``actions[t]`` is the MuZero action chosen at state ``s_t``.
* ``rewards[t]`` is the **real** environment reward produced by that action,
  i.e. ``r_{t+1} = log(equity[t+1] / equity[t])``.  ``rewards[t]`` is therefore
  the reward that ``recurrent_inference(latent_t, a_t)`` must predict.
* ``root_policies[t]`` / ``root_values[t]`` are the MCTS outputs at state ``s_t``.
* ``action_masks[t]`` is the mask used at state ``s_t``.
* ``terminated[t]`` / ``truncated[t]`` describe the *outcome* of the transition
  taken at state ``s_t``, so they refer to state ``s_{t+1}``.
* ``planning_exposure[i]`` / ``planning_is_flat[i]`` are the **actual** account
  position at state ``s_i``, read from the live environment.  They are ground
  truth, so they can never disagree with the real position; the deterministic
  ``after(action)`` rule from Stage 4.2 (which MCTS uses for imagined nodes) is
  only compared against them as a diagnostic.

Required invariants for ``T`` actions::

    observations    T + 1
    actions         T
    rewards         T
    root_policies   T
    root_values     T
    action_masks    T
    planning_exposure    T + 1   (actual account position per state)
    planning_is_flat     T + 1

The final observation exists because ``a_{T-1}`` produces ``o_T``.

Only **real** environment transitions are stored.  Predicted rewards from
imagined MCTS transitions never enter a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forexmind.muzero.actions import MUZERO_NUM_ACTIONS, PlanningState


@dataclass(frozen=True, slots=True)
class TrajectoryMetadata:
    """Provenance of one trajectory (everything needed to reproduce it)."""

    trajectory_id: int
    instrument: str
    split: str
    start_index: int
    horizon: int
    episode_seed: int
    search_seed: int
    model_version: str
    num_simulations: int
    discount: float
    temperature: float
    num_steps: int
    training: bool
    #: Integrated-loop model version used for the searches in this trajectory
    #: (Stage 4.5).  Staleness is ``current_version - network_version``.
    network_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "instrument": self.instrument,
            "split": self.split,
            "start_index": self.start_index,
            "horizon": self.horizon,
            "episode_seed": self.episode_seed,
            "search_seed": self.search_seed,
            "model_version": self.model_version,
            "num_simulations": self.num_simulations,
            "discount": self.discount,
            "temperature": self.temperature,
            "num_steps": self.num_steps,
            "training": self.training,
            "network_version": self.network_version,
        }


@dataclass(slots=True)
class MuZeroTrajectory:
    """One real episode stored in MuZero form (see the module docstring)."""

    observations: np.ndarray  # [T+1, obs_dim] float32
    actions: np.ndarray  # [T]        int64 (MuZero indices)
    rewards: np.ndarray  # [T]        float32 (rewards[t] = r_{t+1})
    root_policies: np.ndarray  # [T, 6]    float32 (normalized visit distribution)
    root_values: np.ndarray  # [T]        float32 (MCTS root value)
    action_masks: np.ndarray  # [T, 6]    bool
    terminated: np.ndarray  # [T]        bool (outcome of the transition at t)
    truncated: np.ndarray  # [T]        bool
    planning_exposure: np.ndarray  # [T+1]      float64 (from the live account)
    planning_is_flat: np.ndarray  # [T+1]      bool
    boundary_value: float
    metadata: TrajectoryMetadata
    extra: dict[str, Any] = field(default_factory=dict)

    # -- basics ---------------------------------------------------------------

    def __len__(self) -> int:
        """Number of real transitions ``T`` (equal to ``len(actions)``)."""
        return int(self.actions.shape[0])

    @property
    def num_transitions(self) -> int:
        return len(self)

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def num_actions(self) -> int:
        return int(self.root_policies.shape[-1])

    def planning_state(self, index: int) -> PlanningState:
        """Actual planning state of the account at state index ``0 <= index <= T``.

        These values are read from the live environment, never derived from the
        action chain, so they can never disagree with the real position.
        """
        if not 0 <= index <= len(self):
            raise IndexError(f"planning state index {index} out of range [0, {len(self)}]")
        return PlanningState(
            exposure=float(self.planning_exposure[index]),
            is_flat=bool(self.planning_is_flat[index]),
        )

    def planning_chain_disagreements(self) -> list[int]:
        """Indices where the deterministic ``after(action)`` rule differs from reality.

        The rule assumes the account reaches its target exposure exactly.  Under
        mark-to-market drift the two can differ by more than the masking
        tolerance, which is a legitimate diagnostic rather than corruption.
        """
        return [
            t
            for t in range(len(self))
            if self.planning_state(t + 1) != self.planning_state(t).after(int(self.actions[t]))
        ]

    def action_frequencies(self) -> np.ndarray:
        """Counts per MuZero action over the stored transitions."""
        return np.bincount(self.actions, minlength=MUZERO_NUM_ACTIONS).astype(np.float64)

    def event_counts(self) -> dict[str, int]:
        """Decision-event counts (entry / exit / resize / hold) for diagnostics."""
        from forexmind.muzero.actions import FLAT, HOLD

        counts = {"hold": 0, "flat": 0, "entry": 0, "exit": 0, "resize": 0}
        for t in range(len(self)):
            action = int(self.actions[t])
            if action == HOLD:
                counts["hold"] += 1
                continue
            previous = self.planning_state(t).exposure
            if action == FLAT:
                counts["flat"] += 1
                counts["exit"] += 1
                continue
            counts["entry" if previous == 0.0 else "resize"] += 1
        return counts

    # -- memory ---------------------------------------------------------------

    def memory_breakdown(self) -> dict[str, int]:
        """Bytes per stored array."""
        return {
            "observations": int(self.observations.nbytes),
            "actions": int(self.actions.nbytes),
            "rewards": int(self.rewards.nbytes),
            "root_policies": int(self.root_policies.nbytes),
            "root_values": int(self.root_values.nbytes),
            "action_masks": int(self.action_masks.nbytes),
            "terminated": int(self.terminated.nbytes),
            "truncated": int(self.truncated.nbytes),
            "planning_exposure": int(self.planning_exposure.nbytes),
            "planning_is_flat": int(self.planning_is_flat.nbytes),
        }

    def nbytes(self) -> int:
        return sum(self.memory_breakdown().values())

    # -- validation -----------------------------------------------------------

    def validate(self, *, atol: float = 1e-5) -> None:
        """Fail loudly on any indexing, masking, or alignment corruption."""
        steps = len(self)
        self._require_shape("observations", (steps + 1,), min_ndim=2)
        self._require_shape("actions", (steps,))
        self._require_shape("rewards", (steps,))
        self._require_shape("root_policies", (steps, MUZERO_NUM_ACTIONS))
        self._require_shape("root_values", (steps,))
        self._require_shape("action_masks", (steps, MUZERO_NUM_ACTIONS))
        self._require_shape("terminated", (steps,))
        self._require_shape("truncated", (steps,))
        self._require_shape("planning_exposure", (steps + 1,))
        self._require_shape("planning_is_flat", (steps + 1,))

        if not np.isfinite(self.observations).all():
            raise ValueError("observations contain non-finite values")
        if not np.isfinite(self.rewards).all():
            raise ValueError("rewards contain non-finite values")
        if not np.isfinite(self.root_values).all():
            raise ValueError("root_values contain non-finite values")
        if not np.isfinite(self.root_policies).all():
            raise ValueError("root_policies contain non-finite values")
        if not np.isfinite(self.boundary_value):
            raise ValueError("boundary_value must be finite")
        if not np.isfinite(self.planning_exposure).all():
            raise ValueError("planning_exposure contains non-finite values")
        if self.terminated.dtype != np.bool_ or self.truncated.dtype != np.bool_:
            raise ValueError("terminated/truncated must be boolean arrays")

        if steps > 0:
            if not self.terminated[:-1].sum() == 0:
                raise ValueError("termination may only occur on the final stored transition")
            if self.terminated[-1] and self.truncated[-1]:
                raise ValueError("final transition cannot be both terminated and truncated")
            if self.terminated[-1] and self.boundary_value != 0.0:
                raise ValueError("a true terminal must have boundary_value == 0")

        policies = self.root_policies.astype(np.float64)
        if np.any(policies < -atol):
            raise ValueError("root_policies must be non-negative")
        sums = policies.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=atol):
            raise ValueError(f"root_policies rows must sum to 1 (got {sums.min()}..{sums.max()})")

        for t in range(steps):
            mask = self.action_masks[t]
            if mask.shape != (MUZERO_NUM_ACTIONS,):
                raise ValueError(f"action_masks[{t}] must have {MUZERO_NUM_ACTIONS} entries")
            if not mask[0]:
                raise ValueError(f"action_masks[{t}] must keep HOLD valid")
            if not mask.any():
                raise ValueError(f"action_masks[{t}] must keep at least one action valid")
            action = int(self.actions[t])
            if not 0 <= action < MUZERO_NUM_ACTIONS:
                raise ValueError(f"actions[{t}]={action} out of range")
            if not mask[action]:
                raise ValueError(f"actions[{t}]={action} is not valid under action_masks[{t}]")
            if float(policies[t][~mask].sum()) > atol:
                raise ValueError(f"root_policies[{t}] places mass on invalid actions")
            expected = self.planning_state(t).action_mask()
            if not np.array_equal(expected, mask):
                raise ValueError(
                    f"action_masks[{t}] disagrees with the deterministic planning state"
                )

        if self.metadata.num_steps != steps:
            raise ValueError(
                f"metadata.num_steps={self.metadata.num_steps} != stored steps {steps}"
            )

    def _require_shape(self, name: str, head: tuple[int, ...], *, min_ndim: int = 1) -> None:
        array = getattr(self, name)
        if array.ndim < min_ndim:
            raise ValueError(f"{name} must have at least {min_ndim} dimension(s)")
        if array.shape[: len(head)] != head:
            raise ValueError(f"{name} shape {array.shape} does not start with {head}")

    # -- reporting ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Compact JSON-serializable summary (never the full arrays)."""
        return {
            "metadata": self.metadata.to_dict(),
            "num_transitions": len(self),
            "obs_dim": self.obs_dim,
            "action_counts": self.action_frequencies().astype(int).tolist(),
            "event_counts": self.event_counts(),
            "reward_sum": float(self.rewards.sum()),
            "reward_mean": float(self.rewards.mean()) if len(self) else 0.0,
            "boundary_value": float(self.boundary_value),
            "final_terminated": bool(self.terminated[-1]) if len(self) else False,
            "final_truncated": bool(self.truncated[-1]) if len(self) else False,
            "nbytes": self.nbytes(),
        }


def model_version(model: Any, *, prefix: str = "") -> str:
    """Short deterministic fingerprint of a model's parameters.

    Stored in trajectory metadata so a dataset can always be traced back to the
    exact weights that generated it.
    """
    import hashlib

    digest = hashlib.blake2b(digest_size=8)
    for name, param in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(param.detach().cpu().numpy()).tobytes())
    return f"{prefix}{digest.hexdigest()}"
