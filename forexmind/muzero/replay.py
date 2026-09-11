"""Trajectory-level MuZero replay (Stage 4.3).

MuZero replay stores **complete trajectories**, not flat transitions, because the
learner unrolls the dynamics over ``K`` steps from each sampled position.  A
sampled training position is the pair ``(trajectory_id, position)``.

Sampling strategies are pluggable::

    uniform        canonical baseline (default)
    decision_rich  mixture of uniform and decision-event positions, disabled
                   unless ``decision_rich_weight > 0``

Both are implemented as plain functions registered in
:data:`SAMPLING_STRATEGIES`, so a later stage can add ``prioritized`` or
``reanalysis`` without changing the buffer API.

Storage is compact NumPy (float32 observations/targets, int64 actions, bool
masks) and growth is bounded by ``max_trajectories`` / ``max_transitions`` with
FIFO eviction.

Split protection: :meth:`TrajectoryReplayBuffer.add` refuses trajectories whose
metadata split is not the required one (``train`` by default), so validation or
test data can never leak into training replay.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from forexmind.muzero.actions import FLAT, HOLD
from forexmind.muzero.targets import MuZeroBatch, TargetConfig, build_unroll_sample, collate_samples
from forexmind.muzero.trajectory import MuZeroTrajectory

__all__ = [
    "SAMPLING_STRATEGIES",
    "ReplayConfig",
    "TrajectoryReplayBuffer",
    "action_distribution_diagnostics",
    "event_distribution_diagnostics",
]

SHORT_ACTIONS: tuple[int, ...] = (2, 3)
LONG_ACTIONS: tuple[int, ...] = (4, 5)


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Capacity and default sampling configuration."""

    max_trajectories: int = 256
    max_transitions: int | None = None
    sampling: str = "uniform"
    decision_rich_weight: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_trajectories < 1:
            raise ValueError(f"max_trajectories must be >= 1, got {self.max_trajectories}")
        if self.max_transitions is not None and self.max_transitions < 1:
            raise ValueError(f"max_transitions must be >= 1 or None, got {self.max_transitions}")
        if self.sampling not in SAMPLING_STRATEGIES:
            raise ValueError(
                f"unknown sampling strategy {self.sampling!r}; "
                f"expected one of {sorted(SAMPLING_STRATEGIES)}"
            )
        if not 0.0 <= self.decision_rich_weight <= 1.0:
            raise ValueError(
                f"decision_rich_weight must be in [0, 1], got {self.decision_rich_weight}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_trajectories": self.max_trajectories,
            "max_transitions": self.max_transitions,
            "sampling": self.sampling,
            "decision_rich_weight": self.decision_rich_weight,
            "seed": self.seed,
        }


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def action_distribution_diagnostics(actions: np.ndarray) -> dict[str, float]:
    """Share of HOLD / FLAT / SHORT / LONG actions in ``actions``."""
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    total = int(actions.size)
    if total == 0:
        return {"pct_hold": 0.0, "pct_flat": 0.0, "pct_short": 0.0, "pct_long": 0.0, "n": 0.0}
    hold = float(np.count_nonzero(actions == HOLD)) / total
    flat = float(np.count_nonzero(actions == FLAT)) / total
    short = float(np.count_nonzero(np.isin(actions, SHORT_ACTIONS))) / total
    long = float(np.count_nonzero(np.isin(actions, LONG_ACTIONS))) / total
    return {
        "pct_hold": hold,
        "pct_flat": flat,
        "pct_short": short,
        "pct_long": long,
        "n": float(total),
    }


def event_distribution_diagnostics(
    trajectory: MuZeroTrajectory, positions: Sequence[int] | None = None
) -> dict[str, float]:
    """Share of entry / exit / resize / HOLD decisions at the given positions."""
    indices = range(len(trajectory)) if positions is None else positions
    counts = {"hold": 0, "exit": 0, "entry": 0, "resize": 0}
    total = 0
    for position in indices:
        total += 1
        action = int(trajectory.actions[position])
        if action == HOLD:
            counts["hold"] += 1
            continue
        before = trajectory.planning_state(position)
        after = before.after(action)
        if action == FLAT:
            counts["exit"] += 1
        elif before.exposure == 0.0:
            counts["entry"] += 1
        elif after != before:
            counts["resize"] += 1
    if total == 0:
        return {"pct_hold": 0.0, "pct_exit": 0.0, "pct_entry": 0.0, "pct_resize": 0.0, "n": 0.0}
    return {
        "pct_hold": counts["hold"] / total,
        "pct_exit": counts["exit"] / total,
        "pct_entry": counts["entry"] / total,
        "pct_resize": counts["resize"] / total,
        "n": float(total),
    }


# --------------------------------------------------------------------------- #
# Sampling strategies
# --------------------------------------------------------------------------- #

#: A strategy maps ``(buffer, batch_size, rng, weight)`` to ``(traj_idx, position)`` pairs.
SamplingStrategy = Callable[
    ["TrajectoryReplayBuffer", int, np.random.Generator, float],
    list[tuple[int, int]],
]


def _uniform_positions(
    buffer: TrajectoryReplayBuffer, count: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    if buffer.total_positions == 0:
        raise ValueError("replay buffer contains no sampleable positions")
    draws = rng.integers(0, buffer.total_positions, size=count)
    return buffer.decode_flat_positions(draws)


def sample_uniform(
    buffer: TrajectoryReplayBuffer,
    count: int,
    rng: np.random.Generator,
    weight: float = 0.0,
) -> list[tuple[int, int]]:
    """Canonical baseline: every ``(trajectory, position)`` is equally likely."""
    del weight
    return _uniform_positions(buffer, count, rng)


def sample_decision_rich(
    buffer: TrajectoryReplayBuffer,
    count: int,
    rng: np.random.Generator,
    weight: float = 0.0,
) -> list[tuple[int, int]]:
    """Mixture of uniform and decision-event sampling.

    ``weight`` is the share of the batch drawn from positions whose action
    changes the position (entry / exit / resize).  ``weight = 0`` reduces
    exactly to :func:`sample_uniform`, so the baseline stays measurable.
    """
    if weight <= 0.0:
        return _uniform_positions(buffer, count, rng)
    events = buffer.event_positions()
    if events.size == 0:
        return _uniform_positions(buffer, count, rng)
    rich = round(count * weight)
    rich = min(rich, count)
    draws = rng.choice(events, size=rich, replace=events.size < rich)
    rich_pairs = buffer.decode_flat_positions(draws)
    rest = _uniform_positions(buffer, count - rich, rng)
    pairs = rich_pairs + rest
    rng.shuffle(pairs)
    return pairs


SAMPLING_STRATEGIES: dict[str, SamplingStrategy] = {
    "uniform": sample_uniform,
    "decision_rich": sample_decision_rich,
}


# --------------------------------------------------------------------------- #
# Buffer
# --------------------------------------------------------------------------- #


class TrajectoryReplayBuffer:
    """FIFO-bounded store of MuZero trajectories with position-level sampling."""

    def __init__(self, config: ReplayConfig | None = None) -> None:
        self.config = config or ReplayConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._trajectories: list[MuZeroTrajectory] = []
        self._offsets = np.zeros(0, dtype=np.int64)  # start of each trajectory's positions
        self._events: np.ndarray | None = None  # cached concatenated event flags
        self._total_positions = 0

    # -- capacity -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._trajectories)

    @property
    def num_trajectories(self) -> int:
        return len(self._trajectories)

    @property
    def num_transitions(self) -> int:
        return int(sum(len(trajectory) for trajectory in self._trajectories))

    @property
    def total_positions(self) -> int:
        """Number of sampleable positions (one per real transition)."""
        return self._total_positions

    @property
    def trajectories(self) -> tuple[MuZeroTrajectory, ...]:
        return tuple(self._trajectories)

    # -- mutation -------------------------------------------------------------

    def add(
        self,
        trajectory: MuZeroTrajectory,
        *,
        require_split: str | None = "train",
        validate: bool = True,
    ) -> None:
        """Append a trajectory, evicting oldest entries when over capacity.

        ``require_split`` guards against dataset leakage: a validation or test
        trajectory raises instead of silently entering training replay.
        """
        if require_split is not None and trajectory.metadata.split != require_split:
            raise ValueError(
                f"refusing to add a {trajectory.metadata.split!r} trajectory to replay "
                f"that requires {require_split!r} (dataset leakage)"
            )
        if validate:
            trajectory.validate()
        if len(trajectory) == 0:
            raise ValueError("cannot add an empty trajectory (no transitions)")

        self._trajectories.append(trajectory)
        self._offsets = np.append(self._offsets, self._total_positions)
        self._total_positions += len(trajectory)
        self._events = None
        self._evict()

    def _evict(self) -> None:
        keep = self.config.max_trajectories
        limit = self.config.max_transitions
        while self._trajectories and (
            len(self._trajectories) > keep
            or (limit is not None and self.num_transitions > limit and len(self._trajectories) > 1)
        ):
            self._trajectories.pop(0)
            self._reindex()

    def _reindex(self) -> None:
        offsets = np.zeros(len(self._trajectories), dtype=np.int64)
        running = 0
        for i, trajectory in enumerate(self._trajectories):
            offsets[i] = running
            running += len(trajectory)
        self._offsets = offsets
        self._total_positions = running
        self._events = None

    def clear(self) -> None:
        self._trajectories.clear()
        self._offsets = np.zeros(0, dtype=np.int64)
        self._total_positions = 0
        self._events = None

    # -- position decoding ----------------------------------------------------

    def decode_flat_positions(self, flat: np.ndarray) -> list[tuple[int, int]]:
        """Map flat position indices to ``(trajectory_index, position)`` pairs."""
        flat = np.asarray(flat, dtype=np.int64).reshape(-1)
        if self.total_positions == 0:
            raise ValueError("replay buffer contains no sampleable positions")
        if flat.size and (flat.min() < 0 or flat.max() >= self.total_positions):
            raise IndexError("flat position out of range")
        trajectory_index = np.searchsorted(self._offsets, flat, side="right") - 1
        position = flat - self._offsets[trajectory_index]
        return [(int(i), int(p)) for i, p in zip(trajectory_index, position, strict=True)]

    def event_positions(self) -> np.ndarray:
        """Flat positions whose action changes the position (entry/exit/resize)."""
        if self._trajectories and self._events is None:
            flags = []
            for trajectory in self._trajectories:
                actions = trajectory.actions
                for t in range(len(trajectory)):
                    action = int(actions[t])
                    flag = action != HOLD and (
                        trajectory.planning_state(t + 1) != trajectory.planning_state(t)
                    )
                    flags.append(flag)
            self._events = np.asarray(flags, dtype=bool)
        if self._events is None:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(self._events)

    # -- sampling -------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        *,
        target_config: TargetConfig | None = None,
        strategy: str | None = None,
        decision_rich_weight: float | None = None,
        rng: np.random.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> MuZeroBatch:
        """Sample a training batch of unroll positions."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not self._trajectories:
            raise ValueError("replay buffer is empty")
        config = target_config or TargetConfig()
        name = strategy or self.config.sampling
        if name not in SAMPLING_STRATEGIES:
            raise ValueError(f"unknown sampling strategy {name!r}")
        weight = (
            self.config.decision_rich_weight
            if decision_rich_weight is None
            else float(decision_rich_weight)
        )
        generator = rng if rng is not None else self._rng
        pairs = SAMPLING_STRATEGIES[name](self, batch_size, generator, weight)
        samples = [
            build_unroll_sample(self._trajectories[traj_index], position, config)
            for traj_index, position in pairs
        ]
        return collate_samples(samples, device=device)

    # -- diagnostics ----------------------------------------------------------

    def action_diagnostics(self) -> dict[str, float]:
        actions = (
            np.concatenate(
                [trajectory.actions for trajectory in self._trajectories], dtype=np.int64
            )
            if self._trajectories
            else np.zeros(0, dtype=np.int64)
        )
        return action_distribution_diagnostics(actions)

    def event_diagnostics(self) -> dict[str, float]:
        """Entry / exit / resize shares across every stored position."""
        counts = {"hold": 0, "exit": 0, "entry": 0, "resize": 0}
        total = 0
        for trajectory in self._trajectories:
            total += len(trajectory)
            for key, value in trajectory.event_counts().items():
                if key in counts:
                    counts[key] += value
        if total == 0:
            return {"pct_hold": 0.0, "pct_exit": 0.0, "pct_entry": 0.0, "pct_resize": 0.0, "n": 0.0}
        return {
            "pct_hold": counts["hold"] / total,
            "pct_exit": counts["exit"] / total,
            "pct_entry": counts["entry"] / total,
            "pct_resize": counts["resize"] / total,
            "n": float(total),
        }

    def sampling_diagnostics(self) -> dict[str, Any]:
        """§15 diagnostics: HOLD-heavy share plus decision-rich event shares."""
        return {
            "actions": self.action_diagnostics(),
            "events": self.event_diagnostics(),
            "num_positions": self.total_positions,
            "sampling_strategy": self.config.sampling,
            "decision_rich_weight": self.config.decision_rich_weight,
        }

    def memory_report(self) -> dict[str, Any]:
        """Approximate replay memory footprint and capacity usage."""
        per_trajectory = [trajectory.nbytes() for trajectory in self._trajectories]
        total_bytes = int(sum(per_trajectory))
        transitions = self.num_transitions
        return {
            "num_trajectories": self.num_trajectories,
            "num_transitions": transitions,
            "num_positions": self.total_positions,
            "max_trajectories": self.config.max_trajectories,
            "max_transitions": self.config.max_transitions,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / 1e6, 4),
            "bytes_per_trajectory": float(np.mean(per_trajectory)) if per_trajectory else 0.0,
            "bytes_per_transition": (total_bytes / transitions) if transitions else 0.0,
            "capacity_used_fraction": (
                self.num_trajectories / self.config.max_trajectories
                if self.config.max_trajectories
                else 0.0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "memory": self.memory_report(),
            "sampling": self.sampling_diagnostics(),
        }
