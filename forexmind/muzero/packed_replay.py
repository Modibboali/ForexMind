"""Packed, vectorized MuZero target construction (Stage 4.7, brief S4-S12).

Stage 4.6 measured replay sampling at ~170 samples/s (~5.8 ms per sample)
because every sample was built in Python: ``build_unroll_sample`` walked the
unroll offsets, computed an ``n``-step value target with a Python loop and
allocated nine small arrays, after which ``collate_samples`` stacked them.

This module keeps the *exact same mathematics* (see
:mod:`forexmind.muzero.targets`, which remains the reference implementation) and
replaces the per-sample work with batch-wide array operations over a packed
layout::

    packed observations  [P, obs_dim]      trajectory_offsets [M]
    packed actions       [P]               trajectory_lengths [M]
    packed rewards       [P]               boundary_values    [M]
    packed policies      [P, 6]            trajectory_ids     [M]
    packed root values   [P]
    packed masks         [P, 6] bool
    packed terminated    [P] bool

where ``P`` is the total number of decision positions across the stored
trajectories.  A sample is ``(trajectory_index, position)`` and every gather is
``offsets[trajectory_index] + position + k``, clamped into the *own* trajectory
block and masked by ``position + k <= length`` — a packed layout can never let
one trajectory's unroll read into the next one's data (brief S5).

Value targets use the Stage 4.3 equation unchanged
(``z_t = sum_k gamma^(k-1) r_{t+k} + gamma^n V_bootstrap``); the accumulation
order is kept identical to the reference (left-to-right over ``k``, float64,
exact-terminal early stop) so the vectorized path reproduces the reference bit
for bit, not merely approximately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from forexmind.muzero.actions import MUZERO_NUM_ACTIONS
from forexmind.muzero.profiling import PhaseTimer, phase
from forexmind.muzero.targets import (
    PAD_ACTION,
    PAD_ACTION_MASK,
    MuZeroBatch,
    TargetConfig,
    build_unroll_sample,
    collate_samples,
)
from forexmind.muzero.trajectory import MuZeroTrajectory

__all__ = [
    "PackedTrajectories",
    "build_vectorized_batch",
    "sample_reference",
]


@dataclass(frozen=True, slots=True)
class PackedTrajectories:
    """Contiguous view of every stored trajectory (built once per mutation)."""

    observations: np.ndarray  # [P, obs_dim] float32 (decision observations)
    actions: np.ndarray  # [P] int64
    rewards: np.ndarray  # [P] float32
    root_policies: np.ndarray  # [P, 6] float32
    root_values: np.ndarray  # [P] float32
    action_masks: np.ndarray  # [P, 6] bool
    terminated: np.ndarray  # [P] bool
    offsets: np.ndarray  # [M] int64, start of each trajectory block
    lengths: np.ndarray  # [M] int64, transitions per trajectory
    #: float64: ``MuZeroTrajectory.boundary_value`` is a Python float and the
    #: reference value target adds it at full precision; storing it as float32
    #: here would perturb every bootstrap by ~3e-8 relative.
    boundary_values: np.ndarray  # [M] float64
    trajectory_ids: np.ndarray  # [M] int64
    splits: tuple[str, ...]
    obs_dim: int
    total_positions: int

    @property
    def num_trajectories(self) -> int:
        return int(self.lengths.shape[0])

    @classmethod
    def from_trajectories(cls, trajectories: Sequence[MuZeroTrajectory]) -> PackedTrajectories:
        """Pack trajectories in buffer order (FIFO), preserving their arrays."""
        if not trajectories:
            raise ValueError("cannot pack an empty trajectory list")
        lengths = np.asarray([len(trajectory) for trajectory in trajectories], dtype=np.int64)
        if np.any(lengths <= 0):
            raise ValueError("every packed trajectory must contain at least one transition")
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        total = int(lengths.sum())
        obs_dim = int(trajectories[0].observations.shape[-1])
        if any(int(t.observations.shape[-1]) != obs_dim for t in trajectories):
            raise ValueError("all packed trajectories must share one observation dimension")

        observations = np.empty((total, obs_dim), dtype=np.float32)
        actions = np.empty(total, dtype=np.int64)
        rewards = np.empty(total, dtype=np.float32)
        policies = np.empty((total, MUZERO_NUM_ACTIONS), dtype=np.float32)
        values = np.empty(total, dtype=np.float32)
        masks = np.empty((total, MUZERO_NUM_ACTIONS), dtype=bool)
        terminated = np.empty(total, dtype=bool)

        for index, trajectory in enumerate(trajectories):
            start = int(offsets[index])
            stop = start + int(lengths[index])
            observations[start:stop] = np.asarray(
                trajectory.observations[: stop - start], dtype=np.float32
            )
            actions[start:stop] = np.asarray(trajectory.actions, dtype=np.int64)
            rewards[start:stop] = np.asarray(trajectory.rewards, dtype=np.float32)
            policies[start:stop] = np.asarray(trajectory.root_policies, dtype=np.float32)
            values[start:stop] = np.asarray(trajectory.root_values, dtype=np.float32)
            masks[start:stop] = np.asarray(trajectory.action_masks, dtype=bool)
            terminated[start:stop] = np.asarray(trajectory.terminated, dtype=bool)

        return cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            root_policies=policies,
            root_values=values,
            action_masks=masks,
            terminated=terminated,
            offsets=offsets,
            lengths=lengths,
            boundary_values=np.asarray(
                [float(t.boundary_value) for t in trajectories], dtype=np.float64
            ),
            trajectory_ids=np.asarray(
                [int(t.metadata.trajectory_id) for t in trajectories], dtype=np.int64
            ),
            splits=tuple(str(t.metadata.split) for t in trajectories),
            obs_dim=obs_dim,
            total_positions=total,
        )

    def nbytes(self) -> int:
        return int(
            self.observations.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.root_policies.nbytes
            + self.root_values.nbytes
            + self.action_masks.nbytes
            + self.terminated.nbytes
            + self.offsets.nbytes
            + self.lengths.nbytes
            + self.boundary_values.nbytes
            + self.trajectory_ids.nbytes
        )


def _value_targets_vectorized(
    packed: PackedTrajectories,
    trajectory_index: np.ndarray,  # [N] int64 indices into the packed blocks
    starts: np.ndarray,  # [N] int64 local start positions (0 <= start <= T)
    config: TargetConfig,
    *,
    timer: PhaseTimer | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Stage 4.3 value targets for many ``(trajectory, position)``.

    Returns ``(targets_float64, valid)`` where ``valid`` marks states that exist
    inside the trajectory (``start <= length``); invalid rows are left at 0.
    Accumulation order matches the scalar reference exactly.
    """
    n = int(config.td_steps)
    counts = int(starts.shape[0])
    lengths = packed.lengths[trajectory_index]
    offsets = packed.offsets[trajectory_index]
    valid = starts <= lengths
    clamped_lengths = np.maximum(lengths - 1, 0)

    total = np.zeros(counts, dtype=np.float64)
    consumed = np.zeros(counts, dtype=np.int64)
    hit_terminal = np.zeros(counts, dtype=bool)
    alive = np.ones(counts, dtype=bool)
    # Powers are built by repeated multiplication, exactly like the reference
    # loop, so the floating-point sequence is identical.
    powers = np.ones(n + 1, dtype=np.float64)
    for k in range(1, n + 1):
        powers[k] = powers[k - 1] * float(config.discount)

    with phase(timer, "value_target_construction"):
        for k in range(n):
            local = starts + k
            available = local < lengths
            contributes = available & alive
            gather = offsets + np.minimum(local, np.maximum(lengths - 1, 0))
            reward = packed.rewards[gather].astype(np.float64)
            total += np.where(contributes, powers[k] * reward, 0.0)
            consumed = np.where(contributes, k + 1, consumed)
            terminal_here = packed.terminated[gather] & contributes
            hit_terminal |= terminal_here
            alive &= ~available | ~packed.terminated[gather]

        # NB: the bootstrap state index uses the *true* start; only the gather
        # index is clamped.  A state index equal to the trajectory length means
        # "past the last decision" and must use the boundary value, not the last
        # stored root value.
        bootstrap_local = starts + consumed
        inside = bootstrap_local < lengths
        bootstrap_gather = offsets + np.minimum(bootstrap_local, clamped_lengths)
        bootstrap = np.where(
            inside,
            packed.root_values[bootstrap_gather].astype(np.float64),
            np.where(
                config.use_boundary_value,
                packed.boundary_values[trajectory_index].astype(np.float64),
                0.0,
            ),
        )
        power_after = powers[consumed]
        targets = np.where(hit_terminal, total, total + power_after * bootstrap)
        targets = np.where(valid, targets, 0.0)
    return targets, valid


def build_vectorized_batch(
    packed: PackedTrajectories,
    trajectory_index: np.ndarray,
    positions: np.ndarray,
    config: TargetConfig,
    *,
    timer: PhaseTimer | None = None,
    device: torch.device | str | None = None,
) -> MuZeroBatch:
    """Build a whole :class:`MuZeroBatch` with packed gathers (no Python loop)."""
    trajectory_index = np.asarray(trajectory_index, dtype=np.int64).reshape(-1)
    positions = np.asarray(positions, dtype=np.int64).reshape(-1)
    if trajectory_index.shape != positions.shape:
        raise ValueError("trajectory_index and positions must have the same length")
    batch = int(positions.shape[0])
    if batch == 0:
        raise ValueError("cannot build an empty batch")
    if np.any(trajectory_index < 0) or np.any(trajectory_index >= packed.num_trajectories):
        raise IndexError("trajectory index out of range for the packed buffer")
    lengths = packed.lengths[trajectory_index]
    if np.any(positions < 0) or np.any(positions >= lengths):
        raise IndexError("sample position out of range for its trajectory")

    k = int(config.num_unroll_steps)
    offsets = packed.offsets[trajectory_index]
    base = offsets + positions  # decision index of the sample's own state
    clamped_lengths = np.maximum(lengths - 1, 0)

    # ---------------------------------------------------------------- decisions
    decision_local = positions[:, None] + np.arange(k, dtype=np.int64)[None, :]
    decision_valid = decision_local < lengths[:, None]
    decision_gather = offsets[:, None] + np.minimum(decision_local, clamped_lengths[:, None])
    with phase(timer, "action_reward_gather"):
        actions = np.where(
            decision_valid, packed.actions[decision_gather], PAD_ACTION
        ).astype(np.int64)
        target_rewards = np.where(
            decision_valid, packed.rewards[decision_gather], np.float32(0.0)
        ).astype(np.float32)
        reward_masks = decision_valid.astype(np.float32)

    # ------------------------------------------------------------------- states
    state_local = positions[:, None] + np.arange(k + 1, dtype=np.int64)[None, :]
    state_valid = state_local <= lengths[:, None]  # state index 0..T inclusive
    state_gather = offsets[:, None] + np.minimum(state_local, clamped_lengths[:, None])
    with phase(timer, "value_target_construction"):
        flat_targets, _ = _value_targets_vectorized(
            packed,
            np.repeat(trajectory_index, k + 1),
            state_local.reshape(-1),
            config,
        )
    target_values = np.where(state_valid, flat_targets.reshape(batch, k + 1), 0.0).astype(
        np.float32
    )
    value_masks = state_valid.astype(np.float32)

    with phase(timer, "policy_gather"):
        decision_states = state_local < lengths[:, None]
        # ``state_is_terminal(index)`` is ``terminated[index - 1]``; index 0 is
        # never terminal.
        previous = np.maximum(state_local - 1, 0)
        terminal_before = np.where(
            state_local >= 1,
            packed.terminated[offsets[:, None] + np.minimum(previous, clamped_lengths[:, None])],
            False,
        )
        policy_masks = (decision_states & ~terminal_before).astype(np.float32)
        target_policies = np.where(
            policy_masks[:, :, None] > 0.0,
            packed.root_policies[state_gather],
            np.float32(0.0),
        ).astype(np.float32)

    with phase(timer, "action_mask_gather"):
        action_masks = np.where(
            decision_states[:, :, None],
            packed.action_masks[state_gather],
            PAD_ACTION_MASK[None, None, :],
        ).astype(bool)

    with phase(timer, "observation_gather"):
        observations = packed.observations[base].astype(np.float32)

    with phase(timer, "numpy_to_torch"):
        result = MuZeroBatch(
            observation=torch.as_tensor(np.ascontiguousarray(observations), dtype=torch.float32),
            actions=torch.as_tensor(np.ascontiguousarray(actions), dtype=torch.int64),
            target_rewards=torch.as_tensor(
                np.ascontiguousarray(target_rewards), dtype=torch.float32
            ),
            target_values=torch.as_tensor(
                np.ascontiguousarray(target_values), dtype=torch.float32
            ),
            target_policies=torch.as_tensor(
                np.ascontiguousarray(target_policies), dtype=torch.float32
            ),
            policy_masks=torch.as_tensor(
                np.ascontiguousarray(policy_masks), dtype=torch.float32
            ),
            value_masks=torch.as_tensor(np.ascontiguousarray(value_masks), dtype=torch.float32),
            reward_masks=torch.as_tensor(
                np.ascontiguousarray(reward_masks), dtype=torch.float32
            ),
            action_masks=torch.as_tensor(np.ascontiguousarray(action_masks), dtype=torch.bool),
            trajectory_ids=torch.as_tensor(
                np.ascontiguousarray(packed.trajectory_ids[trajectory_index]), dtype=torch.int64
            ),
            positions=torch.as_tensor(np.ascontiguousarray(positions), dtype=torch.int64),
            splits=tuple(packed.splits[index] for index in trajectory_index.tolist()),
        )
    if device is not None:
        result = result.to(device)
    return result


def sample_reference(
    trajectories: Sequence[MuZeroTrajectory],
    trajectory_index: np.ndarray,
    positions: np.ndarray,
    config: TargetConfig,
    *,
    device: torch.device | str | None = None,
    timer: PhaseTimer | None = None,
) -> MuZeroBatch:
    """The Stage 4.3/4.6 per-sample implementation (kept as the reference)."""
    samples = [
        build_unroll_sample(trajectories[int(index)], int(position), config, timer=timer)
        for index, position in zip(trajectory_index, positions, strict=True)
    ]
    return collate_samples(samples, device=device, timer=timer)


def describe_batch(batch: MuZeroBatch) -> dict[str, Any]:
    """Small helper for diagnostics/tests."""
    return batch.to_dict()
