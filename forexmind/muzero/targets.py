"""MuZero training-target construction (Stage 4.3).

Turns a stored :class:`~forexmind.muzero.trajectory.MuZeroTrajectory` into the
targets a recurrent MuZero learner needs.  Targets are **scalar economic
values**; categorical support encoding (`scalar_to_support`) happens in the loss
layer, never here, so replay never stores support-encoded targets.

Unroll alignment (the regression that must never break)
-------------------------------------------------------

For a sample starting at real position ``t`` with unroll length ``K``::

    initial_inference(o_t)                      -> policy target pi_t,  value target z_t
    recurrent(a_t)                              -> reward target r_{t+1},
                                                   policy target pi_{t+1}, value target z_{t+1}
    recurrent(a_{t+1})                          -> reward target r_{t+2}, ...
    ...
    recurrent(a_{t+K-1})                        -> reward target r_{t+K},
                                                   policy target pi_{t+K}, value target z_{t+K}

So ``target_rewards[k]`` corresponds to ``actions[k]`` = ``a_{t+k}`` and equals
the stored ``rewards[t+k]``, which is the real ``r_{t+k+1}``.  There is no
reward target for ``k = 0`` (the initial step has no dynamics transition), which
is why ``target_rewards`` has length ``K`` while ``target_values`` and
``target_policies`` have length ``K+1``.

Value target equation
---------------------

::

    z_t = sum_{k=1..n} gamma^(k-1) * r_{t+k}  +  gamma^n * V_bootstrap(t+n)

where ``n = td_steps`` and ``V_bootstrap`` is the *stored MCTS root value* of
state ``t+n`` when that state is inside the trajectory.  Bootstrap semantics:

* bootstrap state inside the recorded trajectory -> stored ``root_values[b]``;
* bootstrap state is the final stored observation -> ``trajectory.boundary_value``
  (a real final MCTS search value when the episode was truncated, ``0`` for a
  true terminal);
* a true terminal encountered earlier -> the sum stops and the bootstrap is ``0``;
* never a neural value substituted on the fly.

ForexMind is single-agent, so ``gamma`` is applied with a ``+`` sign at every
step; there is no sign flipping between "players".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from forexmind.muzero.actions import MUZERO_NUM_ACTIONS
from forexmind.muzero.profiling import PhaseTimer, phase
from forexmind.muzero.trajectory import MuZeroTrajectory

__all__ = [
    "MuZeroBatch",
    "MuZeroSample",
    "TargetConfig",
    "build_unroll_sample",
    "collate_samples",
    "state_is_terminal",
    "value_target",
]

#: Action used to pad an unroll past the end of a trajectory.  Its losses are
#: always masked, so the value never influences training.
PAD_ACTION = 0

#: Legal placeholder mask used for padded (loss-masked) unroll positions.
PAD_ACTION_MASK = np.array([True, False, False, False, False, False], dtype=bool)


@dataclass(frozen=True, slots=True)
class TargetConfig:
    """Unroll length, TD steps, and discount for target construction."""

    num_unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 0.99
    use_boundary_value: bool = True

    def __post_init__(self) -> None:
        if self.num_unroll_steps < 1:
            raise ValueError(f"num_unroll_steps must be >= 1, got {self.num_unroll_steps}")
        if self.td_steps < 1:
            raise ValueError(f"td_steps must be >= 1, got {self.td_steps}")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {self.discount}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_unroll_steps": self.num_unroll_steps,
            "td_steps": self.td_steps,
            "discount": self.discount,
            "use_boundary_value": self.use_boundary_value,
        }


def state_is_terminal(trajectory: MuZeroTrajectory, index: int) -> bool:
    """``True`` when state ``index`` is a *true* terminal state.

    A transition stored at position ``j`` describes state ``j`` -> state
    ``j + 1``, so state ``i`` is terminal exactly when ``terminated[i - 1]``.
    Truncation (an artificial time limit) is **not** terminal.
    """
    if index <= 0 or index > len(trajectory):
        return False
    return bool(trajectory.terminated[index - 1])


def value_target(
    trajectory: MuZeroTrajectory,
    t: int,
    *,
    td_steps: int,
    discount: float,
    use_boundary_value: bool = True,
) -> float:
    """Discounted ``n``-step value target for state index ``t`` (see module docs)."""
    steps = len(trajectory)
    if not 0 <= t <= steps:
        raise IndexError(f"position {t} out of range [0, {steps}]")
    if td_steps < 1:
        raise ValueError(f"td_steps must be >= 1, got {td_steps}")

    remaining = min(td_steps, steps - t)
    total = 0.0
    power = 1.0
    consumed = 0
    hit_terminal = False
    for k in range(1, remaining + 1):
        j = t + k - 1
        total += power * float(trajectory.rewards[j])
        consumed = k
        if trajectory.terminated[j]:
            hit_terminal = True
            break
        power *= discount

    if hit_terminal:
        return total

    bootstrap_index = t + consumed
    if bootstrap_index < steps:
        bootstrap = float(trajectory.root_values[bootstrap_index])
    elif use_boundary_value:
        bootstrap = float(trajectory.boundary_value)
    else:
        bootstrap = 0.0
    return total + power * bootstrap


@dataclass(frozen=True, slots=True)
class MuZeroSample:
    """One unroll position extracted from a trajectory.

    Shapes for unroll length ``K``::

        observation       [obs_dim]
        actions           [K]        int64
        target_rewards    [K]
        target_values     [K+1]
        target_policies   [K+1, 6]
        policy_masks      [K+1]
        value_masks       [K+1]
        reward_masks      [K]
        action_masks      [K+1, 6]   bool
    """

    trajectory_id: int
    position: int
    observation: np.ndarray
    actions: np.ndarray
    target_rewards: np.ndarray
    target_values: np.ndarray
    target_policies: np.ndarray
    policy_masks: np.ndarray
    value_masks: np.ndarray
    reward_masks: np.ndarray
    action_masks: np.ndarray
    #: Split of the source trajectory; the learner refuses anything but "train".
    split: str = "train"

    def validate(self) -> None:
        if self.observation.ndim != 1:
            raise ValueError(f"observation must be 1-D [obs_dim], got {self.observation.shape}")
        k = self.actions.shape[0]
        checks = [
            ("actions", self.actions.shape, (k,)),
            ("target_rewards", self.target_rewards.shape, (k,)),
            ("target_values", self.target_values.shape, (k + 1,)),
            ("target_policies", self.target_policies.shape, (k + 1, MUZERO_NUM_ACTIONS)),
            ("policy_masks", self.policy_masks.shape, (k + 1,)),
            ("value_masks", self.value_masks.shape, (k + 1,)),
            ("reward_masks", self.reward_masks.shape, (k,)),
            ("action_masks", self.action_masks.shape, (k + 1, MUZERO_NUM_ACTIONS)),
        ]
        for name, actual, expected in checks:
            if actual != expected:
                raise ValueError(f"{name} shape {actual} != {expected}")
        for name, array in (
            ("observation", self.observation),
            ("target_rewards", self.target_rewards),
            ("target_values", self.target_values),
            ("target_policies", self.target_policies),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True, slots=True)
class MuZeroBatch:
    """A collated batch of unroll samples (torch tensors, CPU by default).

    Shapes for batch size ``B`` and unroll length ``K``::

        observation       [B, obs_dim]
        actions           [B, K]
        target_rewards    [B, K]
        target_values     [B, K+1]
        target_policies   [B, K+1, 6]
        policy_masks      [B, K+1]
        value_masks       [B, K+1]
        reward_masks      [B, K]
        action_masks      [B, K+1, 6]
        trajectory_ids    [B]
        positions         [B]
    """

    observation: torch.Tensor
    actions: torch.Tensor
    target_rewards: torch.Tensor
    target_values: torch.Tensor
    target_policies: torch.Tensor
    policy_masks: torch.Tensor
    value_masks: torch.Tensor
    reward_masks: torch.Tensor
    action_masks: torch.Tensor
    trajectory_ids: torch.Tensor  # [B]
    positions: torch.Tensor  # [B]
    #: Split name of every row's source trajectory (dataset-leakage guard).
    splits: tuple[str, ...] = ()

    @property
    def batch_size(self) -> int:
        return int(self.observation.shape[0])

    @property
    def unroll_steps(self) -> int:
        return int(self.actions.shape[1])

    def shapes(self) -> dict[str, list[int]]:
        return {
            "observation": list(self.observation.shape),
            "actions": list(self.actions.shape),
            "target_rewards": list(self.target_rewards.shape),
            "target_values": list(self.target_values.shape),
            "target_policies": list(self.target_policies.shape),
            "policy_masks": list(self.policy_masks.shape),
            "value_masks": list(self.value_masks.shape),
            "reward_masks": list(self.reward_masks.shape),
            "action_masks": list(self.action_masks.shape),
            "trajectory_ids": list(self.trajectory_ids.shape),
            "positions": list(self.positions.shape),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "unroll_steps": self.unroll_steps,
            "shapes": self.shapes(),
        }

    def to(self, device: torch.device | str) -> MuZeroBatch:
        """Move every tensor to ``device`` (returns a new batch)."""
        moved = {
            name: getattr(self, name).to(device)
            for name in (
                "observation",
                "actions",
                "target_rewards",
                "target_values",
                "target_policies",
                "policy_masks",
                "value_masks",
                "reward_masks",
                "action_masks",
                "trajectory_ids",
                "positions",
            )
        }
        moved["splits"] = self.splits
        return MuZeroBatch(**moved)


def build_unroll_sample(
    trajectory: MuZeroTrajectory,
    position: int,
    config: TargetConfig,
    *,
    timer: PhaseTimer | None = None,
) -> MuZeroSample:
    """Build the unroll sample for real position ``position``.

    Pads with loss masks near the end of the trajectory (preferred over
    rejecting near-terminal states) and never crosses into another trajectory.

    ``timer`` is optional Stage 4.7 instrumentation (brief S3): it is a no-op
    unless a profiling run enables it.
    """
    trajectory.validate()
    steps = len(trajectory)
    if not 0 <= position < steps:
        raise IndexError(f"position {position} out of range [0, {steps})")

    k = config.num_unroll_steps
    with phase(timer, "padding_mask_construction"):
        actions = np.full(k, PAD_ACTION, dtype=np.int64)
        target_rewards = np.zeros(k, dtype=np.float32)
        reward_masks = np.zeros(k, dtype=np.float32)
        target_values = np.zeros(k + 1, dtype=np.float32)
        target_policies = np.zeros((k + 1, MUZERO_NUM_ACTIONS), dtype=np.float32)
        policy_masks = np.zeros(k + 1, dtype=np.float32)
        value_masks = np.zeros(k + 1, dtype=np.float32)
        action_masks = np.tile(PAD_ACTION_MASK, (k + 1, 1))

    with phase(timer, "action_reward_gather"):
        for offset in range(k):
            j = position + offset
            if j >= steps:
                break  # padded: losses stay masked, padding action is never trained
            actions[offset] = int(trajectory.actions[j])
            target_rewards[offset] = float(trajectory.rewards[j])
            reward_masks[offset] = 1.0

    with phase(timer, "policy_gather"):
        for offset in range(k + 1):
            index = position + offset
            if index > steps:
                break
            value_masks[offset] = 1.0
            if index < steps:
                action_masks[offset] = trajectory.action_masks[index]
                if not state_is_terminal(trajectory, index):
                    target_policies[offset] = trajectory.root_policies[index]
                    policy_masks[offset] = 1.0

    with phase(timer, "value_target_construction"):
        for offset in range(k + 1):
            index = position + offset
            if index > steps:
                break
            target_values[offset] = value_target(
                trajectory,
                index,
                td_steps=config.td_steps,
                discount=config.discount,
                use_boundary_value=config.use_boundary_value,
            )

    with phase(timer, "observation_gather"):
        observation = np.asarray(trajectory.observations[position], dtype=np.float32)

    with phase(timer, "sample_validation"):
        sample = MuZeroSample(
            trajectory_id=int(trajectory.metadata.trajectory_id),
            position=int(position),
            observation=observation,
            actions=actions,
            target_rewards=target_rewards,
            target_values=target_values,
            target_policies=target_policies,
            policy_masks=policy_masks,
            value_masks=value_masks,
            reward_masks=reward_masks,
            action_masks=action_masks,
            split=str(trajectory.metadata.split),
        )
        sample.validate()
    return sample


def collate_samples(
    samples: list[MuZeroSample],
    *,
    device: torch.device | str | None = None,
    timer: PhaseTimer | None = None,
) -> MuZeroBatch:
    """Stack unroll samples into a :class:`MuZeroBatch` of torch tensors."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    unroll = {sample.actions.shape[0] for sample in samples}
    if len(unroll) != 1:
        raise ValueError(f"all samples must share one unroll length, got {sorted(unroll)}")

    def stack(field: str, dtype: torch.dtype) -> torch.Tensor:
        array = np.stack([getattr(sample, field) for sample in samples], axis=0)
        return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype)

    with phase(timer, "batch_stacking"):
        batch = MuZeroBatch(
            observation=stack("observation", torch.float32),
            actions=stack("actions", torch.int64),
            target_rewards=stack("target_rewards", torch.float32),
            target_values=stack("target_values", torch.float32),
            target_policies=stack("target_policies", torch.float32),
            policy_masks=stack("policy_masks", torch.float32),
            value_masks=stack("value_masks", torch.float32),
            reward_masks=stack("reward_masks", torch.float32),
            action_masks=stack("action_masks", torch.bool),
            trajectory_ids=torch.as_tensor([s.trajectory_id for s in samples], dtype=torch.int64),
            positions=torch.as_tensor([s.position for s in samples], dtype=torch.int64),
            splits=tuple(s.split for s in samples),
        )
    if device is not None:
        batch = batch.to(device)
    return batch
