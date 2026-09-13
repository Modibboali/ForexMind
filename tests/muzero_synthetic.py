"""Synthetic MuZero trajectory builders shared by the Stage 4.3 tests.

Not a test module itself (pytest only collects ``test_*.py``): it exists so the
trajectory/target/replay tests can construct exactly-indexed trajectories
without duplicating scaffolding.
"""

from __future__ import annotations

import numpy as np
from forexmind.muzero.actions import (
    HOLD,
    MUZERO_NUM_ACTIONS,
    PlanningState,
    project_action_mask,
    valid_action_mask,
)
from forexmind.muzero.trajectory import MuZeroTrajectory, TrajectoryMetadata

ALL_VALID = np.ones(MUZERO_NUM_ACTIONS, dtype=bool)


def one_hot(index: int) -> np.ndarray:
    policy = np.zeros(MUZERO_NUM_ACTIONS, dtype=np.float32)
    policy[index] = 1.0
    return policy


def uniform_policy(mask: np.ndarray | None = None) -> np.ndarray:
    mask = ALL_VALID if mask is None else np.asarray(mask, dtype=bool)
    policy = np.zeros(MUZERO_NUM_ACTIONS, dtype=np.float32)
    policy[mask] = 1.0 / float(mask.sum())
    return policy


def make_trajectory(
    *,
    actions: list[int],
    rewards: list[float],
    root_policies: np.ndarray | list[np.ndarray] | None = None,
    root_values: list[float] | None = None,
    action_masks: np.ndarray | None = None,
    terminated: list[bool] | None = None,
    truncated: list[bool] | None = None,
    boundary_value: float = 0.0,
    trajectory_id: int = 0,
    split: str = "train",
    obs_dim: int = 4,
    initial_exposure: float = 0.0,
    initial_is_flat: bool = True,
    instrument: str = "EURUSD",
    num_simulations: int = 8,
    discount: float = 0.99,
    temperature: float = 1.0,
    training: bool = True,
    observation_tag: float = 0.0,
) -> MuZeroTrajectory:
    """Build a valid trajectory from explicit actions/rewards.

    Observations are synthetic ``[t, 0, 0, ...]`` vectors and masks/planning
    states are derived consistently, so callers only need to specify the
    economic content.
    """
    steps = len(actions)
    if len(rewards) != steps:
        raise ValueError("actions and rewards must have the same length")

    observations = np.zeros((steps + 1, obs_dim), dtype=np.float32)
    for i in range(steps + 1):
        observations[i, 0] = float(i)
        if obs_dim > 1:
            # A per-trajectory marker so different trajectories are distinguishable
            # from their observations alone (needed by the value/policy sanity tests).
            observations[i, 1] = observation_tag

    if action_masks is None:
        planning = [PlanningState(exposure=initial_exposure, is_flat=initial_is_flat)]
        for action in actions:
            planning.append(planning[-1].after(action))
        masks = np.asarray([planning[t].action_mask() for t in range(steps)], dtype=bool).reshape(
            steps, MUZERO_NUM_ACTIONS
        )
    else:
        masks = np.asarray(action_masks, dtype=bool).reshape(steps, MUZERO_NUM_ACTIONS)
        planning = []
        for t in range(steps):
            planning.append(PlanningState.from_action_mask(masks[t]))
        planning.append(planning[-1].after(actions[-1]) if steps else PlanningState.flat())
        for t in range(steps):
            if not np.array_equal(planning[t].action_mask(), masks[t]):
                raise ValueError(f"action_masks[{t}] is not reproducible from a planning state")

    if root_policies is None:
        if steps == 0:
            policies = np.zeros((0, MUZERO_NUM_ACTIONS), dtype=np.float32)
        else:
            policies = np.stack([uniform_policy(masks[t]) for t in range(steps)]).astype(np.float32)
    else:
        policies = np.asarray(root_policies, dtype=np.float32).reshape(steps, MUZERO_NUM_ACTIONS)

    values = (
        np.zeros(steps, dtype=np.float32)
        if root_values is None
        else np.asarray(root_values, dtype=np.float32)
    )
    term = np.zeros(steps, dtype=bool) if terminated is None else np.asarray(terminated, dtype=bool)
    trunc = np.zeros(steps, dtype=bool) if truncated is None else np.asarray(truncated, dtype=bool)

    return MuZeroTrajectory(
        observations=observations,
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        root_policies=policies,
        root_values=values,
        action_masks=masks,
        terminated=term,
        truncated=trunc,
        planning_exposure=np.asarray([s.exposure for s in planning], dtype=np.float32),
        planning_is_flat=np.asarray([s.is_flat for s in planning], dtype=bool),
        boundary_value=float(boundary_value),
        metadata=TrajectoryMetadata(
            trajectory_id=trajectory_id,
            instrument=instrument,
            split=split,
            start_index=0,
            horizon=steps,
            episode_seed=0,
            search_seed=0,
            model_version="test",
            num_simulations=num_simulations,
            discount=discount,
            temperature=temperature,
            num_steps=steps,
            training=training,
        ),
    )


def valid_mask_for(exposure: float, *, is_flat: bool) -> np.ndarray:
    return project_action_mask(valid_action_mask(exposure, is_flat=is_flat))


__all__ = [
    "ALL_VALID",
    "HOLD",
    "make_trajectory",
    "one_hot",
    "uniform_policy",
    "valid_mask_for",
]
