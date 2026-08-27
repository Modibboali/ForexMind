"""High-throughput replay buffer for off-policy RL (Phase 3).

A numpy ring buffer storing ``(obs, action, reward, next_obs, terminated,
truncated)`` float32 arrays.  Only training-split transitions ever enter it
(the collector is configured with ``split="train"``).  Capacity is
configurable; buffer metadata (size, fill) is exposed for monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Transition = tuple[np.ndarray, float, float, np.ndarray, bool, bool]
# (obs, action, reward, next_obs, terminated, truncated)


@dataclass(frozen=True)
class TransitionBatch:
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    @property
    def size(self) -> int:
        return len(self.reward)


class ReplayBuffer:
    """Fixed-capacity numpy ring buffer (single-writer, multi-reader safe for
    the learner's own sampling)."""

    def __init__(
        self, obs_dim: int, capacity: int, action_dim: int = 1, seed: int = 0
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.obs_dim = obs_dim
        self.capacity = capacity
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=bool)
        self._truncated = np.zeros(capacity, dtype=bool)
        self._pos = 0
        self._size = 0
        self._total_pushed = 0

    # -- monitoring -----------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size == self.capacity

    def metadata(self) -> dict[str, object]:
        return {
            "buffer_size": self.size,
            "buffer_capacity": self.capacity,
            "training_split": "train",
            "transitions_collected": self._total_pushed,
        }

    # -- writing --------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: float | np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._obs[self._pos] = obs
        self._action[self._pos] = action
        self._reward[self._pos] = reward
        self._next_obs[self._pos] = next_obs
        self._terminated[self._pos] = terminated
        self._truncated[self._pos] = truncated
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._total_pushed += 1

    def extend(self, transitions: list[Transition]) -> None:
        for t in transitions:
            self.push(*t)

    # -- reading --------------------------------------------------------------

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size > self._size:
            raise ValueError(
                f"batch_size {batch_size} > buffer size {self._size}; collect more first"
            )
        idx = self._rng.integers(0, self._size, size=batch_size)
        return TransitionBatch(
            obs=self._obs[idx],
            action=self._action[idx],
            reward=self._reward[idx],
            next_obs=self._next_obs[idx],
            terminated=self._terminated[idx],
            truncated=self._truncated[idx],
        )

    def sample_seeded(self, batch_size: int, seed: int) -> TransitionBatch:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self._size, size=batch_size)
        return TransitionBatch(
            obs=self._obs[idx],
            action=self._action[idx],
            reward=self._reward[idx],
            next_obs=self._next_obs[idx],
            terminated=self._terminated[idx],
            truncated=self._truncated[idx],
        )

    # -- checkpoint metadata --------------------------------------------------

    def state_meta(self) -> dict[str, Any]:
        """Replay-buffer metadata for checkpoints (buffer itself is omitted to
        keep checkpoints small; see README)."""
        return {
            "size": self._size,
            "capacity": self.capacity,
            "pos": self._pos,
            "total_pushed": self._total_pushed,
        }

    def load_meta(self, meta: dict[str, Any]) -> None:
        """Restore counter metadata after a resume (buffer starts empty)."""
        self._total_pushed = int(meta.get("total_pushed", 0) or 0)
