"""Trajectory abstraction (Phase 2).

A compact record of one evaluated episode.  Full observations are not stored
(they can be enormous); the arrays below carry everything needed for metrics
and debugging.  Streaming evaluation is a future extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from forexmind.episodes.sampler import EpisodeSpec


@dataclass
class Trajectory:
    """One evaluated episode."""

    agent_name: str
    spec: EpisodeSpec
    timestamps: np.ndarray  # (n_steps,) datetime64[ns] (post-step observation)
    actions: np.ndarray  # (n_steps,) target exposure
    rewards: np.ndarray  # (n_steps,) float
    equity: np.ndarray  # (n_steps + 1,) includes initial equity
    log_returns: np.ndarray  # (n_steps,) log(E_{t+1}/E_t)
    position_units: np.ndarray  # (n_steps,) signed units after each step
    trade_log: list[dict[str, object]] = field(default_factory=list)
    info: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, object] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.rewards)

    @property
    def simple_returns(self) -> np.ndarray:
        if self.n_steps == 0:
            return np.empty(0)
        return self.equity[1:] / self.equity[:-1] - 1.0

    def to_dict(self, *, include_curves: bool = True) -> dict[str, object]:
        data: dict[str, object] = {
            "agent_name": self.agent_name,
            "spec": self.spec.to_dict(),
            "metrics": self.metrics,
            "trade_log": self.trade_log,
            "info": self.info,
        }
        if include_curves:
            data.update(
                {
                    "timestamps": [str(t) for t in self.timestamps],
                    "actions": self.actions.tolist(),
                    "rewards": self.rewards.tolist(),
                    "equity": self.equity.tolist(),
                    "log_returns": self.log_returns.tolist(),
                    "position_units": self.position_units.tolist(),
                }
            )
        return data
