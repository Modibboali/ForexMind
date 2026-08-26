"""Random baseline: uniform target exposure from {-1, -0.5, 0, +0.5, +1}.

Uses a seeded RNG; the same seed produces the same trajectory.  Evaluation
runs many seeds and aggregates them - a single random run is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forexmind.baselines.base import TradingAgent, _register
from forexmind.environment.actions import Action
from forexmind.observation.schema import EncodedObservation


@dataclass(frozen=True)
class RandomConfig:
    targets: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    seed: int | None = None  # per-run agent seed (overrides episode seed if set)
    name: str = "random"

    def to_dict(self) -> dict[str, object]:
        return {"targets": list(self.targets), "seed": self.seed, "name": self.name}


class RandomAgent:
    """Uniform random target-exposure agent (seeded)."""

    name = "random"

    def __init__(self, config: RandomConfig | None = None) -> None:
        self.config = config or RandomConfig()
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(
            self.config.seed if self.config.seed is not None else seed
        )

    def act(self, observation: EncodedObservation) -> Action:
        if self._rng is None:
            raise RuntimeError("RandomAgent.act called before reset(seed)")
        return Action(float(self._rng.choice(self.config.targets)))


@_register
def make_random(config: RandomConfig | None = None) -> TradingAgent:
    return RandomAgent(config)
