"""SMA crossover baseline (Phase 2).

Causal moving-average crossover on M5 closes:

* ``short SMA > long SMA``  -> +1
* ``short SMA < long SMA``  -> -1
* otherwise (equal)         ->  0

SMAs are computed only from closes up to ``t`` (the observation window).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forexmind.baselines.base import TradingAgent, _register
from forexmind.environment.actions import Action
from forexmind.observation.schema import EncodedObservation


@dataclass(frozen=True)
class SmaCrossoverConfig:
    short_window: int = 5
    long_window: int = 20
    name: str = "sma_crossover"

    def __post_init__(self) -> None:
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("SMA windows must be > 0")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be < long_window")

    def to_dict(self) -> dict[str, object]:
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "name": self.name,
        }


class SmaCrossoverAgent:
    name = "sma_crossover"

    def __init__(self, config: SmaCrossoverConfig | None = None) -> None:
        self.config = config or SmaCrossoverConfig()

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: EncodedObservation) -> Action:
        closes = observation.closes
        if len(closes) < self.config.long_window:
            return Action(0.0)
        short_sma = float(np.mean(closes[-self.config.short_window :]))
        long_sma = float(np.mean(closes[-self.config.long_window :]))
        if short_sma > long_sma:
            return Action(1.0)
        if short_sma < long_sma:
            return Action(-1.0)
        return Action(0.0)


@_register
def make_sma_crossover(config: SmaCrossoverConfig | None = None) -> TradingAgent:
    return SmaCrossoverAgent(config)
