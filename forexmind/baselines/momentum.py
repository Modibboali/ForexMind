"""Momentum baseline (Phase 2).

Transparent deterministic momentum on M5 closes:

* ``return over lookback > +threshold``  -> +1 (long)
* ``return over lookback < -threshold``  -> -1 (short)
* otherwise                              ->  0 (flat)

All inputs come from the causal observation window (closes up to ``t``).  The
parameters are documented defaults and are NOT tuned on the test set.
"""

from __future__ import annotations

from dataclasses import dataclass

from forexmind.baselines.base import TradingAgent, _register
from forexmind.environment.actions import Action
from forexmind.observation.schema import EncodedObservation


@dataclass(frozen=True)
class MomentumConfig:
    lookback: int = 24  # M5 bars
    threshold: float = 0.001  # absolute return threshold
    name: str = "momentum"

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be > 0")
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return {
            "lookback": self.lookback,
            "threshold": self.threshold,
            "name": self.name,
        }


class MomentumAgent:
    name = "momentum"

    def __init__(self, config: MomentumConfig | None = None) -> None:
        self.config = config or MomentumConfig()

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: EncodedObservation) -> Action:
        closes = observation.closes
        need = self.config.lookback + 1
        if len(closes) < need:
            return Action(0.0)
        ret = float(closes[-1] / closes[-need] - 1.0)
        if ret > self.config.threshold:
            return Action(1.0)
        if ret < -self.config.threshold:
            return Action(-1.0)
        return Action(0.0)


@_register
def make_momentum(config: MomentumConfig | None = None) -> TradingAgent:
    return MomentumAgent(config)
