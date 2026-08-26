"""Mean-reversion baseline (Phase 2).

Uses a z-score of the current close vs. its rolling mean/std, computed only
from past observations (causal)::

    z_t = (P_t - mu_t) / sigma_t

* ``z > upper_threshold``  -> -1 (short, expecting reversion down)
* ``z < lower_threshold``  -> +1 (long, expecting reversion up)
* otherwise                ->  0 (flat)

``mu_t``/``sigma_t`` are the mean/std of the trailing ``lookback`` closes
including ``t``.  No future values are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forexmind.baselines.base import TradingAgent, _register
from forexmind.environment.actions import Action
from forexmind.observation.schema import EncodedObservation


@dataclass(frozen=True)
class MeanReversionConfig:
    lookback: int = 32  # trailing bars for rolling mean/std
    upper_threshold: float = 1.0
    lower_threshold: float = -1.0
    name: str = "mean_reversion"

    def __post_init__(self) -> None:
        if self.lookback <= 1:
            raise ValueError("lookback must be > 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "lookback": self.lookback,
            "upper_threshold": self.upper_threshold,
            "lower_threshold": self.lower_threshold,
            "name": self.name,
        }


class MeanReversionAgent:
    name = "mean_reversion"

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self.config = config or MeanReversionConfig()

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: EncodedObservation) -> Action:
        closes = observation.closes
        if len(closes) < self.config.lookback + 1:
            return Action(0.0)
        recent = closes[-(self.config.lookback + 1) :]
        mu = float(np.mean(recent))
        sigma = float(np.std(recent))
        z = (float(closes[-1]) - mu) / sigma if sigma > 0.0 else 0.0
        if z > self.config.upper_threshold:
            return Action(-1.0)
        if z < self.config.lower_threshold:
            return Action(1.0)
        return Action(0.0)


@_register
def make_mean_reversion(config: MeanReversionConfig | None = None) -> TradingAgent:
    return MeanReversionAgent(config)
