"""Long/short exposure baselines (target-exposure equivalents of buy-and-hold).

In a target-exposure simulator "buy and hold" means holding a constant target
exposure.  ``long`` holds +1 (long exposure), ``short`` holds -1 (short
exposure reference).  These are not conventional equity-style buy-and-hold;
they are exposure reference strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

from forexmind.baselines.base import TradingAgent, _register
from forexmind.environment.actions import Action
from forexmind.observation.schema import EncodedObservation


@dataclass(frozen=True)
class LongExposureConfig:
    name: str = "long"

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name}


@dataclass(frozen=True)
class ShortExposureConfig:
    name: str = "short"

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name}


class LongExposureAgent:
    """Target +1.0 exposure after the initial observation."""

    name = "long"

    def __init__(self, config: LongExposureConfig | None = None) -> None:
        self.config = config or LongExposureConfig()

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: EncodedObservation) -> Action:
        return Action(1.0)


class ShortExposureAgent:
    """Target -1.0 exposure after the initial observation."""

    name = "short"

    def __init__(self, config: ShortExposureConfig | None = None) -> None:
        self.config = config or ShortExposureConfig()

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: EncodedObservation) -> Action:
        return Action(-1.0)


@_register
def make_long(config: LongExposureConfig | None = None) -> TradingAgent:
    return LongExposureAgent(config)


@_register
def make_short(config: ShortExposureConfig | None = None) -> TradingAgent:
    return ShortExposureAgent(config)
