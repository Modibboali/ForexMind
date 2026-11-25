"""Flat baseline: target exposure 0 at every step (no-trading reference)."""

from __future__ import annotations

from dataclasses import dataclass

from forexmind.baselines.base import TradingAgent, _register
from forexmind.environment.actions import Action
from forexmind.observation.schema import EncodedObservation


@dataclass(frozen=True)
class FlatConfig:
    name: str = "flat"

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name}


class FlatAgent:
    """Always targets flat (no exposure)."""

    name = "flat"

    def __init__(self, config: FlatConfig | None = None) -> None:
        self.config = config or FlatConfig()

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: EncodedObservation) -> Action:
        return Action(0.0)


@_register
def make_flat(config: FlatConfig | None = None) -> TradingAgent:
    return FlatAgent(config)
