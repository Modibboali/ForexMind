"""Baseline agent interface and registry (Phase 2).

A baseline has NO direct access to future market data.  The only market
information available to :meth:`act` is the current
:class:`EncodedObservation`, which is causal by construction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from forexmind.environment.actions import Action
from forexmind.observation.schema import EncodedObservation

AgentFactory = Callable[[], "TradingAgent"]


class TradingAgent(Protocol):
    """Minimal agent interface shared by every baseline (and later RL agents)."""

    name: str

    def reset(self, seed: int | None = None) -> None:
        """Reset any internal state / RNG before an episode."""

    def act(self, observation: EncodedObservation) -> Action:
        """Choose a target-exposure action from the current observation only."""


def _factory_name(factory: object) -> str:
    name = getattr(factory, "__name__", type(factory).__name__)
    return str(name).removeprefix("make_")


def register_agent(name: str, factory: AgentFactory) -> None:
    """Register a baseline factory under ``name`` (idempotent)."""
    _AGENTS[name] = factory


def make_agent(name: str) -> TradingAgent:
    """Instantiate a registered baseline by name."""
    if name not in _AGENTS:
        raise KeyError(f"unknown agent {name!r}; available: {sorted(_AGENTS)}")
    return _AGENTS[name]()


def available_agents() -> list[str]:
    return sorted(_AGENTS)


_AGENTS: dict[str, AgentFactory] = {}


def _register(factory: AgentFactory) -> AgentFactory:
    register_agent(_factory_name(factory), factory)
    return factory
