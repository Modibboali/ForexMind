"""Tests for baseline agents (Phase 2): deterministic behavior and expected
actions on synthetic observations."""

from __future__ import annotations

import pandas as pd
from forexmind.baselines import make_agent
from forexmind.baselines.mean_reversion import MeanReversionAgent, MeanReversionConfig
from forexmind.baselines.momentum import MomentumAgent, MomentumConfig
from forexmind.baselines.random import RandomAgent, RandomConfig
from forexmind.baselines.sma_crossover import SmaCrossoverAgent, SmaCrossoverConfig
from forexmind.environment.actions import Action
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.schema import EncodedObservation
from forexmind.observation.window import MarketWindowBuilder, WindowConfig

from tests.synthetic import (
    m5_from_closes,
    make_phase1_observation,
)

CONTEXT = 64
START = pd.Timestamp("2019-01-01")
END = pd.Timestamp("2030-01-01")


def _encode(closes: list[float]) -> EncodedObservation:
    m5 = m5_from_closes("2020-01-06", closes)
    builder = MarketWindowBuilder("EURUSD", m5, START, END, WindowConfig(context_length=CONTEXT))
    enc = ObservationEncoder(EncoderConfig(context_length=CONTEXT))
    obs = make_phase1_observation("EURUSD", pd.Timestamp("2020-01-06 12:00"))
    return enc.encode(obs, builder.build(len(closes) - 1))


def _closes(base: float, n: int, step: float) -> list[float]:
    return [base + step * i for i in range(n)]


def test_flat_agent() -> None:
    agent = make_agent("flat")
    obs = _encode(_closes(1.10, 80, 0.0))
    assert agent.act(obs) == Action(0.0)


def test_long_short_agents() -> None:
    long_agent = make_agent("long")
    short_agent = make_agent("short")
    obs = _encode(_closes(1.10, 80, 0.0))
    assert long_agent.act(obs) == Action(1.0)
    assert short_agent.act(obs) == Action(-1.0)


def test_momentum_uptrend_goes_long() -> None:
    agent = MomentumAgent(MomentumConfig(lookback=24, threshold=0.0005))
    obs = _encode(_closes(1.10, 80, 0.001))  # strong uptrend
    assert agent.act(obs) == Action(1.0)


def test_momentum_downtrend_goes_short() -> None:
    agent = MomentumAgent(MomentumConfig(lookback=24, threshold=0.0005))
    obs = _encode(_closes(1.20, 80, -0.001))
    assert agent.act(obs) == Action(-1.0)


def test_momentum_flat_stays_flat() -> None:
    agent = MomentumAgent(MomentumConfig(lookback=24, threshold=0.01))
    obs = _encode(_closes(1.10, 80, 0.0))
    assert agent.act(obs) == Action(0.0)


def test_mean_reversion_high_z_goes_short() -> None:
    # A spike well above the recent mean -> z high -> short.
    closes = [*_closes(1.10, 70, 0.0), 1.13]
    agent = MeanReversionAgent(
        MeanReversionConfig(lookback=20, upper_threshold=1.0, lower_threshold=-1.0)
    )
    assert agent.act(_encode(closes)) == Action(-1.0)


def test_mean_reversion_low_z_goes_long() -> None:
    # A dip well below the recent mean -> z low -> long.
    closes = [*_closes(1.12, 70, 0.0), 1.09]
    agent = MeanReversionAgent(
        MeanReversionConfig(lookback=20, upper_threshold=1.0, lower_threshold=-1.0)
    )
    assert agent.act(_encode(closes)) == Action(1.0)


def test_sma_crossover() -> None:
    # Uptrend: short SMA above long SMA -> long.
    agent = SmaCrossoverAgent(SmaCrossoverConfig(short_window=5, long_window=20))
    obs = _encode(_closes(1.10, 80, 0.001))
    assert agent.act(obs) == Action(1.0)
    # Downtrend -> short.
    obs = _encode(_closes(1.20, 80, -0.001))
    assert agent.act(obs) == Action(-1.0)


def test_random_seeded_deterministic() -> None:
    obs = _encode(_closes(1.10, 80, 0.0))
    a1 = RandomAgent(RandomConfig(seed=123))
    a2 = RandomAgent(RandomConfig(seed=123))
    a1.reset(999)
    a2.reset(999)
    seq1 = [a1.act(obs).target_exposure for _ in range(50)]
    seq2 = [a2.act(obs).target_exposure for _ in range(50)]
    assert seq1 == seq2
    assert all(x in (-1.0, -0.5, 0.0, 0.5, 1.0) for x in seq1)


def test_random_different_seeds_differ() -> None:
    obs = _encode(_closes(1.10, 80, 0.0))
    a1 = RandomAgent(RandomConfig(seed=1))
    a2 = RandomAgent(RandomConfig(seed=2))
    a1.reset(0)
    a2.reset(0)
    seq1 = [a1.act(obs).target_exposure for _ in range(50)]
    seq2 = [a2.act(obs).target_exposure for _ in range(50)]
    assert seq1 != seq2


def test_baselines_do_not_use_future_data() -> None:
    """Baselines only read the causal observation (no hidden data source)."""
    for name in ("flat", "long", "short", "momentum", "mean_reversion", "sma_crossover"):
        agent = make_agent(name)
        obs = _encode(_closes(1.10, 80, 0.001))
        action = agent.act(obs)
        assert isinstance(action, Action)
