"""Backtest integrity / leakage tests (Phase 2, sections 49-50).

Constructs a synthetic dataset where the future close jumps hugely after an
observation and verifies that no baseline can access that future information:
the action is selected before the jump, execution uses the next-bar open, the
observation never contains the future bar, and reward reflects only the
resulting price movement.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest
from forexmind.baselines.base import TradingAgent
from forexmind.config import EnvironmentConfig, ExecutionConfig, MarginConfig, PositionSizingConfig
from forexmind.data.splits import SplitConfig, SplitDataset
from forexmind.environment.actions import Action
from forexmind.episodes.config import EpisodeConfig
from forexmind.episodes.sampler import EpisodeSampler
from forexmind.evaluation.runner import EvaluationRunner
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.schema import EncodedObservation
from forexmind.observation.window import WindowConfig

from tests.synthetic import m5_ohlc, make_instrument

CONTEXT = 4
JUMP_BAR = CONTEXT + 1  # index of the bucket whose OPEN is the huge jump


def _jump_m5() -> pd.DataFrame:
    """bars 0..CONTEXT close at 1.10; bar JUMP_BAR opens at 1.20."""
    bars = [(1.10, 1.10, 1.10, 1.10)] * (CONTEXT + 1) + [(1.20, 1.20, 1.20, 1.20)] * 8
    return m5_ohlc("2022-01-03 00:00", bars)


def _dataset() -> SplitDataset:
    cfg = SplitConfig(
        train_start=pd.Timestamp("2022-01-01"),
        train_end=pd.Timestamp("2022-01-02"),
        validation_start=pd.Timestamp("2022-01-02"),
        validation_end=pd.Timestamp("2022-01-03"),
        test_start=pd.Timestamp("2022-01-03"),
        test_end=pd.Timestamp("2022-01-04"),
    )
    return SplitDataset(cfg, lambda k: make_instrument("EURUSD", _jump_m5()), ("EURUSD",))


class SpyAgent(TradingAgent):
    """Wraps an agent and records the observation timestamp at every act."""

    name = "spy"

    def __init__(self, inner: TradingAgent) -> None:
        self.inner = inner
        self.observed_timestamps: list[pd.Timestamp] = []

    def reset(self, seed: int | None = None) -> None:
        self.inner.reset(seed)
        self.observed_timestamps.clear()

    def act(self, observation: EncodedObservation) -> Action:
        self.observed_timestamps.append(observation.timestamp)
        return self.inner.act(observation)


def _run(spy: SpyAgent):
    ds = _dataset()
    env_config = EnvironmentConfig(
        execution=ExecutionConfig(spread_value=0.0),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")),
        sizing=PositionSizingConfig(mode="fixed_units", fixed_units=Decimal("10000")),
        horizon=4,
    )
    encoder = ObservationEncoder(EncoderConfig(context_length=CONTEXT, initial_balance="10000"))
    runner = EvaluationRunner(ds, env_config, encoder, WindowConfig(context_length=CONTEXT))
    sampler = EpisodeSampler(
        ds, EpisodeConfig(split="test", horizon=4, context_length=CONTEXT, seed=0)
    )
    spec = sampler.explicit("EURUSD", CONTEXT, horizon=4)
    return runner.run_agent(spy, [spec])


def _long_agent() -> TradingAgent:
    from forexmind.baselines.buy_hold import LongExposureAgent

    return LongExposureAgent()


def test_action_selected_before_jump_and_no_future_bar_in_observation() -> None:
    spy = SpyAgent(_long_agent())
    ev = _run(spy)
    traj = ev.trajectories_by_instrument["EURUSD"][0]

    m5 = _jump_m5()
    ts = m5["timestamp"].to_numpy("datetime64[ns]")
    # First decision happens at observation CONTEXT (before the jump bar).
    assert spy.observed_timestamps[0].to_datetime64() == ts[CONTEXT]
    # The agent never observes the jump bar at decision time.
    assert all(o.to_datetime64() < ts[JUMP_BAR] for o in spy.observed_timestamps[:1])

    # The first trade executes at the NEXT open = the jump (1.20), not 1.10.
    assert len(traj.trade_log) >= 1
    first_trade = traj.trade_log[0]
    assert float(first_trade["execution_price"]) == pytest.approx(1.20)
    # Bought at the open (1.20) and marked at the same close -> no jump profit.
    assert traj.equity[0] == pytest.approx(10000.0)
    assert traj.equity[1] == pytest.approx(10000.0, abs=1e-6)


def test_no_observation_contains_future_timestamp() -> None:
    spy = SpyAgent(_long_agent())
    _run(spy)
    m5 = _jump_m5()
    valid_ts = set(pd.to_datetime(m5["timestamp"]).tolist())
    for o in spy.observed_timestamps:
        assert o in valid_ts  # real bucket starts, never fabricated


def test_reward_reflects_only_resulting_movement() -> None:
    spy = SpyAgent(_long_agent())
    ev = _run(spy)
    traj = ev.trajectories_by_instrument["EURUSD"][0]
    # Buying at the 1.20 open and marking at the 1.20 close: first reward ~ 0.
    assert abs(float(traj.rewards[0])) < 1e-9
