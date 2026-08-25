"""Tests for the Gymnasium-style Forex environment (reset/step/termination,
no future leakage, reproducibility, multi-instrument switching)."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest
from forexmind.config import (
    EnvironmentConfig,
    ExecutionConfig,
    MarginConfig,
    PositionSizingConfig,
)
from forexmind.data.dataset import InstrumentData, MarketDataset
from forexmind.environment import EnvironmentError, ForexEnvironment

from tests.synthetic import constant_m1, ladder_m1

REQUIRED_INFO_KEYS = {
    "timestamp",
    "instrument",
    "equity",
    "balance",
    "position",
    "unrealized_pnl",
    "realized_pnl",
    "trade_cost",
    "execution_price",
    "drawdown",
    "margin_used",
}


def _dataset(
    instrument: str = "EURUSD",
    n_minutes: int = 60,
    price: float = 1.10,
    prices: list[float] | None = None,
) -> MarketDataset:
    m1 = (
        ladder_m1("2025-01-06 00:00", prices)
        if prices
        else constant_m1("2025-01-06 00:00", n_minutes, price)
    )
    ds = MarketDataset()
    ds.add(InstrumentData.from_m1(instrument, m1))
    return ds


def _config(**overrides) -> EnvironmentConfig:
    return EnvironmentConfig(
        execution=ExecutionConfig(spread_value=0.0),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")),
        sizing=PositionSizingConfig(mode="fixed_units", fixed_units=Decimal("10000")),
        **overrides,
    )


def test_reset_returns_obs_and_info() -> None:
    env = ForexEnvironment(_dataset(), _config())
    obs, info = env.reset(seed=0, start_index=0, horizon=5)
    assert obs.instrument == "EURUSD"
    assert obs.step_index == 0
    assert obs.account.equity == Decimal("10000")
    assert obs.account.balance == Decimal("10000")
    assert obs.time.hour == 0
    assert obs.time.minute == 0
    assert len(obs.market_window) > 0
    assert set(info) >= REQUIRED_INFO_KEYS


def test_step_returns_five_tuple() -> None:
    env = ForexEnvironment(_dataset(), _config())
    env.reset(seed=0, start_index=0, horizon=5)
    obs, reward, terminated, truncated, info = env.step(2)  # flat
    assert not terminated
    assert not truncated
    assert reward == 0.0
    assert obs.step_index == 1
    assert set(info) >= REQUIRED_INFO_KEYS


def test_step_to_long_opens_position() -> None:
    env = ForexEnvironment(_dataset(), _config())
    env.reset(seed=0, start_index=0, horizon=5)
    _obs, _reward, _terminated, _truncated, info = env.step(4)  # +1.0 full long
    assert info["position"] == "long"
    assert info["position_units"] == Decimal("10000")
    assert info["execution_price"] == Decimal("1.10")
    assert not _terminated and not _truncated


def test_equity_fraction_sizing() -> None:
    cfg = _config()
    cfg = EnvironmentConfig(
        execution=ExecutionConfig(spread_value=0.0),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")),
        sizing=PositionSizingConfig(mode="equity_fraction", fixed_units=Decimal("10000")),
        horizon=5,
    )
    env = ForexEnvironment(_dataset(price=1.10), cfg)
    env.reset(seed=0, start_index=0)
    obs, *_ = env.step(4)
    # exposure 1.0 * equity 10000 / price 1.10
    assert obs.account.position_units == Decimal("10000") / Decimal("1.10")


def test_truncation_at_horizon() -> None:
    env = ForexEnvironment(_dataset(), _config(horizon=3))
    env.reset(seed=0, start_index=0)
    for _ in range(3):
        _obs, _r, terminated, truncated, _info = env.step(2)
    assert truncated
    assert terminated is False
    with pytest.raises(EnvironmentError):
        env.step(2)


def test_no_future_leakage_in_window() -> None:
    m5_ts = set(pd.date_range("2025-01-06 00:00", periods=12, freq="5min"))
    env = ForexEnvironment(_dataset(), _config())
    obs, _ = env.reset(seed=0, start_index=2, horizon=4)
    # Reset observation window must contain bars only up to index 2.
    assert {b.timestamp for b in obs.market_window} <= {
        t for t in m5_ts if t <= pd.Timestamp("2025-01-06 00:10")
    }
    obs, *_ = env.step(2)
    # After one step, window still only contains bars up to the new index.
    latest = max(b.timestamp for b in obs.market_window)
    assert latest == pd.Timestamp("2025-01-06 00:15")
    assert latest in m5_ts


def test_reproducibility_across_instances() -> None:
    actions = [4, 3, 2, 1, 0, 4, 4, 3, 2, 0]
    results = []
    for _ in range(2):
        env = ForexEnvironment(_dataset(n_minutes=120), _config(horizon=10))
        obs, info = env.reset(seed=42, start_index=3)
        eq = [obs.account.equity]
        rw = []
        for a in actions:
            obs, reward, _terminated, _truncated, info = env.step(a)
            eq.append(obs.account.equity)
            rw.append(reward)
        results.append((eq, rw, info["equity"]))
    assert results[0] == results[1]


def test_same_config_different_seed_is_still_deterministic_given_start() -> None:
    env1 = ForexEnvironment(_dataset(), _config(horizon=5))
    env2 = ForexEnvironment(_dataset(), _config(horizon=5))
    obs1, _ = env1.reset(seed=1, start_index=2)
    obs2, _ = env2.reset(seed=999, start_index=2)
    assert obs1.account.equity == obs2.account.equity
    assert obs1.timestamp == obs2.timestamp


def test_multi_instrument_switch() -> None:
    ds = MarketDataset()
    ds.add(InstrumentData.from_m1("EURUSD", constant_m1("2025-01-06 00:00", 60, 1.10)))
    ds.add(InstrumentData.from_m1("GBPUSD", constant_m1("2025-01-06 00:00", 60, 1.26)))
    env = ForexEnvironment(ds, _config(horizon=3))
    obs1, _ = env.reset(seed=0, instrument="EURUSD", start_index=0)
    assert obs1.instrument == "EURUSD"
    obs2, _ = env.reset(seed=0, instrument="GBPUSD", start_index=0)
    assert obs2.instrument == "GBPUSD"
    # Same simulator code, different instrument price level reflected in obs.
    assert obs1.account.equity == obs2.account.equity == Decimal("10000")


def test_liquidation_terminates_episode() -> None:
    # Big fixed size + a price drop triggers the deterministic liquidation.
    cfg = EnvironmentConfig(
        execution=ExecutionConfig(spread_value=0.0),
        margin=MarginConfig(
            initial_balance=Decimal("10000"),
            leverage=Decimal("100"),
            maintenance_margin_ratio=Decimal("0.99"),
        ),
        sizing=PositionSizingConfig(mode="fixed_units", fixed_units=Decimal("200000")),
        horizon=10,
    )
    # 00:00-00:04 at 1.10 (M5[0]), 00:05-00:09 at 1.095 (M5[1]),
    # 00:10+ at 1.05 (M5[2]) -> big unrealised loss after holding long.
    prices = [1.10] * 5 + [1.095] * 5 + [1.05] * 20
    m1 = ladder_m1("2025-01-06 00:00", prices)
    ds = MarketDataset()
    ds.add(InstrumentData.from_m1("EURUSD", m1))
    env = ForexEnvironment(ds, cfg)
    env.reset(seed=0, start_index=0)
    _obs, _r0, t0, _tr0, info0 = env.step(4)  # go long at 00:05 open 1.095
    assert info0["position"] == "long"
    assert not t0
    _obs, _r1, t1, _tr1, info1 = env.step(4)  # hold; marked at 00:10+ close 1.05
    assert t1, "liquidation should terminate the episode"
    assert info1["liquidation"] is True
    assert info1["position"] == "flat"
    with pytest.raises(EnvironmentError):
        env.step(4)


def test_action_sequence_produces_pnl() -> None:
    # Rising market: be long early, then flatten -> positive realised PnL.
    prices = [1.10 + 0.01 * (i // 5) for i in range(60)]
    ds = MarketDataset()
    ds.add(InstrumentData.from_m1("EURUSD", ladder_m1("2025-01-06 00:00", prices)))
    env = ForexEnvironment(ds, _config(horizon=8, close_at_episode_end=True))
    env.reset(seed=0, start_index=0)
    for i in range(8):
        action = 4 if i < 6 else 2  # full long, then flat at the end
        _obs, _reward, _terminated, truncated, _info = env.step(action)
        if truncated:
            break
    assert env.portfolio is not None
    assert env.portfolio.position.is_flat  # closed at episode end
    assert env.portfolio.realized_pnl > 0


def test_close_at_episode_end_option() -> None:
    ds = MarketDataset()
    ds.add(InstrumentData.from_m1("EURUSD", constant_m1("2025-01-06 00:00", 60, 1.10)))
    env = ForexEnvironment(ds, _config(horizon=3, close_at_episode_end=True))
    env.reset(seed=0, start_index=0)
    for _ in range(3):
        _obs, _r, _term, _trunc, info = env.step(4)
    assert env.portfolio is not None
    assert env.portfolio.position.is_flat
    assert info["position"] == "flat"
