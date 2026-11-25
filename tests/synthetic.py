"""Small deterministic synthetic OHLC builders for unit tests.

These let us verify exact PnL and execution calculations without relying on
the real historical data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal

import numpy as np
import pandas as pd
from forexmind.data.dataset import InstrumentData
from forexmind.data.schema import CLOSE, HIGH, LOW, OPEN, TIMESTAMP
from forexmind.data.splits import SplitConfig, SplitDataset
from forexmind.environment.state import (
    AccountState,
    Session,
    TimeInfo,
)
from forexmind.environment.state import (
    Observation as Phase1Observation,
)


def m1_frame(bars: Sequence[tuple]) -> pd.DataFrame:
    """Build a canonical M1 frame from ``(timestamp_str, o, h, l, c)`` rows."""
    rows = [(pd.Timestamp(t), float(o), float(h), float(lo), float(c)) for (t, o, h, lo, c) in bars]
    return pd.DataFrame(rows, columns=[TIMESTAMP, OPEN, HIGH, LOW, CLOSE])


def constant_m1(start: str, n: int, price: float) -> pd.DataFrame:
    """``n`` contiguous 1-minute bars at a constant price."""
    ts = pd.date_range(start, periods=n, freq="min")
    return pd.DataFrame(
        {
            TIMESTAMP: ts,
            OPEN: [price] * n,
            HIGH: [price] * n,
            LOW: [price] * n,
            CLOSE: [price] * n,
        }
    )


def ladder_m1(start: str, prices: Sequence[float]) -> pd.DataFrame:
    """Contiguous 1-minute bars whose open/high/low/close equal ``prices[i]``."""
    ts = pd.date_range(start, periods=len(prices), freq="min")
    return pd.DataFrame(
        {
            TIMESTAMP: ts,
            OPEN: list(prices),
            HIGH: list(prices),
            LOW: list(prices),
            CLOSE: list(prices),
        }
    )


def ohlc_m1(
    start: str,
    prices: Sequence[tuple[float, float, float, float]],
) -> pd.DataFrame:
    """Contiguous 1-minute bars from ``(open, high, low, close)`` tuples."""
    ts = pd.date_range(start, periods=len(prices), freq="min")
    o = [p[0] for p in prices]
    h = [p[1] for p in prices]
    lo = [p[2] for p in prices]
    c = [p[3] for p in prices]
    return pd.DataFrame({TIMESTAMP: ts, OPEN: o, HIGH: h, LOW: lo, CLOSE: c})


def with_gaps(
    frame: pd.DataFrame,
    drop_minutes: Iterable[str],
) -> pd.DataFrame:
    """Drop rows whose timestamp (as ``'%H:%M'``) is in ``drop_minutes``."""
    keep = ~frame[TIMESTAMP].dt.strftime("%H:%M").isin(set(drop_minutes))
    return frame.loc[keep].reset_index(drop=True)


def weekday_series(start: str, n_minutes: int) -> pd.DatetimeIndex:
    """Contiguous minute timestamps starting from ``start`` (assume a weekday)."""
    return pd.date_range(start, periods=n_minutes, freq="min")


def simple_rising_m1(start: str, n: int, start_price: float, step: float = 0.001) -> pd.DataFrame:
    """Monotonic rising 1-minute bars: close = open + step, o==l, c==h."""
    o = [start_price + step * i for i in range(n)]
    c = [p + step for p in o]
    h = c
    lo = o
    return pd.DataFrame(
        {
            TIMESTAMP: pd.date_range(start, periods=n, freq="min"),
            OPEN: o,
            HIGH: h,
            LOW: lo,
            CLOSE: c,
        }
    )


# ---------------------------------------------------------------------------
# Phase 2 synthetic helpers
# ---------------------------------------------------------------------------


def m5_ohlc(start: str, bars: Sequence[tuple[float, float, float, float]]) -> pd.DataFrame:
    """M5 frame from ``(open, high, low, close)`` tuples, 5-minute spaced."""
    ts = pd.date_range(start, periods=len(bars), freq="5min")
    return pd.DataFrame(
        {
            TIMESTAMP: ts,
            OPEN: [b[0] for b in bars],
            HIGH: [b[1] for b in bars],
            LOW: [b[2] for b in bars],
            CLOSE: [b[3] for b in bars],
        }
    )


def m5_from_closes(start: str, closes: Sequence[float]) -> pd.DataFrame:
    """M5 frame where open=high=low=close for each bar."""
    return m5_ohlc(start, [(c, c, c, c) for c in closes])


def m5_constant(start: str, n: int, price: float) -> pd.DataFrame:
    return m5_from_closes(start, [price] * n)


def m5_trend(start: str, n: int, start_price: float, step: float = 0.001) -> pd.DataFrame:
    """Monotonic M5 series (perfect uptrend when step > 0)."""
    closes = [start_price + step * i for i in range(n)]
    return m5_from_closes(start, closes)


def m5_mean_reverting(
    start: str, n: int, mean: float, amplitude: float = 0.005, period: int = 40
) -> pd.DataFrame:
    """Deterministic sinusoidal (mean-reverting) M5 series."""
    closes = [mean + amplitude * np.sin(2 * np.pi * i / period) for i in range(n)]
    return m5_from_closes(start, closes)


def synth_m1_from_m5(m5: pd.DataFrame) -> pd.DataFrame:
    """Synthesise a consistent M1 frame from an M5 frame (5 identical M1 bars
    per bucket), so the Phase-1 environment can execute on it."""
    rows = []
    for _, row in m5.iterrows():
        base = pd.Timestamp(row[TIMESTAMP])
        for k in range(5):
            t = base + pd.Timedelta(minutes=k)
            rows.append((t, row[OPEN], row[HIGH], row[LOW], row[CLOSE]))
    return pd.DataFrame(rows, columns=[TIMESTAMP, OPEN, HIGH, LOW, CLOSE])


def make_instrument(
    instrument: str,
    m5: pd.DataFrame,
    m1: pd.DataFrame | None = None,
) -> InstrumentData:
    m1 = m1 if m1 is not None else synth_m1_from_m5(m5)
    return InstrumentData(instrument=instrument.upper(), m1=m1, m5=m5)


def make_test_split_config() -> SplitConfig:
    """A compact split config spanning 2020-2022 for fast unit tests."""
    return SplitConfig(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2021-01-01"),
        validation_start=pd.Timestamp("2021-01-01"),
        validation_end=pd.Timestamp("2022-01-01"),
        test_start=pd.Timestamp("2022-01-01"),
        test_end=pd.Timestamp("2023-01-01"),
    )


def make_split_dataset(
    instrument_data: Mapping[str, InstrumentData],
    split_config: SplitConfig | None = None,
) -> SplitDataset:
    """A SplitDataset backed by in-memory InstrumentData (dict-based loader)."""
    cfg = split_config or make_test_split_config()
    data = {k.upper(): v for k, v in instrument_data.items()}
    return SplitDataset(cfg, lambda k: data[k.upper()], tuple(data))


def make_phase1_observation(
    instrument: str,
    timestamp: pd.Timestamp,
    *,
    step_index: int = 0,
    equity: float = 10000.0,
    balance: float = 10000.0,
    position_units: float = 0.0,
    entry_price: float = 0.0,
    unrealized_pnl: float = 0.0,
    realized_pnl: float = 0.0,
    gross_exposure: float = 0.0,
    margin_used: float = 0.0,
    free_margin: float = 10000.0,
    drawdown: float = 0.0,
    hour: int | None = None,
    minute: int | None = None,
    day_of_week: int | None = None,
    session: Session = Session.UNKNOWN,
    minutes_since_last_bar: int = 5,
    is_weekend_gap: bool = False,
    market_window: tuple = (),
) -> Phase1Observation:
    """Build a Phase-1 observation with the given account/time state."""
    ts = pd.Timestamp(timestamp)
    account = AccountState(
        balance=Decimal(str(balance)),
        equity=Decimal(str(equity)),
        position_units=Decimal(str(position_units)),
        entry_price=Decimal(str(entry_price)),
        unrealized_pnl=Decimal(str(unrealized_pnl)),
        realized_pnl=Decimal(str(realized_pnl)),
        gross_exposure=Decimal(str(gross_exposure)),
        margin_used=Decimal(str(margin_used)),
        free_margin=Decimal(str(free_margin)),
        drawdown=Decimal(str(drawdown)),
    )
    time_info = TimeInfo(
        timestamp=ts,
        hour=ts.hour if hour is None else hour,
        minute=ts.minute if minute is None else minute,
        day_of_week=ts.dayofweek if day_of_week is None else day_of_week,
        session=session,
        minutes_since_last_bar=minutes_since_last_bar,
        is_weekend_gap=is_weekend_gap,
    )
    return Phase1Observation(
        instrument=instrument,
        step_index=step_index,
        timestamp=ts,
        market_window=tuple(market_window),
        account=account,
        time=time_info,
    )


def timeline_m5(dates: Sequence[str], per_day: int = 20) -> pd.DataFrame:
    """M5 bars for a few specific dates (5-minute spaced, ``per_day`` bars each)."""
    rows = []
    for d in dates:
        base = pd.Timestamp(d)
        for i in range(per_day):
            rows.append((base + pd.Timedelta(minutes=5 * i), 1.10, 1.10, 1.10, 1.10))
    return pd.DataFrame(rows, columns=[TIMESTAMP, OPEN, HIGH, LOW, CLOSE])
