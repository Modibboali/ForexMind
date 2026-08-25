"""Small deterministic synthetic OHLC builders for unit tests.

These let us verify exact PnL and execution calculations without relying on
the real historical data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd
from forexmind.data.schema import CLOSE, HIGH, LOW, OPEN, TIMESTAMP


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
