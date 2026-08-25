"""Tests for the dataset validator and gap classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from forexmind.data.schema import CLOSE, HIGH, LOW, OPEN, TIMESTAMP
from forexmind.data.validator import (
    GapConfig,
    GapType,
    MarketDataValidator,
    classify_gap,
)

# Monday 2025-01-06 (dayofweek 0). Friday 2025-01-03 is dayofweek 4.
MON = pd.Timestamp("2025-01-06 00:00")
FRI = pd.Timestamp("2025-01-03 23:59")
SUN = pd.Timestamp("2025-01-05 17:00")


def _frame(timestamps: list[pd.Timestamp], *, duplicate: bool = False) -> pd.DataFrame:
    data = {
        TIMESTAMP: timestamps,
        OPEN: [1.10] * len(timestamps),
        HIGH: [1.11] * len(timestamps),
        LOW: [1.09] * len(timestamps),
        CLOSE: [1.105] * len(timestamps),
    }
    df = pd.DataFrame(data)
    if duplicate:
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


def test_clean_frame_passes() -> None:
    ts = pd.date_range(MON, periods=10, freq="min")
    result = MarketDataValidator().validate(_frame(list(ts)), "EURUSD")
    assert result.is_valid
    assert result.errors == ()
    assert result.row_count == 10


def test_duplicate_timestamps_error() -> None:
    ts = pd.date_range(MON, periods=5, freq="min")
    df = _frame(list(ts), duplicate=True)
    result = MarketDataValidator().validate(df)
    assert not result.is_valid
    assert any(i.code == "duplicate_timestamps" for i in result.errors)


def test_duplicate_bar_error() -> None:
    ts = list(pd.date_range(MON, periods=5, freq="min"))
    df = pd.concat([_frame(ts), _frame([ts[0]])], ignore_index=True)
    result = MarketDataValidator().validate(df)
    codes = [i.code for i in result.errors]
    assert "duplicate_timestamps" in codes
    assert "duplicate_bars" in codes


def test_unsorted_error() -> None:
    ts = [MON, MON + pd.Timedelta(minutes=2), MON + pd.Timedelta(minutes=1)]
    result = MarketDataValidator().validate(_frame(ts))
    assert not result.is_valid
    assert any(i.code == "unsorted_timestamps" for i in result.errors)


def test_nan_error() -> None:
    df = _frame(list(pd.date_range(MON, periods=3, freq="min")))
    df.loc[1, OPEN] = np.nan
    result = MarketDataValidator().validate(df)
    assert any(i.code == "nan_value" for i in result.errors)


def test_infinite_error() -> None:
    df = _frame(list(pd.date_range(MON, periods=3, freq="min")))
    df.loc[1, HIGH] = float("inf")
    result = MarketDataValidator().validate(df)
    assert any(i.code == "infinite_value" for i in result.errors)


def test_ohlc_inconsistent_error() -> None:
    df = _frame(list(pd.date_range(MON, periods=3, freq="min")))
    df.loc[1, LOW] = 1.2  # low above high
    result = MarketDataValidator().validate(df)
    assert any(i.code == "ohlc_inconsistent" for i in result.errors)


def test_non_positive_price_error() -> None:
    df = _frame(list(pd.date_range(MON, periods=3, freq="min")))
    df.loc[0, CLOSE] = -0.5
    result = MarketDataValidator().validate(df)
    assert not result.is_valid


def test_classify_gap_types() -> None:
    cfg = GapConfig(bar_interval=1, large_gap_minutes=60)
    assert classify_gap(MON, MON + pd.Timedelta(minutes=1), cfg) == GapType.NORMAL
    assert classify_gap(MON, MON + pd.Timedelta(minutes=3), cfg) == GapType.SHORT_GAP
    assert classify_gap(MON, MON + pd.Timedelta(minutes=59), cfg) == GapType.SHORT_GAP
    assert classify_gap(MON, MON + pd.Timedelta(minutes=60), cfg) == GapType.LARGE_GAP
    assert classify_gap(FRI, SUN, cfg) == GapType.WEEKEND_GAP
    assert classify_gap(FRI, MON, cfg) == GapType.WEEKEND_GAP
    # Sunday -> Monday is continuous trading, NOT a weekend gap.
    sun_late = pd.Timestamp("2025-01-05 23:59")
    assert classify_gap(sun_late, MON, cfg) == GapType.NORMAL
    assert classify_gap(sun_late, MON + pd.Timedelta(minutes=3), cfg) == GapType.SHORT_GAP


def test_gap_statistics() -> None:
    # Normal 1-min bars, then a missing-minute short gap, a Monday->Friday
    # large gap, a weekend gap, then another large gap.
    ts = [MON + pd.Timedelta(minutes=i) for i in range(5)]  # Mon 00:00..00:04
    ts.append(MON + pd.Timedelta(minutes=7))  # missing 00:05/00:06 -> SHORT_GAP
    ts.append(FRI)  # Mon -> Fri -> LARGE_GAP
    ts.append(SUN)  # Fri -> Sun -> WEEKEND_GAP
    ts.append(SUN + pd.Timedelta(minutes=120))  # LARGE_GAP
    result = MarketDataValidator().validate(_frame(ts))
    counts = result.gap_counts
    assert counts["normal"] >= 4
    assert counts["short_gap"] >= 1
    assert counts["weekend_gap"] >= 1
    assert counts["large_gap"] >= 1


def test_first_bar_is_unknown() -> None:
    ts = list(pd.date_range(MON, periods=2, freq="min"))
    result = MarketDataValidator().validate(_frame(ts))
    assert result.gaps[0].gap_type == GapType.UNKNOWN
