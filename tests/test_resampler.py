"""Tests for the deterministic M1 -> M5 resampler."""

from __future__ import annotations

import pandas as pd
from forexmind.data.resampler import (
    IS_COMPLETE,
    N_OBSERVATIONS,
    CompletenessPolicy,
    ResampleConfig,
    resample_m1_to_m5,
)
from forexmind.data.schema import CLOSE, HIGH, LOW, OPEN, TIMESTAMP
from forexmind.data.validator import MarketDataValidator

from tests.synthetic import ladder_m1, m1_frame, ohlc_m1


def test_strict_resample_basic() -> None:
    # 25 contiguous 1-min bars with increasing prices.
    prices = [1.1000 + i * 0.001 for i in range(25)]
    m1 = ladder_m1("2025-01-06 00:00", prices)
    m5 = resample_m1_to_m5(m1, ResampleConfig(completeness=CompletenessPolicy.STRICT))

    assert len(m5) == 5
    assert list(m5[TIMESTAMP]) == list(pd.date_range("2025-01-06 00:00", periods=5, freq="5min"))
    # open = first, close = last, high = max, low = min within each bucket.
    assert m5.iloc[0][OPEN] == prices[0]
    assert m5.iloc[0][CLOSE] == prices[4]
    assert m5.iloc[0][HIGH] == max(prices[0:5])
    assert m5.iloc[0][LOW] == min(prices[0:5])
    assert m5.iloc[1][OPEN] == prices[5]
    assert m5.iloc[1][CLOSE] == prices[9]
    assert all(m5[N_OBSERVATIONS] == 5)
    assert all(m5[IS_COMPLETE])


def test_aggregation_rules() -> None:
    # 5 M1 bars: open should be first open, high max high, low min low, close last close.
    bars = [
        (1.1000, 1.1010, 1.0990, 1.1005),
        (1.1005, 1.1020, 1.1000, 1.1015),
        (1.1015, 1.1030, 1.1005, 1.1020),
        (1.1020, 1.1025, 1.0995, 1.1000),
        (1.1000, 1.1010, 1.0990, 1.1010),
    ]
    m1 = ohlc_m1("2025-01-06 00:00", bars)
    m5 = resample_m1_to_m5(m1, ResampleConfig(completeness=CompletenessPolicy.STRICT))
    assert len(m5) == 1
    row = m5.iloc[0]
    assert row[OPEN] == 1.1000
    assert row[HIGH] == 1.1030
    assert row[LOW] == 1.0990
    assert row[CLOSE] == 1.1010


def test_strict_drops_incomplete_bucket() -> None:
    # 9 bars: bucket0 complete (00:00-00:04), bucket1 has 4 bars (missing 00:08).
    m1 = m1_frame(
        [
            ("2025-01-06 00:00", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-06 00:01", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-06 00:02", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-06 00:03", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-06 00:04", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-06 00:05", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-06 00:06", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-06 00:07", 1.10, 1.10, 1.10, 1.10),
            # missing 00:08 -> bucket1 incomplete
            ("2025-01-06 00:09", 1.10, 1.10, 1.10, 1.10),
        ]
    )
    m5 = resample_m1_to_m5(m1, ResampleConfig(completeness=CompletenessPolicy.STRICT))
    assert len(m5) == 1
    assert m5.iloc[0][TIMESTAMP] == pd.Timestamp("2025-01-06 00:00")


def test_partial_keeps_incomplete_buckets() -> None:
    m1 = m1_frame(
        [
            ("2025-01-06 00:00", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-06 00:01", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-06 00:02", 1.10, 1.10, 1.10, 1.10),
            # bucket0 incomplete (3/5)
            ("2025-01-06 00:05", 1.20, 1.20, 1.20, 1.20),
            ("2025-01-06 00:06", 1.20, 1.20, 1.20, 1.20),
            ("2025-01-06 00:07", 1.20, 1.20, 1.20, 1.20),
            ("2025-01-06 00:08", 1.20, 1.20, 1.20, 1.20),
            ("2025-01-06 00:09", 1.20, 1.20, 1.20, 1.20),
        ]
    )
    m5 = resample_m1_to_m5(m1, ResampleConfig(completeness=CompletenessPolicy.PARTIAL))
    assert len(m5) == 2
    assert int(m5.iloc[0][N_OBSERVATIONS]) == 3
    assert not bool(m5.iloc[0][IS_COMPLETE])
    assert int(m5.iloc[1][N_OBSERVATIONS]) == 5
    assert bool(m5.iloc[1][IS_COMPLETE])
    assert m5.iloc[1][OPEN] == 1.20


def test_weekend_gap_not_bridged() -> None:
    # Friday 23:56..23:59, then Sunday 17:00..17:04 (STRICT).
    m1 = m1_frame(
        [
            ("2025-01-03 23:56", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-03 23:57", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-03 23:58", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-03 23:59", 1.10, 1.10, 1.10, 1.10),
            ("2025-01-05 17:00", 1.11, 1.11, 1.11, 1.11),
            ("2025-01-05 17:01", 1.11, 1.11, 1.11, 1.11),
            ("2025-01-05 17:02", 1.11, 1.11, 1.11, 1.11),
            ("2025-01-05 17:03", 1.11, 1.11, 1.11, 1.11),
            ("2025-01-05 17:04", 1.11, 1.11, 1.11, 1.11),
        ]
    )
    # Validate first (must classify the weekend gap).
    result = MarketDataValidator().validate(m1)
    assert any(g.gap_type.value == "weekend_gap" for g in result.gaps)

    m5 = resample_m1_to_m5(m1, ResampleConfig(completeness=CompletenessPolicy.STRICT))
    # Friday bucket 23:55 is incomplete (only 4 bars) -> dropped.
    # Sunday bucket 17:00 is complete -> kept, with close 1.11 (no bridging).
    assert len(m5) == 1
    assert m5.iloc[0][TIMESTAMP] == pd.Timestamp("2025-01-05 17:00")
    assert m5.iloc[0][OPEN] == 1.11


def test_empty_input() -> None:
    empty = m1_frame([])
    m5 = resample_m1_to_m5(empty)
    assert len(m5) == 0
