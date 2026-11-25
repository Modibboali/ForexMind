"""Tests for the canonical market-bar schema and validation."""

from __future__ import annotations

import pandas as pd
import pytest
from forexmind.data.schema import (
    MarketBar,
    SchemaError,
    bar_from_row,
    validate_ohlc,
)


def _ts() -> pd.Timestamp:
    return pd.Timestamp("2025-01-06 10:00")


def test_valid_bar() -> None:
    bar = MarketBar(_ts(), 1.1000, 1.1010, 1.0990, 1.1005)
    assert bar.open == 1.1000
    assert bar.high == 1.1010
    assert bar.low == 1.0990
    assert bar.close == 1.1005


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(open=1.0, high=0.9, low=0.8, close=0.95),  # high < open
        dict(open=1.0, high=1.1, low=0.8, close=1.2),  # high < close
        dict(open=1.0, high=1.1, low=0.8, close=0.7),  # low > close
        dict(open=1.0, high=1.1, low=1.05, close=1.0),  # low > open
        dict(open=1.0, high=1.1, low=1.2, close=1.0),  # low > high
        dict(open=0.0, high=1.1, low=0.9, close=1.0),  # non-positive open
        dict(open=1.0, high=-1.0, low=0.9, close=1.0),  # non-positive high
        dict(open=1.0, high=1.1, low=-0.5, close=1.0),  # non-positive low
        dict(open=1.0, high=1.1, low=0.9, close=0.0),  # non-positive close
    ],
)
def test_invalid_bar_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(SchemaError):
        MarketBar(_ts(), **kwargs)  # type: ignore[arg-type]


def test_validate_ohlc_ok() -> None:
    validate_ohlc(_ts(), 1.1000, 1.1010, 1.0990, 1.1005)


def test_bar_from_row() -> None:
    row = {"timestamp": _ts(), "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15}
    bar = bar_from_row(row)
    assert bar.high == 1.2


def test_bar_is_immutable() -> None:
    bar = MarketBar(_ts(), 1.1, 1.2, 1.0, 1.15)
    with pytest.raises(AttributeError):
        bar.open = 2.0  # type: ignore[misc]
