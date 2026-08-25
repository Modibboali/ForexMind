"""Tests for the instrument-aware dataset abstraction."""

from __future__ import annotations

import pandas as pd
import pytest
from forexmind.data.dataset import InstrumentData, MarketDataset

from tests.synthetic import constant_m1


def _eurusd() -> InstrumentData:
    return InstrumentData.from_m1("EURUSD", constant_m1("2025-01-06 00:00", 60, 1.10))


def _gbpusd() -> InstrumentData:
    return InstrumentData.from_m1("GBPUSD", constant_m1("2025-01-06 00:00", 120, 1.26))


def test_single_instrument() -> None:
    ds = MarketDataset()
    ds.add(_eurusd())
    assert ds.instruments == ["EURUSD"]
    assert "eurusd" in ds  # case-insensitive
    assert len(ds.m5("EURUSD")) == 12  # 60 minutes / 5


def test_multiple_instruments_independent() -> None:
    eur = _eurusd()
    gbp = _gbpusd()
    ds = MarketDataset()
    ds.add(eur)
    ds.add(gbp)
    assert ds.instruments == ["EURUSD", "GBPUSD"]
    # Each instrument keeps its own timeline; nothing is concatenated.
    assert len(ds.m1("EURUSD")) == 60
    assert len(ds.m1("GBPUSD")) == 120
    assert len(ds.m5("EURUSD")) == 12
    assert len(ds.m5("GBPUSD")) == 24
    # The two M5 timelines are independent (not one concatenated series).
    assert ds.m5("EURUSD")["timestamp"].iloc[0] == pd.Timestamp("2025-01-06 00:00")
    assert ds.m5("EURUSD")["timestamp"].iloc[-1] == pd.Timestamp("2025-01-06 00:55")
    assert ds.m5("GBPUSD")["timestamp"].iloc[0] == pd.Timestamp("2025-01-06 00:00")
    assert ds.m5("GBPUSD")["timestamp"].iloc[-1] == pd.Timestamp("2025-01-06 01:55")


def test_unknown_instrument_raises() -> None:
    ds = MarketDataset()
    ds.add(_eurusd())
    with pytest.raises(KeyError):
        ds.get("JPYUSD")


def test_duplicate_add_raises() -> None:
    ds = MarketDataset()
    ds.add(_eurusd())
    with pytest.raises(ValueError):
        ds.add(_eurusd())


def test_iteration_and_len() -> None:
    ds = MarketDataset()
    ds.add(_eurusd())
    ds.add(_gbpusd())
    assert len(ds) == 2
    assert [d.instrument for d in ds] == ["EURUSD", "GBPUSD"]


def test_build_m5_recomputes() -> None:
    ds = MarketDataset()
    ds.add(_eurusd())
    before = len(ds.m5("EURUSD"))
    ds.build_m5()
    assert len(ds.m5("EURUSD")) == before
