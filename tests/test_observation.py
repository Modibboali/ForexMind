"""Tests for the market context window and observation encoder (Phase 2)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from forexmind.environment.state import Session
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.schema import (
    ACCOUNT_FEATURE_NAMES,
    N_ACCOUNT_FEATURES,
    N_TIME_FEATURES,
    EncodedObservation,
)
from forexmind.observation.window import (
    MarketWindowBuilder,
    WindowConfig,
    WindowError,
)

from tests.synthetic import (
    m5_from_closes,
    m5_trend,
    make_phase1_observation,
    make_test_split_config,
    timeline_m5,
)

WEEK = pd.Timestamp("2020-01-06")  # Monday


def _builder(m5, context=8, split_start=None, split_end=None):
    start = split_start or pd.Timestamp("2019-01-01")
    end = split_end or pd.Timestamp("2030-01-01")
    return MarketWindowBuilder("EURUSD", m5, start, end, WindowConfig(context_length=context))


def test_window_exact_shape_and_no_future() -> None:
    m5 = m5_trend("2020-01-06", 40, 1.10, 0.001)
    b = _builder(m5, context=8)
    win = b.build(20)
    assert len(win.closes) == 8
    assert win.current_index == 20
    # Window spans [20-8+1, 20].
    expected = m5["timestamp"].to_numpy(dtype="datetime64[ns]")[13:21]
    assert np.array_equal(win.timestamps, expected)
    assert win.prior_close == float(m5["close"].iloc[12])


def test_window_insufficient_history_at_beginning() -> None:
    m5 = m5_trend("2020-01-06", 40, 1.10, 0.001)
    b = _builder(m5, context=8)
    assert not b.is_eligible(7)  # need index >= context_length = 8
    assert b.is_eligible(8)
    with pytest.raises(WindowError):
        b.build(7)


def test_window_exactly_enough_history() -> None:
    m5 = m5_trend("2020-01-06", 40, 1.10, 0.001)
    b = _builder(m5, context=8)
    win = b.build(8)  # prior_close at index 0
    assert win.prior_close == float(m5["close"].iloc[0])
    assert len(win.closes) == 8


def test_window_strict_split_boundary() -> None:
    # m5 in train range only (2020); context 4 -> min valid = first_idx + 4.
    m5 = timeline_m5(["2020-01-06", "2020-06-01"], per_day=20)
    cfg = make_test_split_config()
    start, end = cfg.range("train")
    b = MarketWindowBuilder(
        "EURUSD", m5, start, end, WindowConfig(context_length=4, context_policy="strict_split")
    )
    first_idx = int(
        np.searchsorted(m5["timestamp"].to_numpy("datetime64[ns]"), np.datetime64(start))
    )
    assert b.min_valid_index() == first_idx + 4
    win = b.build(b.min_valid_index())
    # Entire window (and prior_close at first_idx) inside the split.
    assert win.timestamps[0] >= np.datetime64(start)
    assert m5["timestamp"].to_numpy("datetime64[ns]")[b.min_valid_index() - 4] >= np.datetime64(
        start
    )


def test_window_historical_warmup_allows_prefix() -> None:
    m5 = timeline_m5(["2020-01-06", "2020-06-01"], per_day=20)
    cfg = make_test_split_config()
    start, end = cfg.range("train")
    # Warmup policy: min valid is just context_length (uses data before split).
    b = MarketWindowBuilder(
        "EURUSD", m5, start, end, WindowConfig(context_length=4, context_policy="historical_warmup")
    )
    assert b.min_valid_index() == 4


def test_window_gap_metadata() -> None:
    closes = [1.10] * 10 + [1.11] * 10
    m5 = m5_from_closes("2020-01-06 00:00", closes)
    # Insert a large gap before index 10 (bar 09:50 -> 10:05 becomes 5h later).
    ts = m5["timestamp"].to_numpy(dtype="datetime64[ns]")
    ts[10:] = ts[10:] + np.timedelta64(300, "m")
    m5["timestamp"] = pd.to_datetime(ts)
    b = _builder(m5, context=8)
    win = b.build(14)
    gap_idx = np.flatnonzero(win.minutes_since_previous > 5)
    assert len(gap_idx) == 1
    assert win.minutes_since_previous[gap_idx[0]] > 300


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def _encoder(context=8):
    return ObservationEncoder(EncoderConfig(context_length=context, initial_balance="10000"))


def test_encoded_observation_shapes() -> None:
    m5 = m5_trend("2020-01-06", 40, 1.10, 0.001)
    b = _builder(m5, context=8)
    enc = _encoder(context=8)
    obs = make_phase1_observation("EURUSD", pd.Timestamp("2020-01-06 01:00"))
    window = b.build(20)
    e: EncodedObservation = enc.encode(obs, window)

    assert e.market.shape == (8, 5)
    assert e.account.shape == (N_ACCOUNT_FEATURES,)
    assert e.time.shape == (N_TIME_FEATURES,)
    assert e.instrument_vec.shape == (7,)
    expected_encoded = 8 * 5 + N_ACCOUNT_FEATURES + N_TIME_FEATURES + 7
    assert e.encoded.shape == (expected_encoded,)


def test_market_log_return_correct() -> None:
    closes = [1.10, 1.111, 1.12]
    m5 = m5_from_closes("2020-01-06", closes + [1.13] * 10)
    b = _builder(m5, context=3)
    enc = _encoder(context=3)
    obs = make_phase1_observation("EURUSD", pd.Timestamp("2020-01-06 00:20"))
    window = b.build(3)  # window bars 1..3, prior = bar 0
    e = enc.encode(obs, window)
    market = e.market
    # Bar 0 (index 1 in frame): base = prior close (1.10); close_return = 1.111/1.10 - 1.
    assert market[0, 3] == pytest.approx(1.111 / 1.10 - 1.0, rel=1e-6)
    assert market[0, 4] == pytest.approx(math.log(1.111 / 1.10), rel=1e-6)
    # Bar 1: base = close of bar 0 (1.111).
    assert market[1, 3] == pytest.approx(1.12 / 1.111 - 1.0, rel=1e-6)


def test_encoder_no_future_market() -> None:
    m5 = m5_trend("2020-01-06", 40, 1.10, 0.001)
    b = _builder(m5, context=8)
    enc = _encoder(context=8)
    obs = make_phase1_observation("EURUSD", pd.Timestamp("2020-01-06 01:00"))
    window = b.build(25)
    enc.encode(obs, window)
    # The last market bar corresponds to the current index (25), not beyond.
    assert window.timestamps[-1] == m5["timestamp"].to_numpy("datetime64[ns]")[25]
    assert np.all(window.timestamps <= m5["timestamp"].to_numpy("datetime64[ns]")[25])


def test_instrument_one_hot() -> None:
    enc = _encoder()
    for name, idx in {
        "EURUSD": 0,
        "GBPUSD": 1,
        "USDJPY": 2,
        "USDCHF": 3,
        "AUDUSD": 4,
        "USDCAD": 5,
        "NZDUSD": 6,
    }.items():
        vec = enc.encode_instrument(name)
        assert vec[idx] == 1.0
        assert int(np.sum(vec)) == 1


def test_time_features_cyclic() -> None:
    enc = _encoder()
    m5 = m5_trend("2020-01-06", 20, 1.10, 0.001)
    b = _builder(m5, context=4)
    obs = make_phase1_observation(
        "EURUSD",
        pd.Timestamp("2020-01-06 06:00"),
        hour=6,
        minute=0,
        day_of_week=0,
        session=Session.ASIA,
        minutes_since_last_bar=5,
    )
    e = enc.encode(obs, b.build(10))
    t = e.time
    # hour 6/24 -> sin(2*pi*6/24) = sin(pi/2) = 1
    assert t[0] == pytest.approx(1.0, abs=1e-6)  # hour_sin
    assert t[1] == pytest.approx(0.0, abs=1e-6)  # hour_cos
    assert t[6] == pytest.approx(5.0 / 1440.0)  # minutes_since_previous_norm
    assert t[7] == 0.0  # is_weekend_gap
    assert int(np.sum(t[8:])) == 1  # exactly one session one-hot


def test_account_features() -> None:
    enc = _encoder()
    m5 = m5_trend("2020-01-06", 20, 1.10, 0.001)
    b = _builder(m5, context=4)
    obs = make_phase1_observation(
        "EURUSD",
        pd.Timestamp("2020-01-06 01:00"),
        equity=10100,
        balance=10000,
        position_units=1000,
        entry_price=1.0,
        unrealized_pnl=100,
        realized_pnl=-50,
        gross_exposure=1100,
        margin_used=11,
        free_margin=10089,
        drawdown=50,
    )
    e = enc.encode(obs, b.build(10))
    a = e.account
    names = dict(zip(ACCOUNT_FEATURE_NAMES, a, strict=True))
    # equity_return_from_initial = 10100/10000 - 1 = 0.01
    assert names["equity_return_from_initial"] == pytest.approx(0.01)
    # position_exposure = sign * gross/equity
    assert names["position_exposure"] == pytest.approx(1100 / 10100)
    # unrealized normalized
    assert names["unrealized_pnl_normalized"] == pytest.approx(100 / 10000)
    # drawdown normalized
    assert names["drawdown_normalized"] == pytest.approx(50 / 10000)


def test_encoder_deterministic() -> None:
    m5 = m5_trend("2020-01-06", 40, 1.10, 0.001)
    b = _builder(m5, context=8)
    enc = _encoder(context=8)
    obs = make_phase1_observation("EURUSD", pd.Timestamp("2020-01-06 01:00"))
    e1 = enc.encode(obs, b.build(20))
    e2 = enc.encode(obs, b.build(20))
    assert np.array_equal(e1.encoded, e2.encoded)
