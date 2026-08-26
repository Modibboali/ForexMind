"""Formal observation model (Phase 2).

The agent receives a structured :class:`EncodedObservation` rather than a raw
dictionary: market window, account state, time metadata, and instrument
identity, each with clear shapes and dtypes.  ``encoded`` is an optional flat
concatenation intended for later neural encoders; the structured fields are
kept for debugging and for causal baseline indicators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Default market feature vector (see encoder.py).
DEFAULT_MARKET_FEATURES: tuple[str, ...] = (
    "open_return",
    "high_return",
    "low_return",
    "close_return",
    "log_return",
)
N_MARKET_FEATURES = len(DEFAULT_MARKET_FEATURES)

N_ACCOUNT_FEATURES = 10  # see encoder.py account feature list
N_TIME_FEATURES = 14  # hour/min/dow cyclic (6) + 2 scalars + session one-hot (6)

# Feature names (for inspection/debugging; order must match the encoder).
ACCOUNT_FEATURE_NAMES: tuple[str, ...] = (
    "position_exposure",
    "position_units_normalized",
    "entry_distance",
    "unrealized_pnl_normalized",
    "realized_pnl_normalized",
    "equity_return_from_initial",
    "drawdown_normalized",
    "margin_utilization",
    "free_margin_ratio",
    "leverage_used",
)

SESSION_FEATURE_NAMES: tuple[str, ...] = (
    "session_asia",
    "session_london",
    "session_new_york",
    "session_overlap",
    "session_quiet",
    "session_unknown",
)

TIME_FEATURE_NAMES: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "minutes_since_previous_bar_norm",
    "is_weekend_gap",
    *SESSION_FEATURE_NAMES,
)


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Describes the exact shapes of an :class:`EncodedObservation`."""

    context_length: int
    n_market_features: int = N_MARKET_FEATURES
    n_account_features: int = N_ACCOUNT_FEATURES
    n_time_features: int = N_TIME_FEATURES
    n_instruments: int = 7

    @property
    def market_shape(self) -> tuple[int, int]:
        return (self.context_length, self.n_market_features)

    @property
    def account_shape(self) -> tuple[int]:
        return (self.n_account_features,)

    @property
    def time_shape(self) -> tuple[int]:
        return (self.n_time_features,)

    @property
    def instrument_shape(self) -> tuple[int]:
        return (self.n_instruments,)

    @property
    def encoded_shape(self) -> tuple[int]:
        total = (
            self.context_length * self.n_market_features
            + self.n_account_features
            + self.n_time_features
            + self.n_instruments
        )
        return (total,)

    def to_dict(self) -> dict[str, object]:
        return {
            "context_length": self.context_length,
            "n_market_features": self.n_market_features,
            "n_account_features": self.n_account_features,
            "n_time_features": self.n_time_features,
            "n_instruments": self.n_instruments,
            "market_shape": list(self.market_shape),
            "account_shape": list(self.account_shape),
            "time_shape": list(self.time_shape),
            "instrument_shape": list(self.instrument_shape),
            "encoded_shape": list(self.encoded_shape),
        }


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    """Agent-facing observation.

    ``market`` contains the per-bar return features (shape
    ``(context_length, n_market_features)``) intended for a shared multi-pair
    model.  ``closes`` and ``prior_close`` keep the raw close levels so causal
    baseline indicators (SMA, mean reversion) can be computed without touching
    future data.
    """

    instrument: str
    step_index: int
    timestamp: pd.Timestamp
    spec: ObservationSpec
    market: np.ndarray  # (context_length, n_market_features)
    account: np.ndarray  # (n_account_features,)
    time: np.ndarray  # (n_time_features,)
    instrument_vec: np.ndarray  # (n_instruments,)
    closes: np.ndarray  # (context_length,) raw closes (baseline/analysis use)
    prior_close: float  # close of the bar immediately before the window

    @property
    def encoded(self) -> np.ndarray:
        """Flat concatenation for neural encoders (float32)."""
        return np.concatenate(
            [self.market.ravel(), self.account, self.time, self.instrument_vec],
            dtype=np.float32,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "instrument": self.instrument,
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "market": self.market,
            "closes": self.closes,
            "prior_close": self.prior_close,
            "account": self.account,
            "time": self.time,
            "instrument_vec": self.instrument_vec,
            "encoded": self.encoded,
            "spec": self.spec.to_dict(),
        }
