"""Canonical market-bar schema and immutable domain model.

Market prices are stored as ``float`` in the market-data layer.  This matches
the precision of the raw quote files (5-6 significant digits) and is the
standard representation for pandas/parquet pipelines.  All *accounting*
(balance, PnL, margin, costs) uses :class:`decimal.Decimal`; see
``forexmind.environment.portfolio``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

# Canonical column names used across the whole codebase.
TIMESTAMP: Final[str] = "timestamp"
OPEN: Final[str] = "open"
HIGH: Final[str] = "high"
LOW: Final[str] = "low"
CLOSE: Final[str] = "close"

CANONICAL_COLUMNS: Final[tuple[str, ...]] = (TIMESTAMP, OPEN, HIGH, LOW, CLOSE)
OHLC_COLUMNS: Final[tuple[str, ...]] = (OPEN, HIGH, LOW, CLOSE)


class SchemaError(ValueError):
    """Raised when market-bar data violates the canonical schema."""


def validate_ohlc(
    timestamp: pd.Timestamp,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> None:
    """Validate a single OHLC bar, raising :class:`SchemaError` on problems.

    Enforces the invariants required by the ForexMind spec:

    * all prices strictly positive
    * ``high >= open``, ``high >= close``, ``high >= low``
    * ``low <= open``, ``low <= close``
    """
    prices = {"open": open_, "high": high, "low": low, "close": close}
    for name, value in prices.items():
        if not isinstance(value, (int, float)):
            raise SchemaError(f"{timestamp}: column {name!r} is not numeric: {value!r}")
        if value <= 0:
            raise SchemaError(f"{timestamp}: {name} must be > 0, got {value}")

    if high < low:
        raise SchemaError(f"{timestamp}: high ({high}) < low ({low})")
    if high < open_:
        raise SchemaError(f"{timestamp}: high ({high}) < open ({open_})")
    if high < close:
        raise SchemaError(f"{timestamp}: high ({high}) < close ({close})")
    if low > open_:
        raise SchemaError(f"{timestamp}: low ({low}) > open ({open_})")
    if low > close:
        raise SchemaError(f"{timestamp}: low ({low}) > close ({close})")


@dataclass(frozen=True, slots=True)
class MarketBar:
    """An immutable single OHLC market bar."""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, pd.Timestamp):
            raise SchemaError(f"timestamp must be a pandas Timestamp, got {self.timestamp!r}")
        validate_ohlc(self.timestamp, self.open, self.high, self.low, self.close)


def bar_from_row(row: Mapping[str, Any], timestamp_col: str = TIMESTAMP) -> MarketBar:
    """Build a :class:`MarketBar` from a dict-like row in the canonical schema."""
    return MarketBar(
        timestamp=pd.Timestamp(row[timestamp_col]),
        open=float(row[OPEN]),
        high=float(row[HIGH]),
        low=float(row[LOW]),
        close=float(row[CLOSE]),
    )
