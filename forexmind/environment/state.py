"""Observation state model and session/time metadata.

The raw observation separates *simulator state* from any neural-network
feature representation.  No handcrafted technical indicators live here; the
simulator's job is to provide correct market/account/time state that a later
encoder can consume.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from forexmind.data.schema import MarketBar


class Session(enum.Enum):
    """Possible market-session labels.

    Session boundaries are approximate and depend on the assumed timezone of
    the source data (see README).  ``utc_offset_hours`` in
    :func:`session_for_timestamp` shifts the wall-clock used for labelling.
    """

    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "overlap"
    QUIET = "quiet"
    UNKNOWN = "unknown"


# (start_hour, end_hour_exclusive) per session, evaluated in precedence order.
_SESSION_WINDOWS: tuple[tuple[Session, int, int], ...] = (
    (Session.OVERLAP, 13, 16),  # London/NY overlap
    (Session.LONDON, 7, 13),
    (Session.NEW_YORK, 16, 21),
    (Session.ASIA, 0, 7),
    (Session.QUIET, 21, 24),
)


def session_for_timestamp(timestamp: pd.Timestamp, utc_offset_hours: int = 0) -> Session:
    """Classify an hour-of-day into a session label.

    ``utc_offset_hours`` shifts the source wall-clock to the assumed UTC time
    (default 0).  This is configurable because the source timezone is unknown
    for MT5 exports; no silent guess is made by the loader.
    """
    if not isinstance(timestamp, pd.Timestamp) or pd.isna(timestamp):
        return Session.UNKNOWN
    hour = (timestamp.hour + utc_offset_hours) % 24
    for session, start, end in _SESSION_WINDOWS:
        if start <= hour < end:
            return session
    return Session.UNKNOWN


@dataclass(frozen=True, slots=True)
class TimeInfo:
    """Temporal metadata exposed to later models."""

    timestamp: pd.Timestamp
    hour: int
    minute: int
    day_of_week: int  # 0=Monday .. 6=Sunday
    session: Session
    minutes_since_last_bar: int
    is_weekend_gap: bool
    utc_offset_hours: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "hour": self.hour,
            "minute": self.minute,
            "day_of_week": self.day_of_week,
            "session": self.session.value,
            "minutes_since_last_bar": self.minutes_since_last_bar,
            "is_weekend_gap": self.is_weekend_gap,
        }


@dataclass(frozen=True, slots=True)
class AccountState:
    """Raw account/portfolio state exposed to the observation."""

    balance: Decimal
    equity: Decimal
    position_units: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    gross_exposure: Decimal
    margin_used: Decimal
    free_margin: Decimal
    drawdown: Decimal

    def as_dict(self) -> dict[str, object]:
        return {
            "balance": self.balance,
            "equity": self.equity,
            "position_units": self.position_units,
            "entry_price": self.entry_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "gross_exposure": self.gross_exposure,
            "margin_used": self.margin_used,
            "free_margin": self.free_margin,
            "drawdown": self.drawdown,
        }


@dataclass(frozen=True, slots=True)
class Observation:
    """Raw simulator observation delivered to an agent.

    ``market_window`` is a tuple of the most recent ``window_size`` M5 bars
    (oldest -> newest) as immutable :class:`MarketBar` objects.  ``account``
    and ``time`` carry the raw financial and temporal state.  Feature
    engineering / encoding is deliberately left to a later layer.
    """

    instrument: str
    step_index: int
    timestamp: pd.Timestamp
    market_window: tuple[MarketBar, ...]
    account: AccountState
    time: TimeInfo

    def as_dict(self) -> dict[str, object]:
        return {
            "instrument": self.instrument,
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "market": [bar_from_marketbar(b) for b in self.market_window],
            "account": self.account.as_dict(),
            "time": self.time.as_dict(),
        }


def bar_from_marketbar(bar: MarketBar) -> dict[str, object]:
    return {
        "timestamp": bar.timestamp,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
    }
