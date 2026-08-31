"""Instrument metadata registry (Phase 3.1 multi-currency accounting).

Every supported pair is described by an explicit :class:`InstrumentSpec`:

* ``base_currency`` / ``quote_currency`` — the pair is ``BASE/QUOTE``, so
  ``1 BASE = price QUOTE``.
* ``pip_size`` — the standard pip for the pair (explicit metadata, *not*
  derived from arbitrary price values).
* ``price_precision`` — typical display precision, informational only.

Phase 1 supplied OHLC data for seven USD-major instruments.  This registry is
the single source of truth for their currency orientation and pip metadata.
Unknown instruments are rejected rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from forexmind.config import _dec


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Explicit metadata for one currency pair."""

    instrument: str
    base_currency: str
    quote_currency: str
    pip_size: float
    price_precision: int

    @property
    def pip_decimal(self) -> Decimal:
        """pip_size as a Decimal (for spread arithmetic)."""
        return _dec(self.pip_size)

    def as_dict(self) -> dict[str, object]:
        return {
            "instrument": self.instrument,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "pip_size": self.pip_size,
            "price_precision": self.price_precision,
        }


# The seven instruments supplied in the Phase-1 dataset, with their explicit
# currency orientation and pip conventions.
_INSTRUMENT_SPECS: dict[str, InstrumentSpec] = {
    "EURUSD": InstrumentSpec("EURUSD", "EUR", "USD", 0.0001, 5),
    "GBPUSD": InstrumentSpec("GBPUSD", "GBP", "USD", 0.0001, 5),
    "USDJPY": InstrumentSpec("USDJPY", "USD", "JPY", 0.01, 3),
    "USDCHF": InstrumentSpec("USDCHF", "USD", "CHF", 0.0001, 5),
    "AUDUSD": InstrumentSpec("AUDUSD", "AUD", "USD", 0.0001, 5),
    "USDCAD": InstrumentSpec("USDCAD", "USD", "CAD", 0.0001, 5),
    "NZDUSD": InstrumentSpec("NZDUSD", "NZD", "USD", 0.0001, 5),
}

# All currencies referenced by the supported instruments.
SUPPORTED_CURRENCIES: tuple[str, ...] = ("EUR", "GBP", "USD", "JPY", "CHF", "CAD", "AUD", "NZD")

# Default account currency for the simulator.
DEFAULT_ACCOUNT_CURRENCY: str = "USD"

# Currencies the conversion service can *directly* convert to/from USD using a
# single supported USD-major pair (either BASE/USD or USD/BASE-or-quote).
# Anything outside this set requires a cross rate that the dataset does not
# provide, and must raise rather than silently use 1.0.
_CURRENCIES_REACHABLE_FROM_USD: frozenset[str] = frozenset(
    {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}
)


def instrument_spec(instrument: str) -> InstrumentSpec:
    """Return the explicit :class:`InstrumentSpec` for ``instrument``."""
    key = instrument.upper()
    if key not in _INSTRUMENT_SPECS:
        known = ", ".join(sorted(_INSTRUMENT_SPECS))
        raise KeyError(f"unknown instrument {instrument!r}; known pairs: {known}")
    return _INSTRUMENT_SPECS[key]


def base_currency(instrument: str) -> str:
    return instrument_spec(instrument).base_currency


def quote_currency(instrument: str) -> str:
    return instrument_spec(instrument).quote_currency


def known_instruments() -> tuple[str, ...]:
    """All instruments with explicit metadata, in dataset order."""
    return tuple(_INSTRUMENT_SPECS)


def currency_reachable_from_usd(currency: str) -> bool:
    """True when ``currency`` can be converted to/from USD via a single pair."""
    return currency in _CURRENCIES_REACHABLE_FROM_USD
