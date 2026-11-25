"""Currency conversion service (Phase 3.1 multi-currency accounting).

A dedicated, self-contained converter keeps FX logic out of
:class:`~forexmind.environment.portfolio.Portfolio`.  It converts a monetary
amount from one currency to another given contemporaneous exchange rates.

Only conversions derivable from *documented* instrument prices are supported.
The Phase-1 dataset contains seven USD-major pairs (EURUSD, GBPUSD, USDJPY,
USDCHF, AUDUSD, USDCAD, NZDUSD); we never fabricate arbitrary cross rates.  A
conversion that cannot be derived reliably raises :class:`AccountCurrencyError`
instead of silently using ``1.0``.

Price orientation convention (consistent with the portfolio):
``BASE/QUOTE`` means ``1 BASE = price QUOTE``.
"""

from __future__ import annotations

from decimal import Decimal

from forexmind.config import _dec
from forexmind.environment.instruments import (
    DEFAULT_ACCOUNT_CURRENCY,
    base_currency,
    currency_reachable_from_usd,
    quote_currency,
)


class AccountCurrencyError(ValueError):
    """Raised when a required currency conversion is unavailable."""


def _rate_for_pair(pair: str, rates: dict[str, Decimal]) -> Decimal:
    """Look up a pair price in the ``rates`` map (case-insensitive)."""
    for key, value in rates.items():
        if key.upper() == pair.upper():
            return value
    raise AccountCurrencyError(
        f"missing exchange rate for {pair.upper()} in conversion rates {sorted(rates)}"
    )


class CurrencyConverter:
    """Converts monetary amounts between currencies using documented rates.

    ``rates`` maps a pair name (e.g. ``"EURUSD"``) to a :class:`Decimal` price
    available *at the mark/execution timestamp*.  It is passed per-call so the
    caller controls the exact pricing timestamp (never future prices).
    """

    def __init__(self, account_currency: str = DEFAULT_ACCOUNT_CURRENCY) -> None:
        acc = account_currency.upper()
        if not currency_reachable_from_usd(acc) and acc != "USD":
            # Only currencies reachable from the provided instruments are valid
            # account currencies for the default instrumentation.
            raise AccountCurrencyError(
                f"unsupported account currency {account_currency!r}; the Phase-1 dataset "
                "only supports conversions to/from USD majors"
            )
        self.account_currency = acc

    # -- generic conversion ---------------------------------------------------

    def convert(
        self,
        amount: float | int | str | Decimal,
        from_currency: str,
        to_currency: str,
        rates: dict[str, float | int | str | Decimal],
    ) -> Decimal:
        """Convert ``amount`` from ``from_currency`` to ``to_currency``.

        ``rates`` maps pairwise prices available at the desired timestamp.

        Supported single-hop derivations (the only ones economically justified
        by the dataset):

        * ``from == to`` -> unchanged.
        * direct pair ``FROM/TO``: ``amount * price``.
        * inverse pair ``TO/FROM``: ``amount / price``.

        Anything else (a cross requiring two hops, or an unavailable leg)
        raises :class:`AccountCurrencyError`.
        """
        amt = _dec(amount)
        src = from_currency.upper()
        dst = to_currency.upper()
        if src == dst:
            return amt
        rate_map = {k.upper(): _dec(v) for k, v in rates.items()}

        direct = f"{src}{dst}"
        if direct in rate_map:
            return amt * rate_map[direct]

        inverse = f"{dst}{src}"
        if inverse in rate_map:
            return amt / rate_map[inverse]

        raise AccountCurrencyError(
            f"cannot convert {from_currency} -> {to_currency}: no compatible pair in rates "
            f"{sorted(rate_map)}; cross rates are not fabricated"
        )

    # -- portfolio helper -----------------------------------------------------

    def quote_to_account_factor(
        self,
        instrument: str,
        price: float | int | str | Decimal,
        account_currency: str | None = None,
    ) -> Decimal:
        """Return the multiplier that converts *quote-currency* P&L to account currency.

        For the default USD account this is:

        * quote == account (e.g. EURUSD, GBPUSD, AUDUSD, NZDUSD)  -> ``1``
        * pair is ``USD/quote`` (e.g. USDJPY, USDCHF, USDCAD)     -> ``1 / price``

        so that ``account_pnl = quote_pnl * factor``.  ``price`` must be the
        contemporaneous price of the instrument ``instrument`` at the same
        mark/execution timestamp (never a future price).

        Raises :class:`AccountCurrencyError` when the conversion cannot be
        derived from the available instruments.
        """
        acc = (account_currency or self.account_currency).upper()
        base = base_currency(instrument)
        quote = quote_currency(instrument)
        if quote == acc:
            return Decimal(1)
        # Pair is ACCOUNT/quote (USD/JPY, USD/CHF, USD/CAD for a USD account):
        # quote -> account = 1 / price.
        if base == acc:
            px = _dec(price)
            if px == 0:
                raise AccountCurrencyError(
                    f"cannot convert {quote} -> {acc}: {instrument} price is zero"
                )
            return Decimal(1) / px
        raise AccountCurrencyError(
            f"cannot convert quote currency {quote} of {instrument} to account "
            f"currency {acc}: no compatible single-hop pair available; cross rates "
            "are not fabricated"
        )

    def gross_exposure_to_account(
        self,
        instrument: str,
        notional_units: float | int | str | Decimal,
        price: float | int | str | Decimal,
        account_currency: str | None = None,
    ) -> Decimal:
        """Account-currency gross exposure for ``|units|`` of ``instrument`` at ``price``.

        ``grossExposure_account = |units| * price * quote_to_account_factor``

        For USD-quote pairs this is ``|units| * price``; for USD/XXX pairs the
        price is in the quote currency, which converts back to ~1 account unit
        per base unit of USD.
        """
        acc = (account_currency or self.account_currency).upper()
        base = base_currency(instrument)
        quote = quote_currency(instrument)
        if quote == acc:
            # PnL/exposure already in account currency.
            return abs(_dec(notional_units)) * _dec(price)
        if base == acc:
            # pair is ACCOUNT/quote; notional in base already account currency.
            return abs(_dec(notional_units))
        raise AccountCurrencyError(
            f"cannot express gross exposure of {instrument} in account currency {acc}: "
            "no compatible single-hop pair available"
        )
