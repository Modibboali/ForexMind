"""Portfolio and accounting engine (Phase 3.1 multi-currency aware).

Accounting convention (matches how brokers report a margin account), with all
account-level monetary values expressed in the **account currency**
(``account_currency``, default ``"USD"``):

* ``balance`` is the account balance: ``initial + net realized PnL`` (account
  currency).  It changes only when PnL is realised (closing/reducing positions)
  or when commissions are charged.  Opening a position does *not* change the
  balance.
* ``unrealized_pnl`` (floating PnL) is the mark-to-market profit of the open
  position, converted into the account currency.
* ``equity = balance + unrealized_pnl`` (all account currency).
* ``realized_pnl = balance - initial``.
* Invariant that always holds: ``equity - initial == realized_pnl + unrealized_pnl``.

Raw trade P&L is computed in the instrument's **quote currency** and converted
to the account currency before it enters any account-level field.  The
conversion uses the contemporaneous price of the instrument at the same
mark/execution timestamp (never future prices); see
:mod:`forexmind.environment.fx_conversion`.

All values use :class:`decimal.Decimal` with a fixed context precision so
accounting is exact and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from forexmind.config import _dec
from forexmind.environment.fx_conversion import AccountCurrencyError, CurrencyConverter
from forexmind.environment.instruments import InstrumentSpec, instrument_spec


@dataclass(frozen=True, slots=True)
class Position:
    """A single net position in one instrument."""

    instrument: str
    units: Decimal  # signed base units (+ long, - short, 0 flat)
    entry_price: Decimal  # average entry (0 when flat)

    @property
    def direction(self) -> str:
        if self.units > 0:
            return "long"
        if self.units < 0:
            return "short"
        return "flat"

    @property
    def is_flat(self) -> bool:
        return self.units == 0


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Immutable view of the portfolio state at a point in time.

    All monetary fields (``balance``, ``equity``, ``unrealized_pnl``,
    ``realized_pnl``, ``gross_exposure``) are in the **account currency**.
    """

    balance: Decimal
    equity: Decimal
    position: Position
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    gross_exposure: Decimal
    drawdown: Decimal
    drawdown_pct: Decimal
    peak_equity: Decimal
    account_currency: str = "USD"
    base_currency: str = ""
    quote_currency: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "balance": self.balance,
            "equity": self.equity,
            "position_units": self.position.units,
            "entry_price": self.position.entry_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "gross_exposure": self.gross_exposure,
            "drawdown": self.drawdown,
            "drawdown_pct": self.drawdown_pct,
            "peak_equity": self.peak_equity,
            "account_currency": self.account_currency,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
        }


@dataclass(frozen=True, slots=True)
class TradeResult:
    """Outcome of an executed adjustment to a target position.

    ``raw_pnl`` / ``raw_pnl_currency`` expose the quote-currency P&L *before*
    conversion; ``realized_pnl`` / ``converted_pnl`` are in the account
    currency.  ``conversion_rate`` is the quote->account multiplier applied at
    this execution's price.
    """

    units_delta: Decimal
    executed_units: Decimal  # |units_delta|
    direction: str  # "buy" | "sell" | "none"
    execution_price: Decimal
    commission: Decimal  # account currency (already converted)
    realized_pnl: Decimal  # net realised PnL of this trade, account currency
    balance_delta: Decimal  # account currency
    units_after: Decimal
    entry_price_after: Decimal
    # Currency diagnostics (Phase 3.1)
    commission_raw: Decimal = Decimal(0)  # quote-currency commission before conversion
    raw_pnl: Decimal = Decimal(0)  # quote-currency price PnL before conversion
    raw_pnl_currency: str = "USD"
    converted_pnl: Decimal = Decimal(0)  # account-currency price PnL before costs
    conversion_rate: Decimal = Decimal(1)  # quote -> account multiplier
    account_currency: str = "USD"
    quote_currency: str = "USD"


class Portfolio:
    """Mutable accounting state for a single instrument (one net position).

    ``initial_balance`` is in ``account_currency``.  ``instrument_spec`` and
    ``converter`` default to the standard registry / USD account so existing
    EURUSD (USD-quote) callers keep identical behaviour: for a USD account the
    quote->account factor is ``1``.
    """

    def __init__(
        self,
        instrument: str,
        initial_balance: float | str | Decimal,
        *,
        account_currency: str = "USD",
        instrument_spec_: InstrumentSpec | None = None,
        converter: CurrencyConverter | None = None,
    ) -> None:
        self._instrument = instrument.upper()
        spec = instrument_spec_ or instrument_spec(self._instrument)
        self._spec = spec
        self._base_currency = spec.base_currency
        self._quote_currency = spec.quote_currency
        self._account_currency = account_currency.upper()
        self._converter = converter or CurrencyConverter(self._account_currency)

        self._initial_balance = _dec(initial_balance)
        self._balance = self._initial_balance
        self._units = Decimal(0)
        self._entry = Decimal(0)
        self._unrealized_pnl = Decimal(0)
        self._raw_unrealized_pnl = Decimal(0)  # quote-currency floating PnL
        self._conversion_rate = Decimal(1)  # last quote->account factor used
        self._current_mid: Decimal | None = None
        self._peak_equity = self._initial_balance

    # -- read-only properties -------------------------------------------------

    @property
    def instrument(self) -> str:
        return self._instrument

    @property
    def initial_balance(self) -> Decimal:
        return self._initial_balance

    @property
    def account_currency(self) -> str:
        return self._account_currency

    @property
    def instrument_spec(self) -> InstrumentSpec:
        return self._spec

    @property
    def base_currency(self) -> str:
        return self._base_currency

    @property
    def quote_currency(self) -> str:
        return self._quote_currency

    @property
    def converter(self) -> CurrencyConverter:
        return self._converter

    @property
    def balance(self) -> Decimal:
        return self._balance

    @property
    def realized_pnl(self) -> Decimal:
        return self._balance - self._initial_balance

    @property
    def unrealized_pnl(self) -> Decimal:
        return self._unrealized_pnl

    @property
    def raw_unrealized_pnl(self) -> Decimal:
        """Floating PnL in the quote currency (before conversion)."""
        return self._raw_unrealized_pnl

    @property
    def conversion_rate(self) -> Decimal:
        return self._conversion_rate

    @property
    def equity(self) -> Decimal:
        return self._balance + self._unrealized_pnl

    @property
    def position(self) -> Position:
        return Position(self._instrument, self._units, self._entry)

    @property
    def current_mid(self) -> Decimal | None:
        return self._current_mid

    @property
    def peak_equity(self) -> Decimal:
        return self._peak_equity

    @property
    def drawdown(self) -> Decimal:
        return self._peak_equity - self.equity

    @property
    def drawdown_pct(self) -> Decimal:
        if self._peak_equity == 0:
            return Decimal(0)
        return (self._peak_equity - self.equity) / self._peak_equity

    # -- currency helpers -----------------------------------------------------

    def _quote_to_account(self, amount: Decimal, price: Decimal) -> Decimal:
        """Convert a quote-currency ``amount`` to account currency at ``price``."""
        factor = self._converter.quote_to_account_factor(
            self._instrument, price, self._account_currency
        )
        self._conversion_rate = factor
        return amount * factor

    def _gross_exposure_account(self, price: Decimal) -> Decimal:
        """Account-currency gross exposure of the current open position."""
        if not self._units:
            return Decimal(0)
        if self._quote_currency == self._account_currency:
            return abs(self._units) * price
        if self._base_currency == self._account_currency:
            # Pair is ACCOUNT/quote (e.g. USDJPY for a USD account): the base
            # notional is already in account currency.
            return abs(self._units)
        raise AccountCurrencyError(
            f"cannot express gross exposure of {self._instrument} in account "
            f"currency {self._account_currency}: no compatible single-hop pair"
        )

    # -- operations -----------------------------------------------------------

    def mark_to_market(self, mid_price: float | str | Decimal) -> None:
        """Mark the open position to ``mid_price`` (updates floating PnL).

        Floating PnL is computed in the quote currency then converted into the
        account currency using the conversion factor at ``mid_price`` (the
        contemporaneous mark price; never a future observation).
        """
        mid = _dec(mid_price)
        raw = self._units * (mid - self._entry) if self._units else Decimal(0)
        self._raw_unrealized_pnl = raw
        if self._units:
            self._unrealized_pnl = self._quote_to_account(raw, mid)
        else:
            self._unrealized_pnl = Decimal(0)
            self._conversion_rate = Decimal(1)
        self._current_mid = mid
        if self.equity > self._peak_equity:
            self._peak_equity = self.equity

    def adjust_to_target(
        self,
        target_units: float | str | Decimal,
        execution_price: float | str | Decimal,
        commission: float | str | Decimal = 0,
    ) -> TradeResult:
        """Adjust the position towards ``target_units`` at ``execution_price``.

        Handles flat -> long, flat -> short, long -> flat, short -> flat,
        long -> short and short -> long transitions with correct realised-PnL
        accounting for the closed exposure and average-cost entry for
        same-direction increases.

        Realised price PnL is computed in the quote currency, then converted to
        the account currency at ``execution_price``.  ``commission`` is treated
        as a quote-currency cost per base unit and likewise converted.
        """
        target = _dec(target_units)
        px = _dec(execution_price)
        comm = _dec(commission)
        old_units, old_entry = self._units, self._entry
        delta = target - old_units

        # Conversion factor at this execution price.
        factor = self._converter.quote_to_account_factor(
            self._instrument, px, self._account_currency
        )
        self._conversion_rate = factor

        if delta == 0 and comm == 0:
            return TradeResult(
                units_delta=Decimal(0),
                executed_units=Decimal(0),
                direction="none",
                execution_price=px,
                commission=Decimal(0),
                realized_pnl=Decimal(0),
                balance_delta=Decimal(0),
                units_after=old_units,
                entry_price_after=old_entry,
                commission_raw=Decimal(0),
                raw_pnl=Decimal(0),
                raw_pnl_currency=self._quote_currency,
                converted_pnl=Decimal(0),
                conversion_rate=factor,
                account_currency=self._account_currency,
                quote_currency=self._quote_currency,
            )

        realized_quote_pnl, new_units, new_entry = self._transition(delta, px)
        realized_acct = realized_quote_pnl * factor
        comm_acct = comm * factor

        self._balance += realized_acct - comm_acct
        self._units, self._entry = new_units, new_entry
        # Keep equity coherent immediately after the trade by marking at the
        # execution price; the environment will re-mark at the next close.
        self.mark_to_market(px)

        direction = "buy" if delta > 0 else "sell"
        return TradeResult(
            units_delta=delta,
            executed_units=abs(delta),
            direction=direction,
            execution_price=px,
            commission=comm_acct,
            realized_pnl=realized_acct - comm_acct,
            balance_delta=realized_acct - comm_acct,
            units_after=new_units,
            entry_price_after=new_entry,
            commission_raw=comm,
            raw_pnl=realized_quote_pnl,
            raw_pnl_currency=self._quote_currency,
            converted_pnl=realized_acct,
            conversion_rate=factor,
            account_currency=self._account_currency,
            quote_currency=self._quote_currency,
        )

    def close_all(
        self, execution_price: float | str | Decimal, commission: float | str | Decimal = 0
    ) -> TradeResult:
        """Close the entire position (no-op when already flat)."""
        return self.adjust_to_target(Decimal(0), execution_price, commission)

    def apply_cash_adjustment(self, amount: float | int | str | Decimal) -> None:
        """Apply an account-currency cash adjustment to the balance.

        Used for financing/side cash flows that are already expressed in the
        account currency.  Does not change the position; unrealized PnL is
        preserved (so equity moves by ``amount``).
        """
        amt = _dec(amount)
        self._balance += amt
        if self.equity > self._peak_equity:
            self._peak_equity = self.equity

    # -- internals ------------------------------------------------------------

    def _transition(self, delta: Decimal, px: Decimal) -> tuple[Decimal, Decimal, Decimal]:
        """Return ``(realized_price_pnl_quote, new_units, new_entry)`` for a delta trade.

        ``realized_price_pnl_quote`` is in the *quote* currency (before any FX
        conversion), consistent with ``PnL_quote = N * (P_exit - P_entry)``.
        """
        old_units, old_entry = self._units, self._entry
        if old_units == 0:
            return Decimal(0), delta, px
        if (old_units > 0) == (delta > 0):
            # Same-direction increase: average-cost entry, nothing realised.
            new_units = old_units + delta
            new_entry = (old_units * old_entry + delta * px) / new_units
            return Decimal(0), new_units, new_entry
        # Reducing or flipping.
        close_units = min(abs(delta), abs(old_units))
        if old_units > 0:
            realized = close_units * (px - old_entry)
        else:
            realized = close_units * (old_entry - px)
        new_units = old_units + delta
        if new_units == 0:
            new_entry = Decimal(0)
        elif (new_units > 0) != (old_units > 0):
            new_entry = px  # flipped direction
        else:
            new_entry = old_entry
        return realized, new_units, new_entry

    def snapshot(self, mid_price: float | str | Decimal | None = None) -> PortfolioSnapshot:
        """Capture an immutable snapshot (marks at ``mid_price`` if given)."""
        if mid_price is not None:
            self.mark_to_market(mid_price)
        mark = self._current_mid or Decimal(0)
        gross = self._gross_exposure_account(mark) if self._units else Decimal(0)
        return PortfolioSnapshot(
            balance=self._balance,
            equity=self.equity,
            position=self.position,
            unrealized_pnl=self._unrealized_pnl,
            realized_pnl=self.realized_pnl,
            gross_exposure=gross,
            drawdown=self.drawdown,
            drawdown_pct=self.drawdown_pct,
            peak_equity=self._peak_equity,
            account_currency=self._account_currency,
            base_currency=self._base_currency,
            quote_currency=self._quote_currency,
        )

    # -- invariant check (used by tests) --------------------------------------

    def check_invariant(self, tol: Decimal | None = None) -> bool:
        """True when ``equity - initial == realized + unrealized``.

        Average-cost entries are computed by division under the fixed Decimal
        context precision, so the identity holds to ~1e-48 relative rather than
        exactly.  The default tolerance is extremely tight (1e-38 relative to
        equity) and far below any value that would matter for accounting.
        """
        lhs = self.equity - self._initial_balance
        rhs = self.realized_pnl + self._unrealized_pnl
        if tol is None:
            tol = max(abs(self.equity), Decimal("1")) * Decimal("1e-38")
        return abs(lhs - rhs) <= tol
