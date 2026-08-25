"""Portfolio and accounting engine.

Accounting convention (matches how brokers report a margin account):

* ``balance`` is the account balance: ``initial + net realized PnL``.  It
  changes only when PnL is realised (closing/reducing positions) or when
  commissions are charged.  Opening a position does *not* change the balance.
* ``unrealized_pnl`` (floating PnL) is the mark-to-market profit of the open
  position: ``units * (mid - entry)``.
* ``equity = balance + unrealized_pnl``.
* ``realized_pnl = balance - initial``.
* Invariant that always holds: ``equity - initial == realized_pnl + unrealized_pnl``.

All values use :class:`decimal.Decimal` with a fixed context precision so
accounting is exact and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from forexmind.config import _dec


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
    """Immutable view of the portfolio state at a point in time."""

    balance: Decimal
    equity: Decimal
    position: Position
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    gross_exposure: Decimal
    drawdown: Decimal
    drawdown_pct: Decimal
    peak_equity: Decimal

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
        }


@dataclass(frozen=True, slots=True)
class TradeResult:
    """Outcome of an executed adjustment to a target position."""

    units_delta: Decimal
    executed_units: Decimal  # |units_delta|
    direction: str  # "buy" | "sell" | "none"
    execution_price: Decimal
    commission: Decimal
    realized_pnl: Decimal  # net realised PnL of this trade (price pnl - commission)
    balance_delta: Decimal
    units_after: Decimal
    entry_price_after: Decimal


class Portfolio:
    """Mutable accounting state for a single instrument (Phase 1: one net position)."""

    def __init__(self, instrument: str, initial_balance: float | str | Decimal) -> None:
        self._instrument = instrument
        self._initial_balance = _dec(initial_balance)
        self._balance = self._initial_balance
        self._units = Decimal(0)
        self._entry = Decimal(0)
        self._unrealized_pnl = Decimal(0)
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
    def balance(self) -> Decimal:
        return self._balance

    @property
    def realized_pnl(self) -> Decimal:
        return self._balance - self._initial_balance

    @property
    def unrealized_pnl(self) -> Decimal:
        return self._unrealized_pnl

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

    # -- operations -----------------------------------------------------------

    def mark_to_market(self, mid_price: float | str | Decimal) -> None:
        """Mark the open position to ``mid_price`` (updates floating PnL)."""
        mid = _dec(mid_price)
        self._unrealized_pnl = self._units * (mid - self._entry) if self._units else Decimal(0)
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
        """
        target = _dec(target_units)
        px = _dec(execution_price)
        comm = _dec(commission)
        old_units, old_entry = self._units, self._entry
        delta = target - old_units

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
            )

        realized_price_pnl, new_units, new_entry = self._transition(delta, px)

        self._balance += realized_price_pnl - comm
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
            commission=comm,
            realized_pnl=realized_price_pnl - comm,
            balance_delta=realized_price_pnl - comm,
            units_after=new_units,
            entry_price_after=new_entry,
        )

    def close_all(
        self, execution_price: float | str | Decimal, commission: float | str | Decimal = 0
    ) -> TradeResult:
        """Close the entire position (no-op when already flat)."""
        return self.adjust_to_target(Decimal(0), execution_price, commission)

    # -- internals ------------------------------------------------------------

    def _transition(self, delta: Decimal, px: Decimal) -> tuple[Decimal, Decimal, Decimal]:
        """Return ``(realized_price_pnl, new_units, new_entry)`` for a delta trade."""
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
        gross = (
            abs(self._units) * self._current_mid
            if (self._units and self._current_mid)
            else Decimal(0)
        )
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
