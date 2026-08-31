"""Execution-cost model.

Phase 1 uses a simple deterministic model built from configured assumptions::

    mid_price = selected market price (next M1 open by default)

    buy_price  = mid + spread/2
    sell_price = mid - spread/2

    buy_price  += slippage      (optional)
    sell_price -= slippage      (optional)

    commission = commission_per_unit * |units|   (per execution side)

These are *model assumptions*: the raw dataset contains only OHLC, so no
historical bid/ask is claimed.  The implementation is isolated so real
bid/ask/tick data can be introduced later without redesigning the rest of
the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from forexmind.config import ExecutionConfig, _dec


@dataclass(frozen=True, slots=True)
class ExecutionPrices:
    """Buy/sell prices derived from a mid price plus configured costs."""

    mid: Decimal
    buy: Decimal
    sell: Decimal


class ExecutionCostModel:
    """Deterministic spread / slippage / commission model.

    The model is instrument-aware: when a per-instrument spread override is
    configured (``ExecutionConfig.instrument_spreads``), it is applied for that
    instrument; otherwise the global ``spread_value`` is used.  This makes it
    possible to use the correct pip size per pair (e.g. 0.01 for JPY pairs
    instead of a single 0.0002 value).
    """

    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config

    def execution_prices(
        self, mid_price: float | Decimal, instrument: str | None = None
    ) -> ExecutionPrices:
        """Compute buy/sell prices around a mid price."""
        mid = _dec(mid_price)
        spread = self.config.spread_for(instrument) if instrument else self.config.spread_decimal
        half_spread = spread / 2
        buy = mid + half_spread
        sell = mid - half_spread
        if self.config.slippage_mode == "fixed":
            buy = buy + self.config.slippage_decimal
            sell = sell - self.config.slippage_decimal
        return ExecutionPrices(mid=mid, buy=buy, sell=sell)

    def commission(self, units: float | int | Decimal) -> Decimal:
        """Total commission for executing ``units`` (per side), quote currency.

        The caller converts this to the account currency before applying it.
        """
        return abs(_dec(units)) * self.config.commission_decimal

    def total_execution_charge(
        self,
        units: float | int | Decimal,
        mid_price: float | Decimal,
        instrument: str | None = None,
    ) -> Decimal:
        """Quote-currency cost of executing ``units`` (incl. spread/slippage)."""
        units_d = _dec(units)
        prices = self.execution_prices(mid_price, instrument)
        unit_cost = prices.buy if units_d > 0 else prices.sell
        return abs(units_d) * (abs(unit_cost - prices.mid)) + self.commission(units_d)
