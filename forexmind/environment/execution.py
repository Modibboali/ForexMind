"""Execution engine: turns a desired position delta into a concrete trade.

The engine is deliberately small: it computes the execution price and costs
for a signed unit delta at a given mid price.  Portfolio accounting is done by
:class:`forexmind.environment.portfolio.Portfolio`.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from forexmind.config import _dec
from forexmind.environment.costs import ExecutionCostModel


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """One executed trade."""

    timestamp: pd.Timestamp
    mid_price: Decimal
    execution_price: Decimal  # buy or sell price actually paid
    units_delta: Decimal  # signed base units (positive = buy, negative = sell)
    direction: str  # "buy" | "sell" | "none"
    commission: Decimal
    gross_flow: Decimal  # signed cash flow before commission
    net_flow: Decimal  # signed cash flow after commission
    instrument: str | None = None  # instrument (for per-pair spread resolution)


class ExecutionEngine:
    """Deterministic executor using an :class:`ExecutionCostModel`."""

    def __init__(self, cost_model: ExecutionCostModel) -> None:
        self.cost_model = cost_model

    def execute(
        self,
        timestamp: pd.Timestamp,
        mid_price: float | Decimal,
        units_delta: float | int | Decimal,
        instrument: str | None = None,
    ) -> ExecutionReport:
        """Execute ``units_delta`` at the cost-adjusted price.

        ``units_delta > 0`` buys at the ask (mid + spread/2 + slippage);
        ``units_delta < 0`` sells at the bid (mid - spread/2 - slippage).
        ``instrument`` selects the per-pair spread override when configured.
        """
        delta = _dec(units_delta)
        prices = self.cost_model.execution_prices(mid_price, instrument)
        if delta == 0:
            return ExecutionReport(
                timestamp=timestamp,
                mid_price=prices.mid,
                execution_price=prices.mid,
                units_delta=Decimal(0),
                direction="none",
                commission=Decimal(0),
                gross_flow=Decimal(0),
                net_flow=Decimal(0),
                instrument=instrument,
            )
        direction = "buy" if delta > 0 else "sell"
        price = prices.buy if delta > 0 else prices.sell
        commission = self.cost_model.commission(delta)
        gross_flow = delta * price  # negative for buys, positive for sells
        net_flow = gross_flow - commission
        return ExecutionReport(
            timestamp=timestamp,
            mid_price=prices.mid,
            execution_price=price,
            units_delta=delta,
            direction=direction,
            commission=commission,
            gross_flow=gross_flow,
            net_flow=net_flow,
            instrument=instrument,
        )
