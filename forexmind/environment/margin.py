"""Configuration-driven margin / leverage model.

Phase 1 keeps margin simple but explicit and fully tested:

* ``margin_used = gross_exposure * margin_requirement``
  (``margin_requirement`` defaults to ``1 / leverage``)
* ``free_margin = equity - margin_used``
* excessive leverage is detected when ``leverage_used > max_leverage``
  (when ``max_leverage`` is configured)
* deterministic liquidation: if ``equity <= margin_used * liquidation_ratio``
  while a position is open, the position is force-closed at the current price.

This is *not* claimed to match any particular broker; it is a configurable
model (see ``MarginConfig``).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from forexmind.config import MarginConfig, _dec


@dataclass(frozen=True, slots=True)
class MarginSnapshot:
    """Margin state at a point in time."""

    gross_exposure: Decimal
    margin_used: Decimal
    free_margin: Decimal
    leverage_used: Decimal  # gross_exposure / equity
    margin_call: bool  # free_margin < 0 (excessive leverage)
    liquidation: bool  # position should be force-closed

    def as_dict(self) -> dict[str, object]:
        return {
            "gross_exposure": self.gross_exposure,
            "margin_used": self.margin_used,
            "free_margin": self.free_margin,
            "leverage_used": self.leverage_used,
            "margin_call": self.margin_call,
            "liquidation": self.liquidation,
        }


class MarginModel:
    def __init__(self, config: MarginConfig) -> None:
        self.config = config

    @property
    def margin_requirement(self) -> Decimal:
        return self.config.effective_margin_requirement

    @property
    def liquidation_ratio(self) -> Decimal:
        return self.config.maintenance_margin_ratio

    def gross_exposure(self, units: float | int | Decimal, price: float | int | Decimal) -> Decimal:
        return abs(_dec(units)) * _dec(price)

    def margin_used(self, units: float | int | Decimal, price: float | int | Decimal) -> Decimal:
        return self.gross_exposure(units, price) * self.margin_requirement

    def snapshot(
        self,
        *,
        equity: float | int | Decimal,
        units: float | int | Decimal,
        price: float | int | Decimal,
    ) -> MarginSnapshot:
        equity_d = _dec(equity)
        units_d = _dec(units)
        gross = self.gross_exposure(units_d, price)
        margin_used = gross * self.margin_requirement
        free_margin = equity_d - margin_used
        leverage_used = gross / equity_d if equity_d > 0 else Decimal("Infinity")
        margin_call = free_margin < 0
        liquidation = bool(units_d) and (equity_d <= margin_used * self.liquidation_ratio)
        return MarginSnapshot(
            gross_exposure=gross,
            margin_used=margin_used,
            free_margin=free_margin,
            leverage_used=leverage_used,
            margin_call=margin_call,
            liquidation=liquidation,
        )

    def can_open(
        self,
        *,
        equity: float | int | Decimal,
        units: float | int | Decimal,
        price: float | int | Decimal,
    ) -> bool:
        """True when the resulting margin requirement is affordable."""
        snap = self.snapshot(equity=equity, units=units, price=price)
        if snap.margin_call:
            return False
        return not (
            self.config.max_leverage is not None and snap.leverage_used > self.config.max_leverage
        )
