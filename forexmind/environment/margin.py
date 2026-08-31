"""Configuration-driven margin / leverage model (Phase 3.1).

Phase 3.1 keeps margin simple but explicit and fully tested.  All monetary
inputs are in the **account currency** — margin must never be compared against
a mismatched currency (e.g. JPY margin vs USD equity):

* ``margin_used = gross_exposure_account * margin_requirement``
  (``margin_requirement`` defaults to ``1 / leverage``)
* ``free_margin = equity - margin_used``
* excessive leverage is detected when ``leverage_used > max_leverage``
  (when ``max_leverage`` is configured)
* deterministic liquidation: if ``equity <= margin_used * liquidation_ratio``
  while a position is open, the position is force-closed at the current price.

``gross_exposure`` is supplied by the caller in account currency (see the
portfolio), so USD/XXX pairs are converted correctly (e.g. ``|N|`` USD notional
for USDJPY) rather than mixing quote-currency notional with USD equity.

This is *not* claimed to match any particular broker; it is a configurable
model (see ``MarginConfig``).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from forexmind.config import MarginConfig, _dec


@dataclass(frozen=True, slots=True)
class MarginSnapshot:
    """Margin state at a point in time (account currency)."""

    gross_exposure: Decimal  # account currency
    margin_used: Decimal  # account currency
    free_margin: Decimal  # account currency
    leverage_used: Decimal  # gross_exposure / equity (dimensionless)
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

    def margin_used(self, gross_exposure_account: float | int | Decimal) -> Decimal:
        """Account-currency margin required for an account-currency gross exposure."""
        return abs(_dec(gross_exposure_account)) * self.margin_requirement

    def gross_exposure(self, units: float | int | Decimal, price: float | int | Decimal) -> Decimal:
        """Legacy quote-currency gross exposure ``|units| * price``.

        Only correct when the quote currency equals the account currency.
        Prefer account-currency gross exposure (from :class:`Portfolio`).
        """
        return abs(_dec(units)) * _dec(price)

    def snapshot(
        self,
        *,
        equity: float | int | Decimal,
        gross_exposure: float | int | Decimal | None = None,
        units: float | int | Decimal | None = None,
        price: float | int | Decimal | None = None,
    ) -> MarginSnapshot:
        """Compute a margin snapshot.

        ``gross_exposure`` must be in **account currency** (see :class:`Portfolio`).
        For backward compatibility, ``(units, price)`` may be supplied instead;
        it is treated as ``gross_exposure = |units| * price`` which is only
        correct when the quote currency equals the account currency (e.g.
        EURUSD in a USD account).
        """
        if gross_exposure is None:
            if units is None or price is None:
                raise ValueError(
                    "MarginModel.snapshot requires account-currency gross_exposure "
                    "(or legacy units and price)"
                )
            gross_exposure = abs(_dec(units)) * _dec(price)
        equity_d = _dec(equity)
        gross = abs(_dec(gross_exposure))
        margin_used = gross * self.margin_requirement
        free_margin = equity_d - margin_used
        leverage_used = gross / equity_d if equity_d > 0 else Decimal("Infinity")
        margin_call = free_margin < 0
        # Liquidation only applies while a position is open (gross_exposure > 0).
        liquidation = bool(gross) and (equity_d <= margin_used * self.liquidation_ratio)
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
        gross_exposure: float | int | Decimal | None = None,
        units: float | int | Decimal | None = None,
        price: float | int | Decimal | None = None,
    ) -> bool:
        """True when the resulting margin requirement is affordable.

        ``gross_exposure`` is in account currency; ``(units, price)`` may be
        supplied as an alias for ``abs(units) * price`` (legacy API).
        """
        if gross_exposure is None:
            if units is None or price is None:
                raise ValueError(
                    "MarginModel.can_open requires account-currency gross_exposure "
                    "(or legacy units and price)"
                )
            gross_exposure = abs(_dec(units)) * _dec(price)
        snap = self.snapshot(equity=equity, gross_exposure=gross_exposure)
        if snap.margin_call:
            return False
        return not (
            self.config.max_leverage is not None and snap.leverage_used > self.config.max_leverage
        )
