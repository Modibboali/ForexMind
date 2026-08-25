"""Tests for the margin / leverage model."""

from __future__ import annotations

from decimal import Decimal

from forexmind.config import MarginConfig
from forexmind.environment.margin import MarginModel


def _model(leverage: int = 100, **kw) -> MarginModel:
    return MarginModel(
        MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal(leverage), **kw)
    )


def test_margin_used() -> None:
    m = _model(leverage=100)
    snap = m.snapshot(equity=10000, units=100000, price=1.10)
    # gross exposure 110,000; margin requirement 1/100 -> 1100
    assert snap.gross_exposure == Decimal("110000")
    assert snap.margin_used == Decimal("1100")
    assert snap.free_margin == Decimal("10000") - Decimal("1100")


def test_margin_requirement_custom() -> None:
    m = _model(leverage=100, margin_requirement=Decimal("0.02"))
    snap = m.snapshot(equity=10000, units=100000, price=1.10)
    assert snap.margin_used == Decimal("2200")


def test_leverage_used() -> None:
    m = _model(leverage=100)
    snap = m.snapshot(equity=10000, units=100000, price=1.10)
    assert snap.leverage_used == Decimal("11")


def test_margin_call_on_negative_free_margin() -> None:
    m = _model(leverage=10)
    # exposure 110,000 needs 11,000 margin but equity is only 10,000.
    snap = m.snapshot(equity=10000, units=100000, price=1.10)
    assert snap.margin_used == Decimal("11000")
    assert snap.free_margin < 0
    assert snap.margin_call


def test_max_leverage_cap() -> None:
    m = _model(leverage=100, max_leverage=Decimal("5"))
    assert m.can_open(equity=10000, units=50000, price=1.00)  # 5x exactly
    assert not m.can_open(equity=10000, units=60000, price=1.00)  # 6x blocked


def test_liquidation_detection() -> None:
    m = _model(leverage=100)
    # margin_used = 1100; liquidation when equity <= 1100 * 0.5 = 550.
    assert not m.snapshot(equity=600, units=100000, price=1.10).liquidation
    assert m.snapshot(equity=550, units=100000, price=1.10).liquidation
    assert m.snapshot(equity=549, units=100000, price=1.10).liquidation
    # Flat position is never "liquidated".
    assert not m.snapshot(equity=1, units=0, price=1.10).liquidation


def test_liquidation_ratio_custom() -> None:
    m = _model(leverage=100, maintenance_margin_ratio=Decimal("0.25"))
    # liquidation when equity <= 1100 * 0.25 = 275
    assert not m.snapshot(equity=300, units=100000, price=1.10).liquidation
    assert m.snapshot(equity=275, units=100000, price=1.10).liquidation


def test_gross_exposure_negative_units() -> None:
    m = _model()
    assert m.gross_exposure(-1000, 1.10) == Decimal("1100")
