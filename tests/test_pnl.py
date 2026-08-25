"""Tests for long/short PnL scenarios (profitable and losing)."""

from __future__ import annotations

from decimal import Decimal

from forexmind.environment.portfolio import Portfolio


def _portfolio() -> Portfolio:
    return Portfolio("EURUSD", "10000")


def test_profitable_long() -> None:
    p = _portfolio()
    p.adjust_to_target(100, "1.10")
    p.mark_to_market("1.12")
    assert p.position.direction == "long"
    assert p.unrealized_pnl == Decimal("2")
    assert p.equity == Decimal("10002")
    p.adjust_to_target(0, "1.12")
    assert p.position.is_flat
    assert p.realized_pnl == Decimal("2")
    assert p.equity == Decimal("10002")


def test_losing_long() -> None:
    p = _portfolio()
    p.adjust_to_target(100, "1.10")
    p.mark_to_market("1.08")
    assert p.unrealized_pnl == Decimal("-2")
    p.adjust_to_target(0, "1.08")
    assert p.realized_pnl == Decimal("-2")
    assert p.equity == Decimal("9998")


def test_profitable_short() -> None:
    p = _portfolio()
    p.adjust_to_target(-100, "1.10")
    assert p.position.direction == "short"
    p.mark_to_market("1.08")
    assert p.unrealized_pnl == Decimal("2")
    p.adjust_to_target(0, "1.08")
    assert p.realized_pnl == Decimal("2")
    assert p.equity == Decimal("10002")


def test_losing_short() -> None:
    p = _portfolio()
    p.adjust_to_target(-100, "1.10")
    p.mark_to_market("1.12")
    assert p.unrealized_pnl == Decimal("-2")
    p.adjust_to_target(0, "1.12")
    assert p.realized_pnl == Decimal("-2")
    assert p.equity == Decimal("9998")


def test_buying_does_not_change_equity() -> None:
    p = _portfolio()
    p.adjust_to_target(100, "1.10")
    p.mark_to_market("1.10")
    assert p.equity == Decimal("10000")
    assert p.realized_pnl == Decimal("0")
    assert p.unrealized_pnl == Decimal("0")


def test_invariant_holds() -> None:
    p = _portfolio()
    for price, units in [("1.10", 100), ("1.12", -50), ("1.09", 0), ("1.11", 200)]:
        p.adjust_to_target(units, price)
        p.mark_to_market(price)
        assert p.check_invariant()
