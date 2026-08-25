"""Tests for portfolio accounting: balance, equity, realized/unrealized PnL,
drawdown, average-cost entry, and position transitions."""

from __future__ import annotations

from decimal import Decimal

from forexmind.environment.portfolio import Portfolio


def _portfolio() -> Portfolio:
    return Portfolio("EURUSD", "10000")


# -- transitions -------------------------------------------------------------


def test_flat_to_long() -> None:
    p = _portfolio()
    p.adjust_to_target(100, "1.10")
    assert p.position.units == 100
    assert p.position.entry_price == Decimal("1.10")
    assert p.position.direction == "long"


def test_flat_to_short() -> None:
    p = _portfolio()
    p.adjust_to_target(-100, "1.10")
    assert p.position.units == -100
    assert p.position.entry_price == Decimal("1.10")
    assert p.position.direction == "short"


def test_flat_to_flat_noop() -> None:
    p = _portfolio()
    res = p.adjust_to_target(0, "1.10")
    assert res.direction == "none"
    assert p.position.is_flat
    assert p.balance == Decimal("10000")


def test_long_to_flat() -> None:
    p = _portfolio()
    p.adjust_to_target(100, "1.10")
    p.mark_to_market("1.14")
    res = p.adjust_to_target(0, "1.14")
    assert p.position.is_flat
    assert res.realized_pnl == Decimal("4")
    assert p.realized_pnl == Decimal("4")


def test_short_to_flat() -> None:
    p = _portfolio()
    p.adjust_to_target(-100, "1.10")
    p.mark_to_market("1.06")
    res = p.adjust_to_target(0, "1.06")
    assert p.position.is_flat
    assert res.realized_pnl == Decimal("4")
    assert p.realized_pnl == Decimal("4")


def test_long_to_short() -> None:
    p = _portfolio()
    p.adjust_to_target(100, "1.10")
    p.mark_to_market("1.12")
    res = p.adjust_to_target(-50, "1.12")
    assert p.position.units == -50
    assert p.position.entry_price == Decimal("1.12")
    # Closing the 100-unit long at 1.12 realises 2.
    assert res.realized_pnl == Decimal("2")
    assert p.realized_pnl == Decimal("2")
    assert p.check_invariant()


def test_short_to_long() -> None:
    p = _portfolio()
    p.adjust_to_target(-100, "1.10")
    p.mark_to_market("1.06")
    res = p.adjust_to_target(50, "1.06")
    assert p.position.units == 50
    assert p.position.entry_price == Decimal("1.06")
    assert res.realized_pnl == Decimal("4")
    assert p.realized_pnl == Decimal("4")
    assert p.check_invariant()


# -- average cost & partial closes -------------------------------------------


def test_average_cost_on_increase() -> None:
    p = _portfolio()
    p.adjust_to_target(100, "1.10")
    p.adjust_to_target(200, "1.20")
    assert p.position.units == 200
    assert p.position.entry_price == Decimal("1.15")
    assert p.realized_pnl == Decimal("0")


def test_partial_close_keeps_entry() -> None:
    p = _portfolio()
    p.adjust_to_target(100, "1.10")
    res = p.adjust_to_target(50, "1.12")
    assert p.position.units == 50
    assert p.position.entry_price == Decimal("1.10")
    assert res.realized_pnl == Decimal("1")
    assert p.realized_pnl == Decimal("1")


def test_commission_affects_balance_and_realized() -> None:
    p = _portfolio()
    p.adjust_to_target(100, "1.10", commission="0.5")
    p.adjust_to_target(0, "1.12", commission="0.5")
    # price pnl 2, minus 1.0 total commission.
    assert p.realized_pnl == Decimal("1")
    assert p.balance == Decimal("10001")
    assert p.equity == Decimal("10001")


# -- drawdown ----------------------------------------------------------------


def test_drawdown_tracking() -> None:
    p = _portfolio()
    assert p.drawdown == 0
    p.adjust_to_target(100, "1.10")
    p.mark_to_market("1.12")  # peak equity 10002
    assert p.drawdown == 0
    p.mark_to_market("1.08")  # equity 9998
    assert p.drawdown == Decimal("4")
    assert p.drawdown_pct == Decimal("4") / Decimal("10002")
    p.mark_to_market("1.15")  # new peak
    assert p.drawdown == 0
    assert p.peak_equity == Decimal("10005")


# -- snapshot ----------------------------------------------------------------


def test_snapshot_fields() -> None:
    p = _portfolio()
    p.adjust_to_target(100, "1.10")
    snap = p.snapshot(mid_price="1.12")
    assert snap.balance == Decimal("10000")
    assert snap.equity == Decimal("10002")
    assert snap.unrealized_pnl == Decimal("2")
    assert snap.gross_exposure == Decimal("112")
    assert snap.position.units == Decimal("100")


def test_invariant_across_random_sequence() -> None:
    import random

    rng = random.Random(7)
    p = _portfolio()
    for _ in range(200):
        units = rng.choice([-150, -100, -50, 0, 50, 100, 150])
        price = Decimal(str(round(rng.uniform(1.0, 1.3), 5)))
        p.adjust_to_target(units, price)
        p.mark_to_market(price + Decimal("0.001"))
        assert p.check_invariant()
