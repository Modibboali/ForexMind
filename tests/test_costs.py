"""Tests for execution-cost model (spread / slippage / commission)."""

from __future__ import annotations

from decimal import Decimal

import pytest
from forexmind.config import ExecutionConfig
from forexmind.environment.costs import ExecutionCostModel


def test_zero_costs() -> None:
    model = ExecutionCostModel(ExecutionConfig())
    p = model.execution_prices(1.1000)
    assert p.mid == Decimal("1.1000")
    assert p.buy == Decimal("1.1000")
    assert p.sell == Decimal("1.1000")
    assert model.commission(100) == 0


def test_fixed_spread_symmetric() -> None:
    model = ExecutionCostModel(ExecutionConfig(spread_mode="fixed", spread_value=0.0002))
    p = model.execution_prices(1.1000)
    assert p.buy == Decimal("1.1001")
    assert p.sell == Decimal("1.0999")
    assert p.buy - p.mid == p.mid - p.sell == Decimal("0.0001")


def test_slippage() -> None:
    model = ExecutionCostModel(
        ExecutionConfig(
            spread_mode="fixed", spread_value=0.0002, slippage_mode="fixed", slippage_value=0.00005
        )
    )
    p = model.execution_prices(1.1000)
    assert p.buy == Decimal("1.10015")
    assert p.sell == Decimal("1.09985")


def test_commission_per_unit() -> None:
    model = ExecutionCostModel(ExecutionConfig(commission_per_unit=0.00007))
    assert model.commission(100000) == Decimal("7.0")
    assert model.commission(-100000) == Decimal("7.0")
    assert model.commission(0) == Decimal("0")


def test_total_execution_charge() -> None:
    model = ExecutionCostModel(ExecutionConfig(spread_value=0.0002, commission_per_unit=0.00007))
    # Buy 100k at mid 1.10: half-spread cost 0.0001 * 100k + 7 commission.
    assert model.total_execution_charge(100000, 1.10) == Decimal("10.0") + Decimal("7.0")


def test_from_pips() -> None:
    cfg = ExecutionConfig.from_pips(
        pip_size=0.0001, spread_pips=2.0, slippage_pips=0.5, commission_per_unit=0.0
    )
    assert cfg.spread_value == 0.0002
    assert cfg.slippage_value == 0.00005
    assert cfg.slippage_mode == "fixed"


def test_invalid_config() -> None:
    with pytest.raises(ValueError):
        ExecutionConfig(spread_mode="dynamic")
    with pytest.raises(ValueError):
        ExecutionConfig(slippage_mode="stochastic")
    with pytest.raises(ValueError):
        ExecutionConfig(spread_value=-0.1)
    with pytest.raises(ValueError):
        ExecutionConfig(commission_per_unit=-1.0)
