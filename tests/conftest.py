"""Shared pytest fixtures for ForexMind tests."""

from __future__ import annotations

from decimal import Decimal

import pytest
from forexmind.config import (
    EnvironmentConfig,
    ExecutionConfig,
    MarginConfig,
    PositionSizingConfig,
    RewardConfig,
)


@pytest.fixture
def zero_cost_config() -> EnvironmentConfig:
    """No spread / slippage / commission; fixed-unit sizing for exact math."""
    return EnvironmentConfig(
        execution=ExecutionConfig(spread_mode="fixed", spread_value=0.0),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")),
        sizing=PositionSizingConfig(mode="fixed_units", fixed_units=Decimal("10000")),
        reward=RewardConfig(reward_type="log_equity_return"),
        close_at_episode_end=False,
    )


@pytest.fixture
def spread_config() -> EnvironmentConfig:
    """Fixed 2-pip spread (0.0002) with fixed-unit sizing."""
    return EnvironmentConfig(
        execution=ExecutionConfig(spread_mode="fixed", spread_value=0.0002),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")),
        sizing=PositionSizingConfig(mode="fixed_units", fixed_units=Decimal("10000")),
    )


@pytest.fixture
def default_env_config() -> EnvironmentConfig:
    from forexmind.config import default_config

    return default_config(
        initial_balance="10000",
        leverage=100,
        spread_value=0.0002,
        sizing_mode="fixed_units",
        fixed_units="10000",
    )
