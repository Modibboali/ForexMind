"""Focused tests for the single-trade terminal-reward lifecycle."""

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
from forexmind.data.dataset import InstrumentData, MarketDataset
from forexmind.environment import EnvironmentError, ForexEnvironment

from tests.synthetic import m5_from_closes, synth_m1_from_m5


def _env(closes: list[float], *, horizon: int, spread: float = 0.0) -> ForexEnvironment:
    m5 = m5_from_closes("2025-01-06 00:00", closes)
    ds = MarketDataset()
    ds.add(InstrumentData.from_m1("EURUSD", synth_m1_from_m5(m5)))
    cfg = EnvironmentConfig(
        execution=ExecutionConfig(spread_value=spread),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")),
        sizing=PositionSizingConfig(mode="fixed_units", fixed_units=Decimal("10000")),
        reward=RewardConfig(idle_penalty=0.001, timeout_penalty=0.002),
        max_episode_steps=horizon,
    )
    env = ForexEnvironment(ds, cfg)
    env.reset(seed=0, start_index=0, horizon=horizon)
    return env


def test_no_trade_is_terminal_with_idle_penalty() -> None:
    env = _env([1.10] * 6, horizon=3)
    rewards = []
    for _ in range(3):
        _obs, reward, terminated, truncated, info = env.step(0)  # HOLD
        rewards.append(reward)
    assert rewards[:-1] == [0.0, 0.0]
    assert rewards[-1] == pytest.approx(-0.001)
    assert terminated and not truncated
    assert info["termination_reason"] == "timeout_no_trade"


def test_profitable_long_closes_immediately_with_terminal_reward() -> None:
    env = _env([1.10, 1.10, 1.12, 1.12, 1.12], horizon=10)
    assert env.step(9)[1] == 0.0  # open
    assert env.step(0)[1] == 0.0  # hold
    _obs, reward, terminated, truncated, info = env.step(1)  # FLAT
    assert terminated and not truncated
    assert reward == pytest.approx(0.02)
    assert info["termination_reason"] == "voluntary_close"
    assert info["position"] == "flat"


def test_open_position_timeout_is_liquidated_and_penalized() -> None:
    env = _env([1.10, 1.10, 1.12, 1.12], horizon=2)
    env.step(9)
    _obs, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert reward == pytest.approx(0.018)
    assert info["termination_reason"] == "timeout_open_position"
    assert info["timeout"] is True
    assert env.portfolio is not None and env.portfolio.position.is_flat


@pytest.mark.parametrize(
    ("open_action", "closes", "positive"),
    [
        (9, [1.10, 1.10, 1.12, 1.12], True),
        (9, [1.10, 1.10, 1.08, 1.08], False),
        (2, [1.10, 1.10, 1.08, 1.08], True),
        (2, [1.10, 1.10, 1.12, 1.12], False),
    ],
)
def test_terminal_reward_has_correct_long_and_short_sign(
    open_action: int, closes: list[float], positive: bool
) -> None:
    env = _env(closes, horizon=10)
    env.step(open_action)
    env.step(0)
    _obs, reward, terminated, truncated, _info = env.step(1)
    assert terminated and not truncated
    assert (reward > 0) is positive


def test_break_even_trade_has_zero_reward() -> None:
    env = _env([1.10, 1.10, 1.10, 1.10], horizon=10)
    env.step(9)
    env.step(0)
    _obs, reward, terminated, truncated, _info = env.step(1)
    assert terminated and not truncated
    assert reward == pytest.approx(0.0)


def test_spread_is_reflected_once_in_net_trade_result() -> None:
    env = _env([1.10, 1.10, 1.12, 1.12], horizon=10, spread=0.0002)
    env.step(9)
    env.step(0)
    _obs, reward, terminated, truncated, info = env.step(1)
    assert terminated and not truncated
    # 10,000 units buy at 1.1001 and sell at 1.1199: 198 account-currency
    # P&L, with no second subtraction of the spread in the reward.
    assert reward == pytest.approx(198 / 10000)
    assert info["net_trade_pnl"] == Decimal("198.0")


def test_direct_reversal_is_ignored() -> None:
    env = _env([1.10] * 8, horizon=5)
    env.step(9)  # long
    _obs, reward, terminated, _truncated, info = env.step(2)  # SHORT_100
    assert reward == 0.0 and not terminated
    assert info["position"] == "long"
    assert info["action_diagnostics"]["ignored_action"] == 1


def test_terminal_environment_rejects_second_step() -> None:
    env = _env([1.10, 1.10, 1.11, 1.11], horizon=5)
    env.step(9)
    env.step(1)
    with pytest.raises(EnvironmentError):
        env.step(1)
