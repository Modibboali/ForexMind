"""Hard determinism / reproducibility tests.

Given the same dataset, configuration, seed, starting state and action
sequence, the simulator must produce identical observations, execution
prices, portfolio states, rewards, and termination conditions.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
from forexmind.config import EnvironmentConfig
from forexmind.data.dataset import InstrumentData, MarketDataset
from forexmind.environment import ForexEnvironment

from tests.synthetic import ohlc_m1


def _wavy_m1(n: int = 600) -> object:
    """Pseudo-random but deterministic OHLC path (seeded)."""
    rng = np.random.default_rng(123)
    o = 1.1000
    rows = []
    for _ in range(n):
        h = o + 0.002 * rng.random()
        lo = o - 0.002 * rng.random()
        c = o + 0.001 * (rng.random() - 0.5)
        rows.append((o, h, lo, c))
        o = c
    return ohlc_m1("2025-01-06 00:00", rows)


def _env_for(frame, *, seed: int = 7, start: int = 10, horizon: int = 40):
    from forexmind.config import (
        ExecutionConfig,
        MarginConfig,
        PositionSizingConfig,
    )

    ds = MarketDataset()
    ds.add(InstrumentData.from_m1("EURUSD", frame))
    cfg = EnvironmentConfig(
        execution=ExecutionConfig(spread_value=0.0002, commission_per_unit=0.00001),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("50")),
        sizing=PositionSizingConfig(mode="equity_fraction", fixed_units=Decimal("100000")),
        horizon=horizon,
        close_at_episode_end=True,
    )
    return ForexEnvironment(ds, cfg)


def _rollout(env: ForexEnvironment, actions: list[int]) -> list[object]:
    env.reset(seed=7, start_index=10)
    events: list[object] = []
    for a in actions:
        obs, reward, term, trunc, info = env.step(a)
        events.append(
            (
                float(obs.account.equity),
                str(obs.account.position_units),
                str(obs.account.unrealized_pnl),
                str(obs.account.realized_pnl),
                reward,
                str(info["execution_price"]),
                str(info["equity"]),
                term,
                trunc,
                info["liquidation"],
            )
        )
    return events


def test_identical_rollouts_across_instances() -> None:
    actions = [4, 3, 2, 1, 0, 4, 4, 3, 2, 2, 0, 1, 3, 4, 0]
    frame = _wavy_m1()
    e1 = _rollout(_env_for(frame), actions)
    e2 = _rollout(_env_for(frame), actions)
    e3 = _rollout(_env_for(frame), actions)
    assert e1 == e2 == e3


def test_identical_across_reset_reuse() -> None:
    actions = [4, 2, 0, 4, 3, 2, 1, 0, 4, 4]
    frame = _wavy_m1()
    env = _env_for(frame)
    first = _rollout(env, actions)
    second = _rollout(env, actions)  # reuse the same instance
    assert first == second


def test_seed_controls_episode_start_sampling() -> None:
    frame = _wavy_m1()
    env = _env_for(frame)
    env.reset(seed=11, start_index=None)
    s1 = env.current_obs_index
    env.reset(seed=11, start_index=None)
    s2 = env.current_obs_index
    assert s1 == s2  # same seed -> same sampled start
    env.reset(seed=12, start_index=None)
    s3 = env.current_obs_index
    assert 0 <= s1 < 120 and 0 <= s3 < 120  # valid M5 index range


def test_explicit_start_overrides_seed() -> None:
    frame = _wavy_m1()
    env = _env_for(frame)
    env.reset(seed=11, start_index=25)
    assert env.current_obs_index == 25
