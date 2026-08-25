"""Critical anti-lookahead tests.

We construct tiny synthetic datasets where the price after an observation is
known to be profitable and verify that the agent cannot benefit from
information that occurs after the observation timestamp: execution must use
the *next* M1 open, never a price inside the observed bar.
"""

from __future__ import annotations

from decimal import Decimal

from forexmind.config import EnvironmentConfig
from forexmind.data.dataset import InstrumentData, MarketDataset
from forexmind.environment import ForexEnvironment

from tests.synthetic import m1_frame

# M5[0] closes at 1.1000.  The very next M1 open jumps to 1.1100.
JUMP_UP_M1 = m1_frame(
    [
        ("2025-01-06 00:00", 1.1000, 1.1000, 1.1000, 1.1000),
        ("2025-01-06 00:01", 1.1000, 1.1000, 1.1000, 1.1000),
        ("2025-01-06 00:02", 1.1000, 1.1000, 1.1000, 1.1000),
        ("2025-01-06 00:03", 1.1000, 1.1000, 1.1000, 1.1000),
        ("2025-01-06 00:04", 1.1000, 1.1000, 1.1000, 1.1000),
        ("2025-01-06 00:05", 1.1100, 1.1100, 1.1100, 1.1100),  # next open!
        ("2025-01-06 00:06", 1.1100, 1.1100, 1.1100, 1.1100),
        ("2025-01-06 00:07", 1.1100, 1.1100, 1.1100, 1.1100),
        ("2025-01-06 00:08", 1.1100, 1.1100, 1.1100, 1.1100),
        ("2025-01-06 00:09", 1.1100, 1.1100, 1.1100, 1.1100),
    ]
)

# M5[0] closes at 1.1000.  The next M1 open drops to 1.0900.
JUMP_DOWN_M1 = m1_frame(
    [
        ("2025-01-06 00:00", 1.1000, 1.1000, 1.1000, 1.1000),
        ("2025-01-06 00:01", 1.1000, 1.1000, 1.1000, 1.1000),
        ("2025-01-06 00:02", 1.1000, 1.1000, 1.1000, 1.1000),
        ("2025-01-06 00:03", 1.1000, 1.1000, 1.1000, 1.1000),
        ("2025-01-06 00:04", 1.1000, 1.1000, 1.1000, 1.1000),
        ("2025-01-06 00:05", 1.0900, 1.0900, 1.0900, 1.0900),  # next open!
        ("2025-01-06 00:06", 1.0900, 1.0900, 1.0900, 1.0900),
        ("2025-01-06 00:07", 1.0900, 1.0900, 1.0900, 1.0900),
        ("2025-01-06 00:08", 1.0900, 1.0900, 1.0900, 1.0900),
        ("2025-01-06 00:09", 1.0900, 1.0900, 1.0900, 1.0900),
    ]
)


def _env(frame, *, units: str = "10000") -> ForexEnvironment:
    from forexmind.config import (
        ExecutionConfig,
        MarginConfig,
        PositionSizingConfig,
    )

    ds = MarketDataset()
    ds.add(InstrumentData.from_m1("EURUSD", frame))
    cfg = EnvironmentConfig(
        execution=ExecutionConfig(spread_value=0.0),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")),
        sizing=PositionSizingConfig(mode="fixed_units", fixed_units=Decimal(units)),
        horizon=4,
    )
    return ForexEnvironment(ds, cfg)


def test_cannot_execute_at_observed_close_after_jump_up() -> None:
    env = _env(JUMP_UP_M1)
    obs, _ = env.reset(seed=0, start_index=0)
    # The observed M5[0] close is 1.1000.
    assert float(obs.market_window[-1].close) == 1.1000

    obs, reward, _terminated, _truncated, info = env.step(4)  # full long
    # Execution must occur at the NEXT M1 open (1.1100), not the observed close.
    assert info["execution_price"] == Decimal("1.1100")
    # Marked at M5[1] close (1.1100): no unrealised profit from the jump.
    assert obs.account.unrealized_pnl == Decimal("0")
    assert reward == 0.0
    assert env.portfolio is not None and env.portfolio.equity == Decimal("10000")


def test_cannot_execute_at_observed_close_after_jump_down() -> None:
    env = _env(JUMP_DOWN_M1)
    obs, _ = env.reset(seed=0, start_index=0)
    obs, reward, _terminated, _truncated, info = env.step(0)  # full short
    # Execution at the NEXT M1 open (1.0900), not the observed close (1.1000).
    assert info["execution_price"] == Decimal("1.0900")
    assert obs.account.unrealized_pnl == Decimal("0")
    assert reward == 0.0


def test_no_lookahead_general_sequence() -> None:
    """The agent's reward at step t depends only on data up to M5[t+1]."""
    env = _env(JUMP_UP_M1)
    env.reset(seed=0, start_index=0)
    obs, reward, _term, _trunc, info = env.step(4)
    # Reward only reflects equity marked at M5[1] close (the observation), and
    # execution at the next open -- never a price hidden inside the bar.
    assert info["timestamp"] == obs.timestamp
    # equity exactly initial because buy price == mark price (no costs).
    assert reward == 0.0
