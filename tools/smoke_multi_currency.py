"""Real-data multi-pair accounting smoke test (Phase 3.1, section 33).

Runs the environment over real processed data for a set of instruments and
prints the account-currency accounting fields (raw PnL, converted PnL, equity,
gross exposure, margin, free margin, reward) for each.  This validates the
multi-currency accounting layer against real prices without any training.

Usage:
    python -m tools.smoke_multi_currency [--instrument EURUSD USDJPY ...] [--steps 50]
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal

from forexmind.config import (
    EnvironmentConfig,
    ExecutionConfig,
    MarginConfig,
    PositionSizingConfig,
    RewardConfig,
)
from forexmind.data.dataset import MarketDataset
from forexmind.environment import ForexEnvironment

from tools.common import INSTRUMENTS, load_processed_instrument


def make_config(instrument: str, account_currency: str = "USD") -> EnvironmentConfig:
    """Per-instrument pip-correct spread for the smoke test.

    Non-JPY pairs: 2 pips of 0.0001 = 0.0002.  JPY pairs: 2 pips of 0.01 = 0.02.
    """
    from forexmind.environment.instruments import instrument_spec

    pip = instrument_spec(instrument).pip_size
    return EnvironmentConfig(
        execution=ExecutionConfig(spread_value=2 * pip, commission_per_unit=0.0),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("50")),
        sizing=PositionSizingConfig(mode="equity_fraction"),
        reward=RewardConfig(reward_type="log_equity_return"),
        account_currency=account_currency,
        close_at_episode_end=False,
    )


def run_instrument(instrument: str, steps: int, start_index: int) -> None:
    data = load_processed_instrument(instrument)
    ds = MarketDataset()
    ds.add(data)
    config = make_config(instrument)
    env = ForexEnvironment(ds, config, instrument=instrument)
    _obs, info = env.reset(seed=12345, start_index=start_index, horizon=steps)

    print(f"\n=== {instrument} ===")
    print(
        f"base/quote       : {info['base_currency']}/{info['quote_currency']}  "
        f"account currency: {info['account_currency']}"
    )
    print(f"observation      : {info['timestamp']}")

    # A deterministic long then flat sequence.
    actions = []
    half = max(steps // 2, 1)
    actions += [1.0] * half
    actions += [0.0] * (steps - half)
    total_reward = 0.0
    final: dict = {}
    for a in actions:
        _obs, reward, _term, _trunc, info = env.step(a)
        total_reward += float(reward)
        final = info
        if info.get("terminated") or info.get("truncated"):
            break

    print(f"position         : {final.get('position')}  units={final.get('position_units')}")
    print(f"execution price  : {final.get('execution_price')}")
    print(f"raw PnL          : {final.get('raw_pnl')} {final.get('raw_pnl_currency')}")
    print(f"converted PnL    : {final.get('converted_pnl')} {final.get('account_currency')}")
    print(f"equity           : {final.get('equity')} {final.get('account_currency')}")
    print(f"gross exposure   : {final.get('gross_exposure')} {final.get('account_currency')}")
    print(f"margin used      : {final.get('margin_used')} {final.get('account_currency')}")
    print(f"free margin      : {final.get('free_margin')} {final.get('account_currency')}")
    print(f"reward (cumulative): {total_reward:.8f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-data multi-pair accounting smoke test")
    parser.add_argument(
        "--instrument", nargs="*", default=["EURUSD", "USDJPY", "USDCHF", "USDCAD", "GBPUSD"]
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=10000)
    args = parser.parse_args()

    available = set(INSTRUMENTS)
    for instrument in args.instrument:
        key = instrument.upper()
        if key not in available:
            print(f"unknown instrument {key!r}; available {sorted(available)}", file=sys.stderr)
            return 2
        run_instrument(key, args.steps, args.start_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
