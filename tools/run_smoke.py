"""Real-data end-to-end smoke test (ForexMind Phase 1, section 29).

Loads a real pair -> validates M1 -> builds M5 -> creates the environment ->
resets -> runs a deterministic action sequence twice -> verifies identical
results -> switches to a second instrument without changing simulator code.

Usage:
    python -m tools.run_smoke [--instrument EURUSD] [--second GBPUSD]
                              [--steps 50] [--start-index 10000]
"""

from __future__ import annotations

import argparse
import sys

from forexmind.config import default_config
from forexmind.data.dataset import InstrumentData, MarketDataset
from forexmind.data.loaders import LoadConfig, load_many_concat
from forexmind.data.resampler import ResampleConfig
from forexmind.data.validator import MarketDataValidator
from forexmind.environment import ForexEnvironment

from tools.common import instrument_files

# A fixed deterministic action sequence (discrete target-exposure indices).
DEFAULT_ACTION_SEQUENCE = [0, 1, 2, 3, 4, 2, 2, 1, 0, 4, 3, 2, 1, 0]


def build_dataset(instrument: str) -> InstrumentData:
    files = instrument_files(instrument)
    res = load_many_concat(files, LoadConfig(sep=",", has_header=False))
    result = MarketDataValidator().validate(res.frame, instrument)
    if not result.is_valid:
        print(f"  !! validation issues for {instrument}: {[str(e) for e in result.errors[:5]]}")
    return InstrumentData.from_m1(instrument, res.frame, ResampleConfig())


def run_episode(
    dataset: MarketDataset,
    instrument: str,
    *,
    start_index: int,
    steps: int,
    seed: int = 12345,
) -> tuple[float, dict[str, object]]:
    config = default_config(
        initial_balance="10000",
        leverage=50,
        spread_value=0.0002,
        commission_per_unit=0.0,
        sizing_mode="equity_fraction",
    )
    env = ForexEnvironment(dataset, config, instrument=instrument)
    _obs, info = env.reset(seed=seed, start_index=start_index, horizon=steps)
    actions = (DEFAULT_ACTION_SEQUENCE * (steps // len(DEFAULT_ACTION_SEQUENCE) + 1))[:steps]
    total_reward = 0.0
    final_info: dict[str, object] = {}
    for a in actions:
        _obs, reward, _terminated, _truncated, info = env.step(a)
        total_reward += reward
        final_info = info
        if info["truncated"] or info["terminated"]:
            break
    return total_reward, final_info


def main() -> int:
    parser = argparse.ArgumentParser(description="ForexMind real-data smoke test")
    parser.add_argument("--instrument", default="EURUSD")
    parser.add_argument("--second", default="GBPUSD")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=10000)
    args = parser.parse_args()

    print(f"Loading {args.instrument} ...")
    data = build_dataset(args.instrument)
    dataset = MarketDataset()
    dataset.add(data)
    print(f"  M1 rows={len(data.m1):,}  M5 rows={len(data.m5):,}")
    print(f"  range {data.first_timestamp} .. {data.last_timestamp}")

    r1, info1 = run_episode(
        dataset, args.instrument, start_index=args.start_index, steps=args.steps
    )
    r2, info2 = run_episode(
        dataset, args.instrument, start_index=args.start_index, steps=args.steps
    )

    print("\nRun 1 final:")
    for k in (
        "equity",
        "balance",
        "position_units",
        "realized_pnl",
        "unrealized_pnl",
        "margin_used",
        "drawdown",
        "execution_price",
    ):
        print(f"  {k:<16} = {info1.get(k)}")
    print(f"  total reward    = {r1:.10f}")

    assert r1 == r2, "reproducibility FAILED: rewards differ across runs"
    assert info1["equity"] == info2["equity"], "reproducibility FAILED: equity differs"
    print(f"\nDeterminism check: PASS (reward={r1:.10f} identical on both runs)")

    # Switch to a second instrument without changing any simulator code.
    print(f"\nSwitching instrument -> {args.second} (same simulator code)")
    data2 = build_dataset(args.second)
    dataset2 = MarketDataset()
    dataset2.add(data2)
    r3, info3 = run_episode(dataset2, args.second, start_index=args.start_index, steps=args.steps)
    print(f"  {args.second}: M1 rows={len(data2.m1):,} M5 rows={len(data2.m5):,}")
    print(
        f"  final equity={info3['equity']}  balance={info3['balance']}  "
        f"realized={info3['realized_pnl']}  reward={r3:.10f}"
    )
    print("\nSmoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
