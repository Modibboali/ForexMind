"""Observation inspection tool (Phase 2, section 41).

Displays a single encoded observation at a given index for debugging and
leakage inspection: timestamp, market window shape/values, account state,
time features, and instrument representation.

Usage:
    python -m tools.inspect_observation --instrument EURUSD --split train --index 1000
"""

from __future__ import annotations

import argparse
import sys

from forexmind.config import default_config
from forexmind.data.dataset import MarketDataset
from forexmind.environment import ForexEnvironment
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.schema import (
    ACCOUNT_FEATURE_NAMES,
    DEFAULT_MARKET_FEATURES,
    TIME_FEATURE_NAMES,
)
from forexmind.observation.window import MarketWindowBuilder, WindowConfig

from tools.common import load_processed_instrument, make_split_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an encoded observation")
    parser.add_argument("--instrument", default="EURUSD")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--index", type=int, default=1000)
    parser.add_argument("--context-length", type=int, default=64)
    args = parser.parse_args()

    ds = make_split_dataset()
    data = load_processed_instrument(args.instrument)
    cfg = default_config(
        initial_balance="10000",
        leverage=50,
        spread_value=0.0002,
        sizing_mode="equity_fraction",
    )
    mds = MarketDataset()
    mds.add(data)
    env = ForexEnvironment(mds, cfg, instrument=args.instrument)

    start, end = ds.split_config.range(args.split)
    builder = MarketWindowBuilder(
        args.instrument, data.m5, start, end, WindowConfig(context_length=args.context_length)
    )
    encoder = ObservationEncoder(
        EncoderConfig(
            context_length=args.context_length,
            initial_balance=cfg.margin.initial_balance,
        )
    )

    if not builder.is_eligible(args.index):
        print(
            f"index {args.index} is not a valid observation start for "
            f"{args.instrument}/{args.split} (eligible "
            f"[{builder.min_valid_index()}, {builder.max_valid_index()}])"
        )
        return 1

    obs, _info = env.reset(seed=0, instrument=args.instrument, start_index=args.index, horizon=1)
    window = builder.build(args.index)
    encoded = encoder.encode(obs, window)

    print(f"instrument       : {encoded.instrument}")
    print(f"split            : {args.split}")
    print(f"step_index       : {encoded.step_index}")
    print(f"timestamp        : {encoded.timestamp}")
    print(f"spec             : {encoded.spec.to_dict()}")
    print(f"market shape     : {encoded.market.shape}  dtype={encoded.market.dtype}")
    print(f"account shape    : {encoded.account.shape}  dtype={encoded.account.dtype}")
    print(f"time shape       : {encoded.time.shape}  dtype={encoded.time.dtype}")
    print(
        f"instrument shape : {encoded.instrument_vec.shape}  dtype={encoded.instrument_vec.dtype}"
    )
    print(f"encoded shape    : {encoded.encoded.shape}")

    print("\nmarket (first 3 bars, per feature):")
    feats = DEFAULT_MARKET_FEATURES
    print("  " + "  ".join(f"{f:>12}" for f in feats))
    for i in range(min(3, encoded.market.shape[0])):
        print("  " + "  ".join(f"{v:12.6f}" for v in encoded.market[i]))
    print("  ...")
    print("  closes[-3:] = " + ", ".join(f"{c:.6f}" for c in encoded.closes[-3:]))
    print(f"  prior_close = {encoded.prior_close:.6f}")

    print("\naccount features:")
    for name, v in zip(ACCOUNT_FEATURE_NAMES, encoded.account, strict=True):
        print(f"  {name:<28} {v:+.6f}")

    print("\ntime features:")
    for name, v in zip(TIME_FEATURE_NAMES, encoded.time, strict=True):
        print(f"  {name:<30} {v:+.6f}")

    print("\ninstrument vector:")
    print("  " + ", ".join(f"{v:.1f}" for v in encoded.instrument_vec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
