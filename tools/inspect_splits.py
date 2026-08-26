"""Split inspection tool (Phase 2, section 42).

Prints per-instrument train/validation/test ranges and verifies all boundary
conditions.

Usage:
    python -m tools.inspect_splits
"""

from __future__ import annotations

import argparse
import sys

from forexmind.data.splits import SPLIT_NAMES

from tools.common import make_split_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ForexMind temporal splits")
    parser.parse_args()

    ds = make_split_dataset()
    print(f"Split config: {ds.split_config.to_dict()}")
    print()
    header = f"{'Instrument':<10}" + "".join(f"{s.capitalize():<28}" for s in SPLIT_NAMES)
    print(header)
    print("-" * len(header))
    for instr in ds.instruments:
        cells = []
        for s in SPLIT_NAMES:
            isp = ds.split(instr, s)
            cells.append(
                f"{isp.start_ts.strftime('%Y-%m-%d')}..{isp.end_ts.strftime('%Y-%m-%d')}"
                f" ({isp.n_bars:,} bars)"
            )
        print(f"{instr:<10}" + "".join(f"{c:<28}" for c in cells))

    print()
    report = ds.verify_integrity(strict=True)
    print(f"Integrity check: {'PASS' if report['ok'] else 'FAIL'}")
    for instr, splits in report["instruments"].items():
        for s, block in splits.items():
            first_ts = block.get("first_ts")
            last_ts = block.get("last_ts")
            print(f"  {instr:<8} {s:<11} {first_ts} .. {last_ts}")
    if report["issues"]:
        for issue in report["issues"]:
            print(f"  !! {issue}")
        return 1
    print("\nAll splits strictly chronological, non-overlapping, per instrument. OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
