"""Dataset inspection / reporting script (ForexMind Phase 1, section 25).

Discovers the raw data files, determines their format, inspects headers and
sample rows, and produces a concise per-instrument report covering date range,
timezone/format, missing periods, duplicates, OHLC validity, row counts, and
gap statistics.  Raw files are treated as immutable -- never renamed/rewritten.

Usage:
    python -m tools.inspect_dataset [--instrument EURUSD] [--top-gaps 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forexmind.data.loaders import LoadConfig, load_many_concat
from forexmind.data.validator import MarketDataValidator, summarize_gaps

from tools.common import INSTRUMENTS, REPORTS_DIR, instrument_files


def inspect_instrument(instrument: str, top_gaps: int = 8) -> dict[str, object]:
    files = instrument_files(instrument)
    print(f"\n=== {instrument} ===  ({len(files)} source files)")
    for f in files[:3]:
        print(f"  source: {f.name}")
    if len(files) > 3:
        print(f"  ... and {len(files) - 3} more")

    cfg = LoadConfig(sep=",", has_header=False)
    res = load_many_concat(files, cfg)
    frame = res.frame
    validator = MarketDataValidator()
    result = validator.validate(frame, instrument)

    first, last = frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1]
    missing_stats = dict(result.gap_counts)

    print(f"  rows            : {len(frame):,}")
    print(f"  date range      : {first} .. {last}")
    print(f"  source timezone : {res.source_timezone!r}")
    print(
        f"  validation      : {'PASS' if result.is_valid else 'FAIL'}  "
        f"({len(result.errors)} errors, {len(result.warnings)} warnings)"
    )
    for issue in result.errors[:10]:
        print(f"    ERROR {issue}")
    for issue in result.warnings[:5]:
        print(f"    warn  {issue}")
    print(summarize_gaps(result, top_n=top_gaps))

    # Duplicate + OHLC sanity numbers.
    dup_ts = int(frame["timestamp"].duplicated().sum())
    print(f"  duplicate timestamps : {dup_ts:,}")
    ohlc = frame[["open", "high", "low", "close"]]
    nan = int(ohlc.isna().sum().sum())
    inf = int(ohlc.isin([float("inf"), float("-inf")]).sum().sum())
    print(f"  NaN cells : {nan:,}   inf cells : {inf:,}")

    return {
        "instrument": instrument,
        "source_files": [str(f) for f in files],
        "row_count": len(frame),
        "first_timestamp": str(first),
        "last_timestamp": str(last),
        "timezone": res.source_timezone,
        "validation_status": "PASS" if result.is_valid else "FAIL",
        "error_codes": [i.code for i in result.errors],
        "missing_bar_stats": missing_stats,
        "duplicate_timestamps": dup_ts,
        "nan_cells": int(nan),
        "inf_cells": int(inf),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect ForexMind raw datasets")
    parser.add_argument("--instrument", choices=list(INSTRUMENTS), default=None)
    parser.add_argument("--top-gaps", type=int, default=8)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else list(INSTRUMENTS)
    report: dict[str, object] = {}
    for instr in instruments:
        try:
            report[instr] = inspect_instrument(instr, top_gaps=args.top_gaps)
        except Exception as exc:
            print(f"  !! failed to inspect {instr}: {exc}")
            report[instr] = {"instrument": instr, "error": str(exc)}

    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / "dataset_report.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nDataset report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
