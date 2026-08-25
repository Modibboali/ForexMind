"""Reproducible processed-data pipeline (ForexMind Phase 1, section 26).

Reads the immutable raw MT5 CSVs, normalises to the canonical M1 schema,
validates, resamples to M5, and writes processed parquet files::

    data/processed/<INSTRUMENT>/m1.parquet
    data/processed/<INSTRUMENT>/m5.parquet

plus ``data/processed/manifest.json`` with per-instrument provenance and
quality metadata.  Raw files are never modified.

Usage:
    python -m tools.process_data [--instruments EURUSD GBPUSD] [--strict]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from forexmind.data.loaders import LoadConfig, load_many_concat
from forexmind.data.resampler import (
    CompletenessPolicy,
    ResampleConfig,
    resample_m1_to_m5,
)
from forexmind.data.schema import TIMESTAMP
from forexmind.data.validator import MarketDataValidator, summarize_gaps

from tools.common import INSTRUMENTS, PROCESSED_DATA_DIR, instrument_files


def process_instrument(
    instrument: str,
    *,
    strict_validation: bool = False,
    completeness: CompletenessPolicy = CompletenessPolicy.STRICT,
) -> dict[str, object]:
    files = instrument_files(instrument)
    cfg = LoadConfig(sep=",", has_header=False)
    res = load_many_concat(files, cfg)
    m1 = res.frame

    validator = MarketDataValidator()
    result = validator.validate(m1, instrument)
    status = "PASS" if result.is_valid else "FAIL"
    if strict_validation and not result.is_valid:
        raise RuntimeError(
            f"{instrument}: validation failed ({len(result.errors)} errors); "
            f"first: {result.errors[0]}"
        )

    m5 = resample_m1_to_m5(m1, ResampleConfig(completeness=completeness))

    out_dir = PROCESSED_DATA_DIR / instrument
    out_dir.mkdir(parents=True, exist_ok=True)
    m1.to_parquet(out_dir / "m1.parquet", index=False)
    m5.to_parquet(out_dir / "m5.parquet", index=False)

    manifest = {
        "instrument": instrument,
        "source_files": [str(f) for f in files],
        "row_count": len(m1),
        "first_timestamp": str(m1[TIMESTAMP].iloc[0]),
        "last_timestamp": str(m1[TIMESTAMP].iloc[-1]),
        "timezone": res.source_timezone,
        "missing_bar_stats": dict(result.gap_counts),
        "m5_row_count": len(m5),
        "validation_status": status,
        "validation_errors": [str(i) for i in result.errors],
        "processed_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    if not result.is_valid:
        (PROCESSED_DATA_DIR / f"{instrument}_validation_report.txt").write_text(
            summarize_gaps(result, top_n=20), encoding="utf-8"
        )
    print(
        f"  {instrument:<8} M1 rows={len(m1):,}  M5 rows={len(m5):,}  "
        f"{manifest['first_timestamp']} .. {manifest['last_timestamp']}  {status}"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Process ForexMind raw data into parquet")
    parser.add_argument("--instruments", nargs="*", default=list(INSTRUMENTS))
    parser.add_argument("--strict", action="store_true", help="fail on validation errors")
    parser.add_argument("--partial", action="store_true", help="use PARTIAL M5 completeness")
    args = parser.parse_args()

    completeness = CompletenessPolicy.PARTIAL if args.partial else CompletenessPolicy.STRICT
    manifests: list[dict[str, object]] = []
    for instr in args.instruments:
        try:
            manifests.append(
                process_instrument(instr, strict_validation=args.strict, completeness=completeness)
            )
        except Exception as exc:
            print(f"  !! failed {instr}: {exc}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DATA_DIR / "manifest.json").write_text(
        json.dumps(manifests, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nProcessed {len(manifests)} instrument(s) -> {PROCESSED_DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
