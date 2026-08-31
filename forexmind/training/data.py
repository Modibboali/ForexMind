"""Training dataset access (Phase 3).

Workers consume the Phase-1 *processed* parquet data (never raw CSVs), loaded
lazily per instrument and cached.  A startup report (§26) prints the memory
footprint estimate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forexmind.data.dataset import InstrumentData
from forexmind.data.splits import DEFAULT_INSTRUMENT_ORDER, SplitConfig, SplitDataset

DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def load_processed_from_dir(instrument: str, processed_dir: str | Path) -> InstrumentData:
    """Load one instrument's m1/m5 parquet into InstrumentData."""
    d = Path(processed_dir) / instrument.upper()
    m1 = pd.read_parquet(d / "m1.parquet")
    m5 = pd.read_parquet(d / "m5.parquet")
    return InstrumentData(instrument=instrument.upper(), m1=m1, m5=m5)


def make_training_dataset(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    split_config: SplitConfig | None = None,
    instruments: tuple[str, ...] = DEFAULT_INSTRUMENT_ORDER,
) -> SplitDataset:
    cfg = split_config or SplitConfig.default()
    return SplitDataset(cfg, lambda k: load_processed_from_dir(k, processed_dir), instruments)


def dataset_summary(dataset: SplitDataset) -> dict[str, object]:
    """Startup report for §26: instruments, rows, memory footprint estimate."""
    total_m5 = 0
    total_m1 = 0
    per_instrument: dict[str, object] = {}
    for instr in dataset.instruments:
        data = dataset.load(instr)
        n1, n5 = len(data.m1), len(data.m5)
        total_m1 += n1
        total_m5 += n5
        # float64 OHLC + datetime64 timestamp per row, roughly.
        per_instrument[instr] = {
            "m1_rows": n1,
            "m5_rows": n5,
            "est_mb": round((n1 * 9 * 8 + n5 * 9 * 8) / 1e6, 1),
        }
    est_bytes = total_m1 * 9 * 8 + total_m5 * 9 * 8
    return {
        "dataset_loaded": True,
        "num_instruments": len(dataset.instruments),
        "total_m1_rows": int(total_m1),
        "total_m5_rows": int(total_m5),
        "estimated_memory_mb": round(est_bytes / 1e6, 1),
        "per_instrument": per_instrument,
    }
