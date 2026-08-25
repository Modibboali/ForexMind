"""Shared helpers for ForexMind tool scripts."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "historical_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

# Instruments supplied in the raw dataset (Phase 1).
INSTRUMENTS: tuple[str, ...] = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
)


def instrument_files(instrument: str) -> list[Path]:
    """Return the sorted raw source files for one instrument."""
    d = RAW_DATA_DIR / instrument.upper()
    if not d.is_dir():
        raise FileNotFoundError(f"no raw data directory for instrument {instrument}: {d}")
    return sorted(d.glob("*.csv"))
