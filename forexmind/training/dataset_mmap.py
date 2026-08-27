"""Read-only shared memory-mapped training dataset (PPO memory audit).

WHY: every spawned worker used to materialise the full 7-instrument M1+M5
parquet into its own private pandas frames (~2.95 GB/worker measured ->
~300 GB at 102 workers on Kaggle).  This module stores each column as a
standalone ``.npy`` file and re-opens them with ``np.load(mmap_mode='r')``;
because every worker maps the SAME files, the OS shares the physical pages,
so the real RAM footprint is roughly one copy regardless of worker count.

The pandas frames are rebuilt zero-copy with
``pd.DataFrame({col: memmap}, copy=False)`` (verified: every column shares
its memmap buffer, including mixed dtypes - datetime64/float64/int64/bool).
Values and dtypes are bit-identical to ``pd.read_parquet`` (verified by
tests/test_dataset_mmap.py), so environment semantics, execution,
accounting, observations, and episode sampling are unchanged.

Layout::

    <store_dir>/<INSTR>/m1/{timestamp,open,high,low,close}.npy
    <store_dir>/<INSTR>/m5/{timestamp,open,high,low,close,n_observations,is_complete}.npy
    <store_dir>/manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forexmind.data.dataset import InstrumentData
from forexmind.data.splits import DEFAULT_INSTRUMENT_ORDER, SplitConfig, SplitDataset
from forexmind.training.data import load_processed_from_dir

FRAMETIMES = ("m1", "m5")


def shared_store_dir(processed_dir: str | Path) -> Path:
    """The shared store lives next to the processed parquet data."""
    return Path(processed_dir) / "_shared"


def store_available(processed_dir: str | Path) -> bool:
    """True when a complete shared store already exists for this data dir."""
    root = shared_store_dir(processed_dir)
    if not (root / "manifest.json").is_file():
        return False
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    for instr, frames in manifest.get("instruments", {}).items():
        for tf in FRAMETIMES:
            for col in frames.get(tf, {}):
                if not (root / instr / tf / f"{col}.npy").is_file():
                    return False
    return True


def build_shared_store(
    processed_dir: str | Path,
    instruments: tuple[str, ...] = DEFAULT_INSTRUMENT_ORDER,
    *,
    out_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert each instrument's processed parquet into per-column .npy files.

    Reads through the reference path (``load_processed_from_dir``) so the
    stored values are bit-identical to what a parquet worker would load.
    """
    root = Path(out_dir) if out_dir is not None else shared_store_dir(processed_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"processed_dir": str(Path(processed_dir)), "instruments": {}}
    for instr in instruments:
        data = load_processed_from_dir(instr, processed_dir)
        instr_meta: dict[str, object] = {}
        for tf in FRAMETIMES:
            frame = getattr(data, tf)
            tf_dir = root / instr.upper() / tf
            tf_dir.mkdir(parents=True, exist_ok=True)
            cols: dict[str, str] = {}
            for col in frame.columns:
                path = tf_dir / f"{col}.npy"
                if overwrite or not path.is_file():
                    np.save(path, frame[col].to_numpy())
                cols[str(col)] = str(frame[col].dtype)
            instr_meta[tf] = cols
        manifest["instruments"][instr.upper()] = instr_meta
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def open_instrument_data(store_dir: str | Path, instrument: str) -> InstrumentData:
    """Open one instrument's columns as a zero-copy memmap-backed InstrumentData."""
    root = Path(store_dir)
    instr = instrument.upper()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    instr_meta = manifest["instruments"][instr]
    frames: dict[str, pd.DataFrame] = {}
    for tf in FRAMETIMES:
        tf_dir = root / instr / tf
        cols_meta = instr_meta[tf]
        col_arrays: dict[str, np.ndarray] = {}
        for col in cols_meta:
            col_arrays[col] = np.load(tf_dir / f"{col}.npy", mmap_mode="r")
        # copy=False keeps every column sharing its memmap buffer (zero-copy).
        frames[tf] = pd.DataFrame(col_arrays, copy=False)
    return InstrumentData(instrument=instr, m1=frames["m1"], m5=frames["m5"])


def make_mmap_dataset(
    store_dir: str | Path,
    split_config: SplitConfig | None = None,
    instruments: tuple[str, ...] = DEFAULT_INSTRUMENT_ORDER,
) -> SplitDataset:
    """A :class:`SplitDataset` whose loader returns zero-copy memmap data."""
    cfg = split_config or SplitConfig.default()
    return SplitDataset(cfg, lambda k: open_instrument_data(store_dir, k), instruments)


def resolve_dataset(
    *,
    processed_dir: str | Path,
    split_config: SplitConfig | None = None,
    instruments: tuple[str, ...] = DEFAULT_INSTRUMENT_ORDER,
    backend: str = "auto",
) -> tuple[SplitDataset, str]:
    """Build the training dataset with the requested backend.

    ``backend``: ``"auto"`` (use the shared store if built, else parquet),
    ``"parquet"`` (current per-worker materialisation), ``"mmap"`` (require
    the shared store; raise a helpful error if it is missing).
    """
    from forexmind.training.data import make_training_dataset

    if backend == "auto":
        backend = "mmap" if store_available(processed_dir) else "parquet"
    if backend == "mmap":
        store = shared_store_dir(processed_dir)
        if not store_available(processed_dir):
            raise FileNotFoundError(
                f"shared dataset store not found at {store}. Build it first with:\n"
                f"  python -m tools.build_shared_dataset --processed-dir "
                f"{Path(processed_dir)}"
            )
        return make_mmap_dataset(store, split_config, instruments), "mmap"
    if backend == "parquet":
        return make_training_dataset(processed_dir, split_config, instruments), "parquet"
    raise ValueError(f"unknown dataset backend {backend!r}; use auto|parquet|mmap")
