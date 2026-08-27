"""Build the shared read-only dataset store (PPO memory audit).

Converts each instrument's processed parquet into per-column ``.npy`` files
that workers open read-only via ``np.load(mmap_mode='r')``, so all worker
processes share the same physical pages instead of each materialising a
~3 GB private copy.

Run ONCE per data directory (on Kaggle, run this before training):

    python -m tools.build_shared_dataset --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import time

from forexmind.data.splits import DEFAULT_INSTRUMENT_ORDER
from forexmind.training.data import DEFAULT_PROCESSED_DIR
from forexmind.training.dataset_mmap import (
    build_shared_store,
    shared_store_dir,
    store_available,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the shared dataset store.")
    parser.add_argument("--processed-dir", type=str, default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--instruments", nargs="+", default=list(DEFAULT_INSTRUMENT_ORDER))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = shared_store_dir(args.processed_dir)
    if store_available(args.processed_dir) and not args.overwrite:
        print(f"Shared store already present: {root} (use --overwrite to rebuild)")
        return
    t0 = time.perf_counter()
    out = build_shared_store(
        args.processed_dir,
        tuple(args.instruments),
        overwrite=args.overwrite,
    )
    size_mb = sum(p.stat().st_size for p in out.rglob("*.npy")) / 1e6
    print(
        f"Built shared store at {out} "
        f"({size_mb:.0f} MB of .npy files) in {time.perf_counter() - t0:.1f}s"
    )
    print("Workers will now share these pages via the OS page cache.")


if __name__ == "__main__":
    main()
