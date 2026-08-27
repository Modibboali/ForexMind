"""SAC training launcher.

Usage::

    python -m forexmind.training.train_sac --config configs/sac_cpu.yaml --seeds 1 2 3
    python -m forexmind.training.train_sac --seeds 1 2 3 --workers 32
    python -m forexmind.training.train_sac --config configs/sac_cpu.yaml \
        --resume runs/sac_cpu_seed42/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse

from forexmind.training.cli import add_common_args, run_multiseed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SAC policy on ForexMind.")
    add_common_args(parser, algorithm="sac")
    args = parser.parse_args()
    run_multiseed(args, algorithm="sac")


if __name__ == "__main__":
    main()
