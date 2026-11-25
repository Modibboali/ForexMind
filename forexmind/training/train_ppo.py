"""PPO training launcher.

Usage::

    python -m forexmind.training.train_ppo --config configs/ppo_cpu.yaml
    python -m forexmind.training.train_ppo --seeds 1 2 3 --workers 32
"""

from __future__ import annotations

import argparse

from forexmind.training.cli import add_common_args, run_multiseed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO policy on ForexMind.")
    add_common_args(parser, algorithm="ppo")
    args = parser.parse_args()
    run_multiseed(args, algorithm="ppo")


if __name__ == "__main__":
    main()
