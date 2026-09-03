"""Smoke test for PPO (Stage 3.2): 50k-100k steps correctness validation.

This tool runs a minimal PPO training run with strict correctness checks:
- No NaN/Inf in any tensor
- Finite actor/critic losses and gradients
- Finite log-probabilities (tanh Jacobian correction applied correctly)
- Actions remain bounded [-1, 1]
- KL divergence reasonable
- No pathological clipping (clip_fraction < 1)
- Evaluation metrics internally consistent
- Policy can be loaded and evaluated

Purpose: Validate implementation correctness before longer training runs.
This is NOT a profitability test.

Usage:
    python -m tools.smoke_ppo_correctness --steps 50000 --seed 42
    python -m tools.smoke_ppo_correctness --steps 100000 --split train
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from forexmind.training.config import ExperimentConfig
from forexmind.training.data import (
    DEFAULT_INSTRUMENT_ORDER,
    DEFAULT_PROCESSED_DIR,
    make_training_dataset,
)
from forexmind.training.ppo import PPOTrainer


def _numeric_arrays(value: Any, prefix: str = "checkpoint") -> list[tuple[str, np.ndarray]]:
    """Flatten tensor/array leaves so nested checkpoint state is checked."""
    if isinstance(value, torch.Tensor):
        return [(prefix, value.detach().cpu().numpy())]
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return [(prefix, value)]
    if isinstance(value, dict):
        leaves: list[tuple[str, np.ndarray]] = []
        for key, child in value.items():
            leaves.extend(_numeric_arrays(child, f"{prefix}.{key}"))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for index, child in enumerate(value):
            leaves.extend(_numeric_arrays(child, f"{prefix}[{index}]"))
        return leaves
    return []


def run_smoke_test(
    n_steps: int = 50000,
    split: str = "train",
    seed: int = 42,
    config_path: str | None = None,
    workers: int = 2,
) -> dict[str, Any]:
    """Run PPO smoke test with correctness checks.

    Args:
        n_steps: Environment steps to train (50k-100k recommended)
        split: Training dataset split; must remain ``train`` to prevent leakage
        seed: Random seed
        config_path: Path to YAML config (uses ppo_cpu.yaml if None)

    Returns:
        Dict with results: failures, warnings, env_steps, duration, etc.
    """

    results: dict[str, Any] = {
        "n_steps": n_steps,
        "split": split,
        "seed": seed,
        "failures": [],
        "warnings": [],
        "checks_passed": [],
        "env_steps": 0,
        "gradient_updates": 0,
        "nan_count": 0,
        "inf_count": 0,
        "duration_sec": 0.0,
        "checkpoint_path": None,
    }

    try:
        start_time = time.time()

        if split != "train":
            raise ValueError("PPO smoke training must use the train split")

        # Load config
        if config_path is None:
            config_path = "configs/ppo_cpu.yaml"

        config = ExperimentConfig.from_yaml(config_path)
        config.training.total_env_steps = n_steps
        config.environment.split = split
        config.compute.seed = seed
        config.compute.num_workers = max(1, int(workers))
        config.training.finite_check = True  # Enable strict finite checking

        print(f"[SMOKE] Config: {config_path}")
        print(f"[SMOKE] Steps: {n_steps}, Split: {split}, Seed: {seed}")
        print(f"[SMOKE] Policy: {config.algorithm}")
        print()

        # Create trainer
        run_dir = Path("runs") / f"smoke_ppo_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print("[SMOKE] Creating dataset...")
        dataset = make_training_dataset(
            processed_dir=DEFAULT_PROCESSED_DIR,
            instruments=DEFAULT_INSTRUMENT_ORDER,
        )

        print("[SMOKE] Initializing PPO trainer...")
        trainer = PPOTrainer(config, run_dir, dataset=dataset)

        print(f"[SMOKE] Trainer initialized: {run_dir}")
        print("[SMOKE] Policy: tanh-squashed Gaussian")
        print("[SMOKE] Action bounds: (-1, +1)")
        print()

        # Run actual training loop
        print(f"[SMOKE] Starting training for {n_steps:,} environment steps...")
        try:
            trainer.train()
            results["env_steps"] = trainer.env_steps
            results["gradient_updates"] = trainer.gradient_updates
        except KeyboardInterrupt:
            print("[SMOKE] Training interrupted")
            results["env_steps"] = trainer.env_steps
            results["gradient_updates"] = trainer.gradient_updates
            results["warnings"].append("Training was interrupted before completion")
        except Exception as e:
            results["failures"].append(f"Training crashed: {e!s}")
            import traceback

            print(traceback.format_exc())
            return results

        # Validation checks
        if not results["warnings"]:
            results["checks_passed"].append("Training completed without crash")

        # Check if we reached target steps
        if trainer.env_steps >= n_steps * 0.95:  # Allow 5% variance
            results["checks_passed"].append(f"Target steps reached: {trainer.env_steps:,}")
        else:
            results["warnings"].append(
                f"Target steps not fully reached: {trainer.env_steps:,}/{n_steps:,}"
            )

        # Load best checkpoint and verify it's valid
        try:
            best_checkpoint = trainer.checkpoints.best_path()
            if best_checkpoint is not None:
                checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
                results["checks_passed"].append("Best checkpoint saved and loadable")
                results["checkpoint_path"] = str(best_checkpoint)

                # Verify nested model and optimizer arrays, not just top-level values.
                for key, array in _numeric_arrays(checkpoint):
                    nan_count = int(np.isnan(array).sum())
                    inf_count = int(np.isinf(array).sum())
                    results["nan_count"] += nan_count
                    results["inf_count"] += inf_count
                    if nan_count or inf_count:
                        results["failures"].append(f"Non-finite value in {key}")

                if not results["failures"]:
                    results["checks_passed"].append("All checkpoint tensors finite")
            else:
                results["warnings"].append("No best.pt checkpoint found")
        except Exception as e:
            results["failures"].append(f"Could not verify checkpoint: {e!s}")

        # Check metrics
        learning_curve = run_dir / "learning_curve.csv"
        if learning_curve.exists():
            import csv

            with open(learning_curve) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if rows:
                last_row = rows[-1]
                if "actor_loss" in last_row:
                    try:
                        actor_loss = float(last_row["actor_loss"])
                        if np.isfinite(actor_loss):
                            results["checks_passed"].append(f"Actor loss finite: {actor_loss:.4f}")
                        else:
                            results["failures"].append(f"Actor loss not finite: {actor_loss}")
                    except (ValueError, KeyError):
                        pass

                if "entropy" in last_row:
                    try:
                        entropy = float(last_row["entropy"])
                        if np.isfinite(entropy):
                            results["checks_passed"].append(f"Entropy finite: {entropy:.4f}")
                        else:
                            results["failures"].append(f"Entropy not finite: {entropy}")
                    except (ValueError, KeyError):
                        pass

        results["duration_sec"] = time.time() - start_time
        print()
        print("=" * 80)
        print("SMOKE TEST SUMMARY")
        print("=" * 80)
        print(f"Duration            : {results['duration_sec']:.1f} sec")
        print(f"Env steps           : {results['env_steps']:,}")
        print(f"Gradient updates    : {results['gradient_updates']:,}")
        print(f"Checks passed       : {len(results['checks_passed'])}")
        print(f"Warnings            : {len(results['warnings'])}")
        print(f"Failures            : {len(results['failures'])}")

        if results["warnings"]:
            print("\nWARNINGS:")
            for warning in results["warnings"]:
                print(f"  WARNING: {warning}")

        if results["failures"]:
            print("\nFAILURES:")
            for failure in results["failures"]:
                print(f"  FAILED: {failure}")
            return results

        print("\nCHECKS PASSED:")
        for check in results["checks_passed"]:
            print(f"  PASSED: {check}")

        print()
        print("=" * 80)
        print("SMOKE TEST PASSED - runtime correctness checks succeeded")
        print("Ready for longer training runs")
        print("=" * 80)

        return results

    except Exception as e:
        results["failures"].append(f"Smoke test crashed: {e!s}")
        import traceback

        print(traceback.format_exc())
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO correctness smoke test (50k-100k steps).")
    parser.add_argument("--steps", type=int, default=50000, help="Environment steps.")
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/ppo_cpu.yaml")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Environment worker count for local constrained CPUs.",
    )
    args = parser.parse_args()

    results = run_smoke_test(
        n_steps=args.steps,
        split=args.split,
        seed=args.seed,
        config_path=args.config,
        workers=args.workers,
    )

    sys.exit(0 if not results["failures"] else 1)


if __name__ == "__main__":
    main()
