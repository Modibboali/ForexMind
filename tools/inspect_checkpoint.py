"""Inspect a training checkpoint for numerical health (PPO NaN audit).

Run this on Kaggle against the rescue checkpoint to determine whether it is
safe to resume.  It scans every tensor in the checkpoint (policy, critics,
optimizers) for NaN/Inf and reports finite status + max |value|.

Examples:
    python -m tools.inspect_checkpoint runs/ppo_cpu_seed42/checkpoints/rescue_step_376832.pt
    python -m tools.inspect_checkpoint runs/ppo_cpu_seed42        # newest checkpoint
    python -m tools.inspect_checkpoint best.pt --run-root runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from forexmind.training.checkpoint import resolve_checkpoint


def _scan(name: str, value: Any, out: list[dict[str, object]]) -> None:
    """Recursively scan a checkpoint subtree for tensors and their finiteness."""
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        arr = None
    if arr is not None and arr.dtype.kind in "fc":
        total = int(arr.size)
        finite = np.isfinite(arr)
        n_finite = int(np.count_nonzero(finite))
        n_nan = int(np.count_nonzero(np.isnan(arr)))
        n_inf = total - n_finite - n_nan
        max_abs = float(np.max(np.abs(arr[finite]))) if n_finite else float("nan")
        if n_finite < total:
            out.append(
                {
                    "name": name,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "finite_count": n_finite,
                    "nan_count": n_nan,
                    "inf_count": n_inf,
                    "max_abs": max_abs,
                }
            )
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _scan(f"{name}.{k}", v, out)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _scan(f"{name}[{i}]", v, out)


def inspect(path: str | Path) -> dict[str, object]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    bad: list[dict[str, object]] = []
    for key in ("policy", "critics", "targets", "optimizers", "log_alpha"):
        if state.get(key):
            _scan(key, state[key], bad)
    cfg = state.get("config") or {}
    return {
        "checkpoint": str(path),
        "algorithm": state.get("algorithm"),
        "env_steps": state.get("env_steps"),
        "gradient_updates": state.get("gradient_updates"),
        "episodes": state.get("episodes"),
        "dataset_version": state.get("dataset_version"),
        "non_finite_tensors": bad,
        "n_non_finite_tensors": len(bad),
        "config": cfg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a checkpoint for NaN/Inf.")
    parser.add_argument("checkpoint", type=str, help="Checkpoint path, run dir, or run name.")
    parser.add_argument("--run-root", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    path = resolve_checkpoint(args.checkpoint, run_root=args.run_root)
    print(f"Resolved checkpoint: {path}")
    report = inspect(path)
    print(f"Algorithm          : {report['algorithm']}")
    print(f"Env steps          : {report['env_steps']:,}")
    print(f"Gradient updates   : {report['gradient_updates']:,}")
    print(f"Episodes           : {report['episodes']:,}")
    print(f"Dataset version    : {report['dataset_version']}")
    print(f"Non-finite tensors : {report['n_non_finite_tensors']}")
    non_finite = cast(list[dict[str, object]], report["non_finite_tensors"])
    for bad in non_finite:
        print(
            f"  !! {bad['name']} shape={bad['shape']} "
            f"finite={bad['finite_count']} nan={bad['nan_count']} "
            f"inf={bad['inf_count']} max_abs={bad['max_abs']}"
        )
    if report["n_non_finite_tensors"]:
        print(
            "\nWARNING: checkpoint contains NaN/Inf.  Resuming from it is "
            "expected to immediately re-produce divergence (the rescue was "
            "saved at the failing step).  Do NOT resume with the poisoned "
            "weights; retrain from the last clean checkpoint instead."
        )
    else:
        print("\nCheckpoint tensors are all finite -> safe to resume.")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nReport saved to {args.json}")


if __name__ == "__main__":
    main()
