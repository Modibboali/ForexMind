"""Metrics store, learning-curve logging, and progress reporting (Phase 3).

Track both ``environment_steps`` and ``gradient_updates`` as separate
quantities.  Learning curves are written as CSV/JSONL (no plotting library
required).  Progress blocks (§13) make training readable without TensorBoard.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


class MetricStore:
    """Append-only metric records keyed by env steps / gradient updates."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def record(self, *, env_steps: int, gradient_updates: int, **values: Any) -> None:
        row: dict[str, Any] = {
            "env_steps": env_steps,
            "gradient_updates": gradient_updates,
            **values,
        }
        self._records.append(row)

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def series(self, name: str) -> list[float]:
        return [float(r[name]) for r in self._records if name in r]

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._records:
            path.write_text("env_steps,gradient_updates\n", encoding="utf-8")
            return
        # Records may carry different keys (e.g. validation rows add val_*),
        # so write a union of all fields instead of DictWriter's fixed names.
        fields: list[str] = []
        seen: set[str] = set()
        for row in self._records:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    fields.append(k)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(fields)
            for row in self._records:
                writer.writerow([row.get(k, "") for k in fields])

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in self._records:
                fh.write(json.dumps(row, default=str) + "\n")


class TrainerLogger:
    """Prints human-readable progress blocks at evaluation intervals."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def progress_block(self, header: str, fields: dict[str, Any]) -> None:
        if not self.verbose:
            return
        width = 60
        print("=" * width)
        print(header)
        print("=" * width)
        for k, v in fields.items():
            if isinstance(v, float):
                print(f"  {k:<28} {v:+.6f}")
            else:
                print(f"  {k:<28} {v}")
        print("=" * width)


def summarize_returns(returns: list[float]) -> dict[str, float]:
    """Summary stats for a list of episode returns."""
    if not returns:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def action_distribution(actions: np.ndarray) -> dict[str, float]:
    """Fraction of short/flat/long actions and mean |exposure|."""
    if len(actions) == 0:
        return {"frac_short": 0.0, "frac_flat": 0.0, "frac_long": 0.0,
                "mean_abs_exposure": 0.0, "mean_action": 0.0}
    a = np.asarray(actions, dtype=np.float64)
    return {
        "frac_short": float(np.mean(a < -0.05)),
        "frac_flat": float(np.mean(np.abs(a) <= 0.05)),
        "frac_long": float(np.mean(a > 0.05)),
        "mean_abs_exposure": float(np.mean(np.abs(a))),
        "mean_action": float(np.mean(a)),
    }


def pathological_warnings(metrics: dict[str, Any]) -> list[str]:
    """Detect pathological policy behaviors (§23); returns warning strings."""
    warnings: list[str] = []
    ad = metrics.get("action_distribution", {})
    if ad.get("frac_long", 0.0) > 0.99:
        warnings.append("policy is almost always LONG (check for degenerate behavior)")
    if ad.get("frac_short", 0.0) > 0.99:
        warnings.append("policy is almost always SHORT (check for degenerate behavior)")
    if ad.get("frac_flat", 0.0) > 0.99:
        warnings.append("policy is almost always FLAT (likely not learning)")
    if metrics.get("turnover", 0.0) is not None and metrics.get("turnover", 0.0) > 100.0:
        warnings.append("extreme turnover detected")
    if metrics.get("max_drawdown_pct", 0.0) > 0.5:
        warnings.append("near-total drawdown detected")
    if np.isnan(metrics.get("mean_reward", 0.0)) or np.isinf(metrics.get("mean_reward", 0.0)):
        warnings.append("NaN/Inf rewards detected")
    if metrics.get("critic_loss", 0.0) is not None and metrics.get("critic_loss", 0.0) > 1e6:
        warnings.append("exploding Q values detected")
    return warnings
