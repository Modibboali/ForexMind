"""Rank PPO checkpoints on one fixed episode set and optionally validate chronologically.

Example:
    python -m tools.evaluate_checkpoint_curve --checkpoint-dir runs/my_run/checkpoints
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from forexmind.evaluation.chronological import ChronologicalEvaluator
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.benchmark import load_checkpoint_policy
from forexmind.training.checkpoint import discover_checkpoints
from forexmind.training.config import ExperimentConfig
from forexmind.training.data import DEFAULT_INSTRUMENT_ORDER, DEFAULT_PROCESSED_DIR
from forexmind.training.dataset_mmap import resolve_dataset
from forexmind.training.evaluator import DEFAULT_SELECTION_METRIC, PolicyEvaluator
from forexmind.training.trainer import build_env_config


def rank_checkpoint_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank only by valid mean episode log return, descending."""
    if not rows:
        raise ValueError("no checkpoint results to rank")
    if any(
        not isinstance(row.get("mean_episode_log_return"), (int, float))
        or not math.isfinite(float(row["mean_episode_log_return"]))
        for row in rows
    ):
        raise ValueError("all checkpoints need a finite mean_episode_log_return")
    return sorted(rows, key=lambda row: row["mean_episode_log_return"], reverse=True)


def checkpoint_curve_row(
    checkpoint: Path, state: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    rows = report["episodes"]
    non_jpy = [row["total_return"] for row in rows if row["instrument"] != "USDJPY"]
    positive_profit = sum(max(row["total_return"], 0.0) for row in rows)
    jpy_positive_profit = sum(
        max(row["total_return"], 0.0)
        for row in rows
        if row["instrument"] == "USDJPY"
    )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "step": int(state.get("env_steps", 0)),
        "gradient_updates": int(state.get("gradient_updates", 0)),
        "mean_episode_log_return": report["mean_episode_log_return"],
        "mean_episode_return": report["mean_episode_return"],
        "median_episode_return": report["median_episode_return"],
        "profitable_episode_fraction": report["profitable_episode_fraction"],
        "p10_return": report["p10_episode_return"],
        "p90_return": report["p90_episode_return"],
        "mean_turnover": report["mean_turnover_per_episode"],
        "mean_executions": report["mean_executions_per_episode"],
        "pct_HOLD": report["actions"]["pct_hold"],
        "pct_FLAT": report["actions"]["pct_flat"],
        "pct_LONG": report["actions"]["pct_long"],
        "pct_SHORT": report["actions"]["pct_short"],
        "mean_episode_return_excluding_usdjpy": (
            sum(non_jpy) / len(non_jpy) if non_jpy else None
        ),
        "usdjpy_contribution_to_positive_episode_profit": (
            jpy_positive_profit / positive_profit if positive_profit else None
        ),
        "overlapping_episode_pairs": report["overlap_diagnostics"][
            "overlapping_episode_pairs"
        ],
        "overlap_fraction": report["overlap_diagnostics"]["overlap_fraction"],
        "largest_overlap_duration_steps": report["overlap_diagnostics"][
            "largest_overlap_duration_steps"
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value) for value in args.checkpoints]
    if args.checkpoint_dir:
        paths.extend(discover_checkpoints(args.checkpoint_dir))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen and path.is_file():
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise FileNotFoundError("no available .pt checkpoints were supplied or discovered")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/reports/stage35_checkpoint_curve")
    parser.add_argument("--chronological-best", action="store_true")
    args = parser.parse_args()
    checkpoints = _checkpoint_paths(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    first_state = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    first_config = ExperimentConfig.from_dict(first_state.get("config") or {})
    instruments = tuple(first_config.environment.instruments) or DEFAULT_INSTRUMENT_ORDER
    dataset, backend = resolve_dataset(
        processed_dir=DEFAULT_PROCESSED_DIR,
        instruments=instruments,
        backend=first_config.compute.dataset_backend,
    )
    env_config = build_env_config(first_config.environment)
    encoder = ObservationEncoder(
        EncoderConfig(
            context_length=first_config.environment.context_length,
            initial_balance=env_config.margin.initial_balance,
        )
    )
    window = WindowConfig(context_length=first_config.environment.context_length)
    evaluator = PolicyEvaluator(
        dataset,
        env_config,
        encoder,
        window,
        selection_metric=DEFAULT_SELECTION_METRIC,
        eval_horizon=first_config.evaluation.eval_horizon,
        eval_seed=args.seed,
    )
    specs = evaluator.selection_episode_specs(args.split, args.episodes, args.seed)
    spec_payload = {
        "evaluation_type": "sampled_independent_episodes",
        "is_portfolio_path": False,
        "selection_metric_name": DEFAULT_SELECTION_METRIC,
        "dataset_backend": backend,
        "seed": args.seed,
        "episode_count": args.episodes,
        "non_overlapping_observation_windows": True,
        "episode_specs": [spec.to_dict() for spec in specs],
    }
    (out / "validation_episode_specs.json").write_text(
        json.dumps(spec_payload, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    full_reports: dict[str, Any] = {}
    policy_evaluations = []
    reference_config = first_config.to_dict()
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = ExperimentConfig.from_dict(state.get("config") or {})
        if config.model != first_config.model or config.environment != first_config.environment:
            raise ValueError("all checkpoints must share model and environment configuration")
        if state.get("algorithm") != "ppo" or state.get("action_policy") != "categorical_v1":
            raise ValueError(f"{checkpoint} is not a categorical PPO checkpoint")
        policy, algorithm = load_checkpoint_policy(
            checkpoint, encoder.config.spec.encoded_shape[0], config.model
        )
        evaluation = evaluator.evaluate(
            policy,
            algorithm,
            args.split,
            args.episodes,
            episode_specs=specs,
        )
        row = checkpoint_curve_row(checkpoint, state, evaluation.metrics)
        rows.append(row)
        full_reports[str(checkpoint)] = {
            "summary": row,
            "per_instrument": evaluation.metrics["per_instrument"],
            "overlap_diagnostics": evaluation.metrics["overlap_diagnostics"],
        }
        policy_evaluations.append((policy, evaluation))
        print(
            f"step={row['step']} checkpoint={checkpoint} "
            f"mean_episode_log_return={row['mean_episode_log_return']:+.8f}",
            flush=True,
        )

    ranked = rank_checkpoint_rows(rows)
    best_path = Path(ranked[0]["checkpoint"])
    best_index = next(i for i, path in enumerate(checkpoints) if path.resolve() == best_path)
    best_policy, best_evaluation = policy_evaluations[best_index]
    matched = evaluator.matched_baseline_report(best_evaluation, specs)
    result = {
        "selection_metric_name": DEFAULT_SELECTION_METRIC,
        "configured_checkpoint_metadata": reference_config,
        "ranked_checkpoints": ranked,
        "corrected_best_checkpoint": ranked[0],
        "checkpoint_reports": full_reports,
        "matched_baselines_for_corrected_best": matched,
    }
    (out / "checkpoint_curve.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_csv(out / "checkpoint_curve.csv", ranked)
    (out / "corrected_best.json").write_text(
        json.dumps(ranked[0], indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    if args.chronological_best:
        chronological = ChronologicalEvaluator(dataset, env_config, encoder, window).evaluate(
            best_policy,
            "ppo",
            split=args.split,
            instruments=list(instruments),
            seed=args.seed,
        )
        (out / "chronological_validation.json").write_text(
            json.dumps(chronological, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
