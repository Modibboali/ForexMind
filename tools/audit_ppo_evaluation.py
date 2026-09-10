"""Reproduce legacy metrics and evaluate matched independent episodes, without training.

python -m tools.audit_ppo_evaluation --checkpoint forexmind/best.pt --episodes 100 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from forexmind.baselines.base import TradingAgent
from forexmind.environment.actions import ACTION_NAMES
from forexmind.episodes.sampler import EpisodeSpec
from forexmind.episodes.trajectory import Trajectory
from forexmind.evaluation.runner import EvaluationRunner
from forexmind.evaluation.sampled import EnterAndHoldAgent, paired_comparison, sampled_report
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.benchmark import load_checkpoint_policy
from forexmind.training.config import ExperimentConfig
from forexmind.training.data import DEFAULT_INSTRUMENT_ORDER, DEFAULT_PROCESSED_DIR
from forexmind.training.dataset_mmap import resolve_dataset
from forexmind.training.evaluator import PolicyEvaluator
from forexmind.training.policies import PolicyAgent
from forexmind.training.trainer import build_env_config


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8"
    )


def legacy_diagnostic_score(
    metrics: dict[str, Any], metric: str, lambda_drawdown: float
) -> float | None:
    """Reproduce old checkpoint arithmetic without making it selectable."""
    if metric == "sharpe_drawdown":
        return float(metrics.get("sharpe", 0.0)) - lambda_drawdown * float(
            metrics.get("max_drawdown_pct", 0.0)
        )
    if metric in {"sharpe", "total_return"}:
        return float(metrics.get(metric, 0.0))
    return None


def raw_json_value(value: Any) -> Any:
    """Preserve infinite legacy derived ratios explicitly as strings in raw archives.

    E.g. profit_factor is +inf if closed trades have profits but no losses.
    This is not a non-finite equity/reward; corrected metrics are recomputed.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: raw_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [raw_json_value(v) for v in value]
    return value


def read_raw(path: Path) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    if not path.exists():
        return trajectories
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            data["spec"] = EpisodeSpec(**data["spec"])
            data["timestamps"] = np.asarray(data["timestamps"], dtype="datetime64[ns]")
            for key in ("actions", "rewards", "equity", "log_returns", "position_units"):
                data[key] = np.asarray(data[key])
            trajectories.append(Trajectory(**data))
    return trajectories


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/ppo_stable_seed1/checkpoints/best.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--out", default="data/reports/ppo_evaluation_audit")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed raw episodes after verifying metadata and specs",
    )
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ExperimentConfig.from_dict(state["config"])
    torch.set_num_threads(config.compute.torch_threads)
    if config.compute.torch_interop_threads is not None:
        torch.set_num_interop_threads(config.compute.torch_interop_threads)
    instruments = tuple(config.environment.instruments) or DEFAULT_INSTRUMENT_ORDER
    dataset, backend = resolve_dataset(
        processed_dir=DEFAULT_PROCESSED_DIR, instruments=instruments, backend="auto"
    )
    env_config = build_env_config(config.environment)
    encoder = ObservationEncoder(
        EncoderConfig(
            context_length=config.environment.context_length,
            initial_balance=env_config.margin.initial_balance,
        )
    )
    window = WindowConfig(context_length=config.environment.context_length)
    policy, algorithm = load_checkpoint_policy(
        checkpoint, encoder.config.spec.encoded_shape[0], config.model
    )
    if algorithm != "ppo":
        raise ValueError("This audit requires a categorical PPO checkpoint")
    evaluator = PolicyEvaluator(
        dataset,
        env_config,
        encoder,
        window,
        eval_horizon=config.evaluation.eval_horizon,
        eval_seed=args.seed,
    )
    specs = evaluator._episode_specs(args.split, args.episodes, args.seed)
    runner = EvaluationRunner(dataset, env_config, encoder, window, capture_account_state=True)
    ppy = runner.periods_per_year(args.split)
    with checkpoint.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    split_start, split_end = dataset.split_config.range(args.split)
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "checkpoint_env_steps": state.get("env_steps"),
        "checkpoint_saved_legacy_selection_score": state.get("best_validation_score"),
        "config": state["config"],
        "split": args.split,
        "seed": args.seed,
        "dataset_backend": backend,
        "periods_per_year": ppy,
        "split_start": str(split_start),
        "split_end": str(split_end),
        "split_m5_rows": {i: dataset.split(i, args.split).n_bars for i in instruments},
        "episode_specs": [s.to_dict() for s in specs],
    }
    if (
        args.resume
        and (out / "metadata.json").exists()
        and json.loads((out / "metadata.json").read_text(encoding="utf-8")) != metadata
    ):
        raise ValueError("Cannot resume: checkpoint, dataset metadata, or episode specs differ")
    write_json(out / "metadata.json", metadata)
    agents: list[TradingAgent] = [
        PolicyAgent(policy, algorithm, name="PPO"),
        *[EnterAndHoldAgent(i) for i in (0, 6, 7, 8, 9, 5, 4, 3, 2)],
    ]
    reports: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for agent in agents:
        start = time.perf_counter()
        raw_path = out / f"{agent.name}_raw.jsonl.gz"
        trajectories = read_raw(raw_path) if args.resume else []
        if len(trajectories) > len(specs) or any(
            t.spec != specs[i] or t.agent_name != agent.name for i, t in enumerate(trajectories)
        ):
            raise ValueError("Cannot resume: raw trajectory specifications differ")
        completed = len(trajectories)
        if completed:
            print(f"{agent.name}: reusing {completed} verified episode records", flush=True)
        with gzip.open(raw_path, "at" if args.resume else "wt", encoding="utf-8") as raw:
            for i in range(completed, len(specs)):
                spec = specs[i]
                trajectory = runner.run_episode(agent, spec)
                trajectories.append(trajectory)
                raw.write(
                    json.dumps(raw_json_value(trajectory.to_dict()), allow_nan=False, default=str)
                    + "\n"
                )
                if (i + 1) % 5 == 0 or i + 1 == len(specs):
                    print(
                        f"{agent.name}: {i + 1}/{len(specs)} episodes, "
                        f"{time.perf_counter() - start:.1f}s",
                        flush=True,
                    )
        report = sampled_report(trajectories, ppy)
        report["wall_seconds"] = time.perf_counter() - start
        if not report["invariants"] or not all(report["invariants"].values()):
            raise AssertionError(
                f"{agent.name}: accounting invariant failure {report['invariants']}"
            )
        write_json(out / f"{agent.name}.json", report)
        reports[agent.name] = report
        all_rows.extend(
            {k: v for k, v in row.items() if not isinstance(v, dict)} for row in report["episodes"]
        )
        if agent.name != "PPO":
            comparisons[agent.name] = paired_comparison(reports["PPO"], report)
            write_json(out / "paired_comparisons.json", comparisons)
        print(
            f"{agent.name}: mean return={report['mean_episode_return']:.8f}, "
            f"mean turnover={report['mean_turnover_per_episode']:.6f}",
            flush=True,
        )
    write_csv(out / "episodes.csv", all_rows)
    write_csv(
        out / "paired_differences.csv",
        [
            dict(baseline=name, **pair)
            for name, comparison in comparisons.items()
            for pair in comparison["pairs"]
        ],
    )
    ppo = reports["PPO"]
    diagnostic = ppo["cross_episode_mean_return_series"]
    legacy = {
        k.removeprefix("diagnostic_"): v
        for k, v in diagnostic.items()
        if k.startswith("diagnostic_")
    }
    legacy["turnover"] = ppo["total_turnover_all_episodes"]
    legacy["actual_executions"] = ppo["actions"]["actual_executions"]
    legacy["legacy_diagnostic_selection_score"] = legacy_diagnostic_score(
        legacy, config.selection.metric, config.selection.lambda_drawdown
    )
    comparison_rows = []
    for metric in (
        "n_periods",
        "total_return",
        "mean_episode_return",
        "median_episode_return",
        "profitable_episode_fraction",
        "sharpe",
        "sortino",
        "turnover",
        "total_turnover_all_episodes",
        "mean_turnover_per_episode",
        "actual_executions",
    ):
        comparison_rows.append(
            {
                "metric": metric,
                "old": legacy.get(metric),
                "corrected": ppo["actions"][metric]
                if metric == "actual_executions"
                else ppo.get(metric),
            }
        )
    write_csv(out / "before_after.csv", comparison_rows)
    write_json(out / "legacy_reproduction.json", legacy)
    write_csv(
        out / "paired_summary.csv",
        [
            {
                "baseline": name,
                "mean_baseline_return": reports[name]["mean_episode_return"],
                "mean_paired_advantage": c["paired_advantage"]["mean"],
                "median_paired_advantage": c["paired_advantage"]["median"],
                "fraction_ppo_beats_baseline": c["fraction_ppo_beats_baseline"],
            }
            for name, c in comparisons.items()
        ],
    )
    write_csv(
        out / "per_instrument.csv",
        [
            {
                "instrument": name,
                **{
                    k: stats[k]
                    for k in (
                        "episode_count",
                        "mean_episode_return",
                        "median_episode_return",
                        "profitable_episode_fraction",
                        "mean_turnover_per_episode",
                        "mean_executions_per_episode",
                    )
                },
            }
            for name, stats in ppo["per_instrument"].items()
        ],
    )
    write_csv(
        out / "action_distribution.csv",
        [
            {
                "action": name,
                "count": ppo["actions"][f"count_{name}"],
                "pct": ppo["actions"][f"pct_{name.lower()}"],
            }
            for name in ACTION_NAMES
        ],
    )
    print(
        json.dumps(
            {
                "before_after": comparison_rows,
                "sharpe_inputs": {
                    k: diagnostic[k]
                    for k in (
                        "return_observation_count",
                        "mean_return",
                        "return_std_ddof1",
                        "sqrt_periods_per_year",
                        "diagnostic_sharpe",
                    )
                },
                "periods_per_year": ppy,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
