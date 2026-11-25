"""Evaluate a frozen training checkpoint (validation or test) and optionally
produce the final benchmark tables (SAC vs baselines).

Usage::

    python -m forexmind.training.evaluate_checkpoint --checkpoint runs/.../checkpoints/best.pt
    python -m forexmind.training.evaluate_checkpoint --checkpoint .../best.pt \
        --split test --episodes 100
    python -m forexmind.training.evaluate_checkpoint --checkpoint .../best.pt \
        --benchmark --out data/reports/benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.benchmark import (
    benchmark_test_split,
    load_checkpoint_policy,
    write_benchmark_results,
)
from forexmind.training.checkpoint import resolve_checkpoint
from forexmind.training.config import ExperimentConfig
from forexmind.training.data import (
    DEFAULT_INSTRUMENT_ORDER,
    DEFAULT_PROCESSED_DIR,
)
from forexmind.training.dataset_mmap import resolve_dataset
from forexmind.training.evaluator import (
    DEFAULT_SELECTION_METRIC,
    VALID_SELECTION_METRICS,
    PolicyEvaluator,
)
from forexmind.training.trainer import build_env_config


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    """Locate the checkpoint to evaluate, defaulting the search root to the
    run root embedded in the config / CLI (``--run-root``)."""
    run_root = args.run_root
    if run_root is None and not args.config:
        # Try to infer from a known config file so the search is meaningful.
        for cand in ("configs/sac_cpu.yaml", "configs/ppo_cpu.yaml"):
            if Path(cand).is_file():
                try:
                    run_root = ExperimentConfig.from_yaml(cand).logging.run_dir
                    break
                except Exception:
                    pass
    return resolve_checkpoint(args.checkpoint, run_root=run_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen training checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint path, a run dir, or a run name "
        "(e.g. 'runs/sac_cpu_seed42/checkpoints/best.pt', "
        "'runs/sac_cpu_seed42', or 'sac_cpu_seed42').",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help="Root directory to search for run checkpoints "
        "(default: from the embedded config, else 'runs').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Which split to evaluate on.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config (fallback: embedded in checkpoint).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Also run the final SAC-vs-baselines benchmark on test.",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output directory for benchmark tables."
    )
    args = parser.parse_args()

    checkpoint = _resolve_checkpoint(args)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    algorithm = state.get("algorithm", "sac")
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        cfg_dict = state.get("config") or {}
        config = (
            ExperimentConfig.from_dict(cfg_dict)
            if isinstance(cfg_dict, dict)
            else ExperimentConfig()
        )

    instruments = (
        tuple(config.environment.instruments)
        if config.environment.instruments
        else DEFAULT_INSTRUMENT_ORDER
    )
    dataset, _backend = resolve_dataset(
        processed_dir=DEFAULT_PROCESSED_DIR, instruments=instruments, backend="auto"
    )
    env_config = build_env_config(config.environment)
    encoder = ObservationEncoder(
        EncoderConfig(
            context_length=config.environment.context_length,
            initial_balance=env_config.margin.initial_balance,
        )
    )
    window_config = WindowConfig(context_length=config.environment.context_length)
    policy, algorithm = load_checkpoint_policy(
        checkpoint, encoder.config.spec.encoded_shape[0], config.model
    )
    print(f"Loaded checkpoint: {checkpoint}")

    checkpoint_metric = state.get("selection_metric_name")
    selection_metric = (
        str(checkpoint_metric)
        if checkpoint_metric in VALID_SELECTION_METRICS
        else DEFAULT_SELECTION_METRIC
    )
    evaluator = PolicyEvaluator(
        dataset,
        env_config,
        encoder,
        window_config,
        selection_metric=selection_metric,
        lambda_drawdown=config.selection.lambda_drawdown,
        eval_horizon=config.evaluation.eval_horizon,
        eval_seed=args.seed if args.seed is not None else config.evaluation.eval_seed,
        context_length=config.environment.context_length,
    )
    result = evaluator.evaluate_sampled(policy, algorithm, args.split, args.episodes)
    print(f"\n=== {algorithm.upper()} frozen policy on {args.split} ({args.episodes} episodes) ===")
    for k, v in sorted(result.items()):
        if not isinstance(v, (dict, list)):
            print(f"  {k:<22} {v}")
    print("  Portfolio Sharpe/Sortino unavailable: independently reset sampled episodes.")
    out = Path(args.out) if args.out else checkpoint.resolve().parent / "evaluation" / args.split
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "sampled_evaluation.json"
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"  Per-episode distributions and diagnostics: {report_path}")

    if args.benchmark:
        print(
            "Benchmark reports independent episode distributions. The old step-aligned "
            "curve is retained only as a named diagnostic. Use tools.audit_ppo_evaluation "
            "for matched enter-and-HOLD comparisons."
        )
        bench = benchmark_test_split(
            dataset=dataset,
            env_config=env_config,
            encoder=encoder,
            window_config=window_config,
            policy=policy,
            algorithm=algorithm,
            split="test",
            n_episodes=args.episodes,
            horizon=config.evaluation.eval_horizon,
            seed=args.seed if args.seed is not None else config.evaluation.eval_seed,
        )
        out = Path(args.out) if args.out else checkpoint.resolve().parent.parent / "benchmark_test"
        paths = write_benchmark_results(bench, out)
        print(f"\nBenchmark tables written to {out}:")
        for name, p in paths.items():
            print(f"  {name:<6} {p}")


if __name__ == "__main__":
    main()
