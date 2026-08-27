"""Phase 3 training benchmark tool.

Two modes:

1. Worker-throughput sweep (default):
   ``python -m tools.benchmark_training --workers 1 4 8 16``
   measures environment steps/second of the *collection* layer for each worker
   count (process backend) — useful for sizing a dedicated compute machine.

2. Final benchmark tables for a frozen checkpoint:
   ``python -m tools.benchmark_training --checkpoint runs/.../checkpoints/best.pt --episodes 100``
   evaluates the trained policy vs the seven baselines on the untouched test
   split and writes JSON/CSV/text tables.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SplitDataset
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.config import ExperimentConfig, ModelConfig
from forexmind.training.data import (
    DEFAULT_INSTRUMENT_ORDER,
    DEFAULT_PROCESSED_DIR,
    make_training_dataset,
)
from forexmind.training.trainer import build_env_config


def _build_components(
    config: ExperimentConfig,
) -> tuple[SplitDataset, EnvironmentConfig, ObservationEncoder, WindowConfig]:
    instruments = (
        tuple(config.environment.instruments)
        if config.environment.instruments
        else DEFAULT_INSTRUMENT_ORDER
    )
    dataset = make_training_dataset(DEFAULT_PROCESSED_DIR, None, instruments)
    env_config = build_env_config(config.environment)
    encoder = ObservationEncoder(
        EncoderConfig(
            context_length=config.environment.context_length,
            initial_balance=env_config.margin.initial_balance,
        )
    )
    window_config = WindowConfig(context_length=config.environment.context_length)
    return dataset, env_config, encoder, window_config


def bench_worker_throughput(
    worker_counts: list[int], *, n_steps: int = 200_000, collect_batch: int = 2048
) -> list[dict[str, float]]:
    from forexmind.episodes.config import EpisodeConfig
    from forexmind.training.collector import ProcessCollector
    from forexmind.training.config import ExperimentConfig
    from forexmind.training.policies import build_policy_network

    cfg = ExperimentConfig.smoke("sac")
    dataset, env_config, encoder, window_config = _build_components(cfg)
    obs_dim = encoder.config.spec.encoded_shape[0]
    model = ModelConfig(hidden_dim=16, num_layers=2)
    episode_config = EpisodeConfig(
        split=cfg.environment.split,
        horizon=cfg.environment.horizon,
        context_length=cfg.environment.context_length,
        seed=cfg.compute.seed,
    )

    rows: list[dict[str, float]] = []
    for workers in worker_counts:
        collector = ProcessCollector(
            processed_dir=str(DEFAULT_PROCESSED_DIR),
            split_config=dataset.split_config,
            instruments=dataset.instruments,
            env_config=env_config,
            encoder_config=encoder.config,
            window_config=window_config,
            episode_config=episode_config,
            algorithm="sac",
            model=model,
            obs_dim=obs_dim,
            action_dim=1,
            global_seed=cfg.compute.seed,
            num_workers=workers,
        )
        policy = build_policy_network("sac", obs_dim, 1, model)
        collector.set_policy(policy)
        t0 = time.perf_counter()
        steps = 0
        episodes = 0
        while steps < n_steps:
            batch = collector.collect(collect_batch, random_action=False)
            steps += len(batch)
            episodes += sum(1 for t in batch if t.terminated or t.truncated)
        dt = time.perf_counter() - t0
        collector.close()
        rows.append(
            {
                "workers": float(workers),
                "env_steps": float(steps),
                "wall_seconds": round(dt, 3),
                "steps_per_second": round(steps / dt, 1) if dt > 0 else 0.0,
                "episodes": float(episodes),
            }
        )
        print(
            f"workers={workers:>3}  {steps:>10,} steps in {dt:7.1f}s  "
            f"-> {steps / dt:9.1f} steps/s"
        )
    return rows


def run_final_benchmark(checkpoint: str, episodes: int, out: str | None) -> None:
    import torch
    from forexmind.training.benchmark import (
        benchmark_test_split,
        load_checkpoint_policy,
        write_benchmark_results,
    )
    from forexmind.training.evaluator import PolicyEvaluator

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    algorithm = state.get("algorithm", "sac")
    cfg_dict = state.get("config") or {}
    config = (
        ExperimentConfig.from_dict(cfg_dict)
        if isinstance(cfg_dict, dict)
        else ExperimentConfig()
    )
    dataset, env_config, encoder, window_config = _build_components(config)
    policy, algorithm = load_checkpoint_policy(
        checkpoint, encoder.config.spec.encoded_shape[0], config.model
    )

    # Validation smoke first (freeze best checkpoint already chosen at train time).
    evaluator = PolicyEvaluator(
        dataset, env_config, encoder, window_config,
        selection_metric=config.selection.metric,
        lambda_drawdown=config.selection.lambda_drawdown,
        eval_horizon=config.evaluation.eval_horizon,
        eval_seed=config.evaluation.eval_seed,
        context_length=config.environment.context_length,
    )
    val = evaluator.evaluate(policy, algorithm, "validation", episodes)
    print(f"\nFrozen policy on validation: score={val.score:.4f} "
          f"sharpe={val.metrics.get('sharpe', 0.0):.4f}")

    bench = benchmark_test_split(
        dataset=dataset,
        env_config=env_config,
        encoder=encoder,
        window_config=window_config,
        policy=policy,
        algorithm=algorithm,
        split="test",
        n_episodes=episodes,
        horizon=config.evaluation.eval_horizon,
        seed=config.evaluation.eval_seed,
    )
    out_dir = Path(out) if out else Path(checkpoint).resolve().parent.parent / "benchmark_test"
    paths = write_benchmark_results(bench, out_dir)
    print(f"\nBenchmark tables written to {out_dir}:")
    for name, p in paths.items():
        print(f"  {name:<6} {p}")
    print("\nAggregate metrics:")
    for r in bench["results"]:
        m = r["metrics"]
        print(
            f"  {r['agent']:<16} ret={m.get('total_return', 0):+.4f}  "
            f"sharpe={m.get('sharpe', 0):+.4f}  "
            f"sortino={m.get('sortino', 0):+.4f}  "
            f"mdd={m.get('max_drawdown_pct', 0):.4f}  "
            f"turnover={m.get('turnover', 0):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="ForexMind Phase 3 training benchmark.")
    parser.add_argument("--workers", type=int, nargs="+", default=None,
                        help="Worker counts to sweep (e.g. 1 4 8 16).")
    parser.add_argument("--n-steps", type=int, default=200_000,
                        help="Env steps to collect per worker count.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Frozen checkpoint for the final benchmark tables.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--json", type=str, default=None,
                        help="Optional JSON path for the throughput sweep.")
    args = parser.parse_args()

    if args.checkpoint:
        run_final_benchmark(args.checkpoint, args.episodes, args.out)
        return

    workers = args.workers or [1, 2, 4]
    rows = bench_worker_throughput(workers, n_steps=args.n_steps)
    if args.json:
        Path(args.json).write_text(
            json.dumps(rows, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nThroughput sweep saved to {args.json}")


if __name__ == "__main__":
    main()
