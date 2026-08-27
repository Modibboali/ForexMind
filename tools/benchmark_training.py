"""Training-path benchmark and final checkpoint evaluation tool.

Kaggle throughput sweep:
    python -m tools.benchmark_training --workers 1,2,4,8,16,32,64,128,192,204

The sweep can measure:
    A_env_only       workers -> environment
    B_collection    workers -> environment -> IPC -> replay
    C_full_sac      workers -> environment -> IPC -> replay -> SAC learner

Final benchmark tables for a frozen checkpoint are still supported:
    python -m tools.benchmark_training --checkpoint runs/.../checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SplitDataset
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.collector import ProcessCollector, Transition
from forexmind.training.config import ExperimentConfig, default_config
from forexmind.training.data import (
    DEFAULT_INSTRUMENT_ORDER,
    DEFAULT_PROCESSED_DIR,
    make_training_dataset,
)
from forexmind.training.replay import ReplayBuffer
from forexmind.training.runtime_diagnostics import (
    ProcessTreeCpuSampler,
    logical_cpu_count,
    print_process_tree_report,
)
from forexmind.training.trainer import build_env_config

from tools.benchmark_env_workers import parse_worker_counts, run_env_only_sweep


def _load_benchmark_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(args.config) if args.config else default_config("sac")
    cfg = replace(cfg, algorithm=replace(cfg.algorithm, name="sac"))
    cfg = replace(
        cfg,
        compute=replace(cfg.compute, collect_backend="process", torch_threads=args.torch_threads),
        training=replace(cfg.training, collect_batch=args.collect_batch),
    )
    cfg.environment.horizon = args.horizon
    cfg.environment.context_length = args.context_length
    if args.instruments:
        cfg.environment.instruments = tuple(args.instruments)
    return cfg


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


def _make_process_collector(
    cfg: ExperimentConfig,
    workers: int,
) -> tuple[ProcessCollector, int]:
    from forexmind.episodes.config import EpisodeConfig

    dataset, env_config, encoder, window_config = _build_components(cfg)
    obs_dim = encoder.config.spec.encoded_shape[0]
    episode_config = EpisodeConfig(
        split=cfg.environment.split,
        horizon=cfg.environment.horizon,
        context_length=cfg.environment.context_length,
        seed=cfg.compute.seed,
    )
    collector = ProcessCollector(
        processed_dir=str(DEFAULT_PROCESSED_DIR),
        split_config=dataset.split_config,
        instruments=dataset.instruments,
        env_config=env_config,
        encoder_config=encoder.config,
        window_config=window_config,
        episode_config=episode_config,
        algorithm="sac",
        model=cfg.model,
        obs_dim=obs_dim,
        action_dim=1,
        global_seed=cfg.compute.seed,
        num_workers=workers,
    )
    return collector, obs_dim


def _push_replay(replay: ReplayBuffer, transitions: list[Transition]) -> None:
    for t in transitions:
        replay.push(t.obs, t.action, t.reward, t.next_obs, t.terminated, t.truncated)


def run_collection_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    from forexmind.training.policies import build_policy_network

    base_cfg = _load_benchmark_config(args)
    rows: list[dict[str, object]] = []
    for workers in parse_worker_counts(args.workers):
        cfg = replace(base_cfg, compute=replace(base_cfg.compute, num_workers=workers))
        collector, obs_dim = _make_process_collector(cfg, workers)
        policy = build_policy_network("sac", obs_dim, 1, cfg.model)
        collector.set_policy(policy)
        replay = ReplayBuffer(
            obs_dim=obs_dim,
            capacity=max(cfg.training.replay_capacity, cfg.training.batch_size),
            action_dim=1,
            seed=cfg.compute.seed,
        )
        cpu_report: dict[str, object] = {}
        wall = 0.0
        steps = 0
        episodes = 0
        producer_ids: set[int] = set()
        try:
            warm = 0
            while warm < args.warmup_steps:
                batch = collector.collect(
                    min(args.collect_batch, args.warmup_steps - warm),
                    random_action=args.random_actions,
                )
                _push_replay(replay, batch)
                warm += len(batch)

            sampler = ProcessTreeCpuSampler(worker_pids=collector.worker_pids)
            sampler.start()
            t0 = time.perf_counter()
            while steps < args.measured_steps:
                batch = collector.collect(
                    min(args.collect_batch, args.measured_steps - steps),
                    random_action=args.random_actions,
                )
                _push_replay(replay, batch)
                steps += len(batch)
                episodes += sum(1 for t in batch if t.terminated or t.truncated)
                producer_ids.update(t.worker_id for t in batch)
            wall = time.perf_counter() - t0
            cpu_report = dict(sampler.stop())
            print_process_tree_report(
                worker_pids=collector.worker_pids,
                workers_configured=workers,
                sample_seconds=0.0,
                cpu_sample=cpu_report,
            )
        finally:
            collector.close()

        row = {
            "mode": "B_collection",
            "workers": workers,
            "env_steps": steps,
            "wall_seconds": round(wall, 3),
            "steps_per_second": round(steps / wall, 3) if wall > 0 else 0.0,
            "episodes": episodes,
            "replay_size": replay.size,
            "workers_producing_transitions": len(producer_ids),
            "logical_cpus": logical_cpu_count(),
            **cpu_report,
        }
        rows.append(row)
        print(
            f"B collection workers={workers:>3} steps={steps:,} steps/s={row['steps_per_second']}"
        )
    return rows


def run_full_sac_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    from forexmind.training.sac import SACTrainer

    base_cfg = _load_benchmark_config(args)
    rows: list[dict[str, object]] = []
    for workers in parse_worker_counts(args.workers):
        cfg = replace(
            base_cfg,
            compute=replace(base_cfg.compute, num_workers=workers),
            training=replace(
                base_cfg.training,
                collect_batch=args.collect_batch,
                warmup_steps=0,
                total_env_steps=args.measured_steps,
            ),
        )
        cpu_report: dict[str, object] = {}
        wall = 0.0
        steps = 0
        gradients = 0
        producer_ids: set[int] = set()
        replay_size = 0
        with tempfile.TemporaryDirectory(prefix=f"forexmind_bench_sac_{workers}_") as tmp:
            trainer = SACTrainer(cfg, tmp)
            try:
                trainer._sync_policy_to_workers()
                warm_target = max(args.warmup_steps, cfg.training.batch_size)
                warm = 0
                while warm < warm_target:
                    batch = trainer.collector.collect(
                        min(args.collect_batch, warm_target - warm),
                        random_action=True,
                    )
                    trainer._ingest(batch)
                    trainer._consume_transitions(batch)
                    warm += len(batch)

                worker_pids = getattr(trainer.collector, "worker_pids", [])
                sampler = ProcessTreeCpuSampler(worker_pids=worker_pids)
                sampler.start()
                t0 = time.perf_counter()
                steps0 = trainer._env_steps
                grad0 = trainer._gradient_updates
                while trainer._env_steps - steps0 < args.measured_steps:
                    remaining = args.measured_steps - (trainer._env_steps - steps0)
                    batch = trainer.collector.collect(
                        min(args.collect_batch, remaining),
                        random_action=False,
                    )
                    producer_ids.update(t.worker_id for t in batch)
                    trainer._ingest(batch)
                    trainer._consume_transitions(batch)
                    trainer._maybe_resync_policy()
                wall = time.perf_counter() - t0
                cpu_report = dict(sampler.stop())
                print_process_tree_report(
                    worker_pids=worker_pids,
                    workers_configured=workers,
                    sample_seconds=0.0,
                    cpu_sample=cpu_report,
                )
                steps = trainer._env_steps - steps0
                gradients = trainer._gradient_updates - grad0
                replay_size = trainer.replay_size()
            finally:
                trainer.collector.close()

        row = {
            "mode": "C_full_sac",
            "workers": workers,
            "env_steps": steps,
            "wall_seconds": round(wall, 3),
            "steps_per_second": round(steps / wall, 3) if wall > 0 else 0.0,
            "gradient_updates": gradients,
            "end_to_end_sac_steps_per_second": round(steps / wall, 3) if wall > 0 else 0.0,
            "replay_size": replay_size,
            "workers_producing_transitions": len(producer_ids),
            "logical_cpus": logical_cpu_count(),
            **cpu_report,
        }
        rows.append(row)
        print(
            f"C full-sac workers={workers:>3} steps={steps:,} "
            f"steps/s={row['steps_per_second']} gradients={gradients:,}"
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
        ExperimentConfig.from_dict(cfg_dict) if isinstance(cfg_dict, dict) else ExperimentConfig()
    )
    dataset, env_config, encoder, window_config = _build_components(config)
    policy, algorithm = load_checkpoint_policy(
        checkpoint, encoder.config.spec.encoded_shape[0], config.model
    )

    evaluator = PolicyEvaluator(
        dataset,
        env_config,
        encoder,
        window_config,
        selection_metric=config.selection.metric,
        lambda_drawdown=config.selection.lambda_drawdown,
        eval_horizon=config.evaluation.eval_horizon,
        eval_seed=config.evaluation.eval_seed,
        context_length=config.environment.context_length,
    )
    val = evaluator.evaluate(policy, algorithm, "validation", episodes)
    print(
        f"\nFrozen policy on validation: score={val.score:.4f} "
        f"sharpe={val.metrics.get('sharpe', 0.0):.4f}"
    )

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
    for name, path in paths.items():
        print(f"  {name:<6} {path}")
    print("\nAggregate metrics:")
    for row in bench["results"]:
        metrics = row["metrics"]
        print(
            f"  {row['agent']:<16} ret={metrics.get('total_return', 0):+.4f}  "
            f"sharpe={metrics.get('sharpe', 0):+.4f}  "
            f"sortino={metrics.get('sortino', 0):+.4f}  "
            f"mdd={metrics.get('max_drawdown_pct', 0):.4f}  "
            f"turnover={metrics.get('turnover', 0):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="ForexMind training throughput benchmark.")
    parser.add_argument(
        "--workers",
        nargs="+",
        default=None,
        help="Comma-separated or space-separated worker counts.",
    )
    parser.add_argument("--mode", choices=("all", "env", "collection", "full-sac"), default="all")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--measured-steps", "--n-steps", dest="measured_steps", type=int, default=100_000
    )
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--collect-batch", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="Use random worker actions in collection-only mode.",
    )
    parser.add_argument("--instruments", nargs="+", default=None)
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Frozen checkpoint for final benchmark tables."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--json", type=str, default=None, help="Optional JSON path for throughput rows."
    )
    args = parser.parse_args()

    if args.checkpoint:
        run_final_benchmark(args.checkpoint, args.episodes, args.out)
        return

    rows: list[dict[str, object]] = []
    if args.mode in ("all", "env"):
        rows.extend(run_env_only_sweep(args))
    if args.mode in ("all", "collection"):
        rows.extend(run_collection_sweep(args))
    if args.mode in ("all", "full-sac"):
        rows.extend(run_full_sac_sweep(args))

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"\nTraining benchmark saved to {args.json}")


if __name__ == "__main__":
    main()
