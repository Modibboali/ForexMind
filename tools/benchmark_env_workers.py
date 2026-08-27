"""Environment-only worker scalability benchmark.

This removes replay insertion and SAC learner updates. Each worker creates a
real ForexMind training environment, repeatedly samples a random action, calls
``env.step()``, and returns only a small summary after the measured interval.

Intended Kaggle use:
    python -m tools.benchmark_env_workers --workers 1,2,4,8,16,32,64,128,192,204
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import torch
from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SplitConfig
from forexmind.episodes.config import EpisodeConfig
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.collector import EnvWorker
from forexmind.training.config import ExperimentConfig, ModelConfig, default_config
from forexmind.training.data import (
    DEFAULT_INSTRUMENT_ORDER,
    DEFAULT_PROCESSED_DIR,
    make_training_dataset,
)
from forexmind.training.runtime_diagnostics import logical_cpu_count, print_process_tree_report
from forexmind.training.trainer import build_env_config


def parse_worker_counts(values: list[str] | None) -> list[int]:
    if not values:
        return [1, 2, 4, 8, 16, 32, 64, 128, 192, 204]
    counts: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                counts.append(int(part))
    return counts


def _worker_main(cfg: dict[str, Any], q_in: Any, q_out: Any) -> None:  # pragma: no cover
    torch.set_num_threads(1)
    worker_id = int(cfg["worker_id"])
    split_config = SplitConfig.from_dict(cfg["split_config"])
    dataset = make_training_dataset(cfg["processed_dir"], split_config, tuple(cfg["instruments"]))
    worker = EnvWorker(
        dataset=dataset,
        env_config=cfg["env_config"],
        encoder_config=cfg["encoder_config"],
        window_config=cfg["window_config"],
        episode_config=cfg["episode_config"],
        algorithm="sac",
        model_config=cfg["model"],
        obs_dim=int(cfg["obs_dim"]),
        action_dim=1,
        worker_id=worker_id,
        global_seed=int(cfg["global_seed"]),
        policy=None,
    )
    warmup_steps = int(cfg["warmup_steps"])
    for _ in range(warmup_steps):
        worker.step(random_action=True)
    q_out.put(("ready", worker_id))
    while True:
        msg = q_in.get()
        if msg is None:
            break
        n_steps = int(msg[1])
        t0 = time.perf_counter()
        episodes0 = worker.completed_episodes
        for _ in range(n_steps):
            worker.step(random_action=True)
        dt = time.perf_counter() - t0
        q_out.put(
            (
                "done",
                {
                    "worker_id": worker_id,
                    "env_steps": n_steps,
                    "wall_seconds": dt,
                    "episodes": worker.completed_episodes - episodes0,
                },
            )
        )


def _benchmark_config(args: argparse.Namespace) -> tuple[ExperimentConfig, EnvironmentConfig]:
    cfg = ExperimentConfig.from_yaml(args.config) if args.config else default_config("sac")
    cfg.environment.horizon = args.horizon
    cfg.environment.context_length = args.context_length
    if args.instruments:
        cfg.environment.instruments = tuple(args.instruments)
    env_config = build_env_config(cfg.environment)
    return cfg, env_config


def run_env_only_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    cfg, env_config = _benchmark_config(args)
    instruments = (
        tuple(cfg.environment.instruments)
        if cfg.environment.instruments
        else DEFAULT_INSTRUMENT_ORDER
    )
    dataset = make_training_dataset(DEFAULT_PROCESSED_DIR, None, instruments)
    encoder = ObservationEncoder(
        EncoderConfig(
            context_length=cfg.environment.context_length,
            initial_balance=env_config.margin.initial_balance,
        )
    )
    window_config = WindowConfig(context_length=cfg.environment.context_length)
    episode_config = EpisodeConfig(
        split=cfg.environment.split,
        horizon=cfg.environment.horizon,
        context_length=cfg.environment.context_length,
        seed=cfg.compute.seed,
    )
    base_cfg = {
        "processed_dir": str(DEFAULT_PROCESSED_DIR),
        "split_config": dataset.split_config.to_dict(),
        "instruments": list(dataset.instruments),
        "env_config": env_config,
        "encoder_config": encoder.config,
        "window_config": window_config,
        "episode_config": episode_config,
        "model": ModelConfig(hidden_dim=16, num_layers=2),
        "obs_dim": encoder.config.spec.encoded_shape[0],
        "global_seed": cfg.compute.seed,
        "warmup_steps": args.warmup_steps,
    }

    rows: list[dict[str, object]] = []
    ctx = mp.get_context("spawn")
    for workers in parse_worker_counts(args.workers):
        steps_per_worker = args.measured_steps // workers
        remainder = args.measured_steps % workers
        q_in: list[Any] = []
        q_out: list[Any] = []
        procs: list[Any] = []
        for wid in range(workers):
            in_q = ctx.Queue()
            out_q = ctx.Queue()
            proc_cfg = dict(base_cfg, worker_id=wid)
            proc = ctx.Process(target=_worker_main, args=(proc_cfg, in_q, out_q))
            proc.start()
            q_in.append(in_q)
            q_out.append(out_q)
            procs.append(proc)
        worker_pids = [int(p.pid) for p in procs if p.pid is not None]
        for out_q in q_out:
            kind, _payload = out_q.get()
            if kind != "ready":
                raise RuntimeError(f"unexpected worker message {kind!r}")

        t0 = time.perf_counter()
        for wid, in_q in enumerate(q_in):
            n = steps_per_worker + (1 if wid < remainder else 0)
            in_q.put(("run", n))
        cpu_report = print_process_tree_report(
            worker_pids=worker_pids,
            workers_configured=workers,
            sample_seconds=args.sample_seconds,
        )
        summaries: list[dict[str, Any]] = []
        for out_q in q_out:
            kind, payload = out_q.get()
            if kind != "done":
                raise RuntimeError(f"unexpected worker message {kind!r}")
            summaries.append(dict(payload))
        wall = time.perf_counter() - t0

        for in_q in q_in:
            in_q.put(None)
        for proc in procs:
            proc.join(timeout=30)
        for queue in [*q_in, *q_out]:
            queue.close()

        steps = sum(int(s["env_steps"]) for s in summaries)
        row = {
            "mode": "A_env_only",
            "workers": workers,
            "env_steps": steps,
            "wall_seconds": round(wall, 3),
            "steps_per_second": round(steps / wall, 3) if wall > 0 else 0.0,
            "episodes": sum(int(s["episodes"]) for s in summaries),
            "workers_producing_transitions": sum(1 for s in summaries if int(s["env_steps"]) > 0),
            "logical_cpus": logical_cpu_count(),
            **cpu_report,
        }
        rows.append(row)
        print(f"A env-only workers={workers:>3} steps={steps:,} steps/s={row['steps_per_second']}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark raw env worker scalability.")
    parser.add_argument(
        "--workers",
        nargs="+",
        default=None,
        help="Comma-separated or space-separated worker counts.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--measured-steps", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--instruments", nargs="+", default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    rows = run_env_only_sweep(args)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"\nEnvironment-only benchmark saved to {args.json}")


if __name__ == "__main__":
    main()
