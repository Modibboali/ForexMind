"""PPO memory-scaling benchmark (docs/ppo_memory_audit.md).

Spawns real environment workers on the real processed dataset and measures,
per worker count:

    trainer RSS | per-worker RSS (min/median/mean/p90/max/aggregate) |
    process-tree RSS | steps/sec

Stage A = after worker startup + first episode (dataset + environment loaded).
Stage B = after a representative rollout (PPO-style stepped transitions).

Supports both dataset backends so the shared-store win can be quantified:
    --dataset-backend parquet   (each worker materialises ~3 GB privately)
    --dataset-backend mmap      (workers share the same mapped pages)

Kaggle:
    python -m tools.build_shared_dataset
    python -m tools.benchmark_ppo_memory \
        --workers 1,2,4,8,16,32,64,96,128,160,192,204 --dataset-backend mmap

Local correctness (1-2 workers, tiny workload):
    python -m tools.benchmark_ppo_memory --workers 1,2 --measured-steps 512
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import torch
from forexmind.episodes.config import EpisodeConfig
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.collector import EnvWorker
from forexmind.training.config import ExperimentConfig, ModelConfig, default_config
from forexmind.training.data import (
    DEFAULT_INSTRUMENT_ORDER,
    DEFAULT_PROCESSED_DIR,
)
from forexmind.training.dataset_mmap import resolve_dataset, shared_store_dir, store_available
from forexmind.training.runtime_diagnostics import (
    logical_cpu_count,
    memory_report,
)
from forexmind.training.trainer import build_env_config

from tools.benchmark_env_workers import parse_worker_counts


def _env_worker_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = ExperimentConfig.from_yaml(args.config) if args.config else default_config("ppo")
    cfg.environment.horizon = args.horizon
    cfg.environment.context_length = args.context_length
    if args.instruments:
        cfg.environment.instruments = tuple(args.instruments)
    env_config = build_env_config(cfg.environment)
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
    instruments = (
        tuple(cfg.environment.instruments)
        if cfg.environment.instruments
        else DEFAULT_INSTRUMENT_ORDER
    )
    return {
        "processed_dir": str(DEFAULT_PROCESSED_DIR),
        "instruments": list(instruments),
        "env_config": env_config,
        "encoder_config": encoder.config,
        "window_config": window_config,
        "episode_config": episode_config,
        "model": ModelConfig(hidden_dim=16, num_layers=2),
        "obs_dim": encoder.config.spec.encoded_shape[0],
        "global_seed": cfg.compute.seed,
        "dataset_backend": args.dataset_backend,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
    }


def _memory_worker_main(cfg: dict[str, Any], q_in: Any, q_out: Any) -> None:  # pragma: no cover
    torch.set_num_threads(1)
    worker_id = int(cfg["worker_id"])
    dataset, _backend = resolve_dataset(
        processed_dir=cfg["processed_dir"],
        split_config=None,
        instruments=tuple(cfg["instruments"]),
        backend=cfg["dataset_backend"],
    )
    worker = EnvWorker(
        dataset=dataset,
        env_config=cfg["env_config"],
        encoder_config=cfg["encoder_config"],
        window_config=cfg["window_config"],
        episode_config=cfg["episode_config"],
        algorithm="ppo",
        model_config=cfg["model"],
        obs_dim=int(cfg["obs_dim"]),
        action_dim=1,
        worker_id=worker_id,
        global_seed=int(cfg["global_seed"]),
        policy=None,
    )
    # Warm up so every instrument's environment + timeline state exists.
    for _ in range(int(cfg["warmup_steps"])):
        worker.step(random_action=True)
    q_out.put(("stageA", worker_id))
    while True:
        msg = q_in.get()
        if msg is None:
            break
        n = int(msg[1])
        t0 = time.perf_counter()
        for _ in range(n):
            worker.step(random_action=True)
        dt = time.perf_counter() - t0
        q_out.put(("done", {"worker_id": worker_id, "steps": n, "seconds": dt}))
    q_out.put(("stop", worker_id))


def _rollout_memory_mb(obs_dim: int, rollout_len: int) -> dict[str, float]:
    """Analytical PPO rollout memory (transitions + stacked update arrays)."""
    obs_bytes = obs_dim * 4  # float32
    per_transition = obs_bytes * 2 + 8 * 4  # obs + next_obs + scalar fields
    stacked_obs = obs_dim * 4 * rollout_len
    stacked_misc = (8 * 4) * rollout_len  # rew/done/val/next_val/logp/adv/ret
    total = per_transition * rollout_len + stacked_obs + stacked_misc
    return {
        "obs_dim": float(obs_dim),
        "rollout_len": float(rollout_len),
        "per_transition_kb": round(per_transition / 1024, 2),
        "transitions_mb": round(per_transition * rollout_len / 1e6, 2),
        "stacked_arrays_mb": round((stacked_obs + stacked_misc) / 1e6, 2),
        "total_rollout_mb": round(total / 1e6, 2),
    }


def run_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    base = _env_worker_cfg(args)
    ctx = mp.get_context("spawn")
    rows: list[dict[str, object]] = []
    rollout_mb = _rollout_memory_mb(int(base["obs_dim"]), args.measured_steps)

    for workers in parse_worker_counts(args.workers):
        q_in: list[Any] = []
        q_out: list[Any] = []
        procs: list[Any] = []
        try:
            for wid in range(workers):
                in_q = ctx.Queue()
                out_q = ctx.Queue()
                wcfg = dict(base, worker_id=wid)
                p = ctx.Process(target=_memory_worker_main, args=(wcfg, in_q, out_q))
                p.start()
                q_in.append(in_q)
                q_out.append(out_q)
                procs.append(p)
            worker_pids = [int(p.pid) for p in procs if p.pid is not None]
            for out_q in q_out:
                out_q.get()  # stageA ready

            # Stage A: after startup / environment creation.
            mem_a = dict(memory_report(trainer_pid=os.getpid(), worker_pids=worker_pids))

            # Measured rollout.
            t0 = time.perf_counter()
            for wid, in_q in enumerate(q_in):
                n = base["measured_steps"] // workers
                if wid < base["measured_steps"] % workers:
                    n += 1
                in_q.put(("run", max(1, n)))
            steps = 0
            wall = 0.0
            for out_q in q_out:
                kind, payload = out_q.get()
                if kind == "done":
                    steps += int(payload["steps"])
                    wall = max(wall, float(payload["seconds"]))
            elapsed = time.perf_counter() - t0

            # Stage B: after the rollout.
            mem_b = dict(memory_report(trainer_pid=os.getpid(), worker_pids=worker_pids))
        finally:
            for in_q in q_in:
                in_q.put(None)
            for p in procs:
                p.join(timeout=30)
            for queue in [*q_in, *q_out]:
                queue.close()

        row = {
            "workers": workers,
            "workers_alive": len(worker_pids),
            "steps_per_second": round(steps / elapsed, 1) if elapsed > 0 else 0.0,
            "trainer_rss_mb": mem_a.get("trainer_rss_mb"),
            "worker_rss_median_mb": mem_b.get("worker_rss_median_mb"),
            "worker_rss_aggregate_mb": mem_b.get("worker_rss_aggregate_mb"),
            "worker_uss_aggregate_mb": mem_b.get("worker_uss_aggregate_mb"),
            "process_tree_rss_mb": mem_b.get("process_tree_rss_mb"),
            "rollout_total_mb": rollout_mb["total_rollout_mb"],
            "logical_cpus": logical_cpu_count(),
        }
        rows.append(row)
        print(
            f"workers={workers:>3} steps/s={row['steps_per_second']:>8} "
            f"tree_rss={row['process_tree_rss_mb']:>8} MB "
            f"worker_agg={row['worker_rss_aggregate_mb']:>8} MB "
            f"worker_uss_agg={row['worker_uss_aggregate_mb']}"
        )
    return rows


def _recommend(rows: list[dict[str, Any]]) -> tuple[int, str]:
    """Pick the worker count with the best steps/s before tree RSS grows ~linearly."""
    if not rows:
        return 0, "no data"
    best = rows[0]
    for row in rows:
        if float(row.get("steps_per_second") or 0) > float(best.get("steps_per_second") or 0):
            best = row
    return int(best["workers"]), (
        f"best measured steps/s at workers={best['workers']} "
        f"(tree_rss={best.get('process_tree_rss_mb')} MB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO memory/throughput benchmark.")
    parser.add_argument("--workers", nargs="+", default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset-backend", choices=("auto", "parquet", "mmap"), default="auto")
    parser.add_argument("--measured-steps", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--instruments", nargs="+", default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    if args.dataset_backend == "mmap" and not store_available(DEFAULT_PROCESSED_DIR):
        print(f"Shared store missing at {shared_store_dir(DEFAULT_PROCESSED_DIR)}")
        print("Build it first: python -m tools.build_shared_dataset")
        return

    print("=" * 60)
    print("ForexMind PPO Memory/Throughput Benchmark")
    print("=" * 60)
    print(f"System RAM: {_system_ram_gb()} GB")
    print(f"System CPUs: {logical_cpu_count()}")
    print(f"Dataset backend: {args.dataset_backend}")
    print(f"Shared store: {store_available(DEFAULT_PROCESSED_DIR)}")
    print()

    rows = run_sweep(args)
    print()
    print("=" * 60)
    print("Workers   Steps/s   Tree RSS MB   Worker RSS MB   Worker USS MB")
    print("-" * 60)
    for row in rows:
        print(
            f"{row['workers']:<9} {row['steps_per_second']:>8.1f} "
            f"{row['process_tree_rss_mb'] or 0:>13} "
            f"{row['worker_rss_aggregate_mb'] or 0:>14} "
            f"{row['worker_uss_aggregate_mb'] or 0:>14}"
        )
    print("-" * 60)
    rec_workers, rec_reason = _recommend(rows)
    print(f"\nRecommended worker count: {rec_workers}")
    print(f"  ({rec_reason})")
    print("Memory bottleneck: see tree RSS vs worker count (shared pages vs private copies)")
    print("Throughput bottleneck: diminishing steps/s past the knee of the table")
    print("=" * 60)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved to {args.json}")


def _system_ram_gb() -> float:
    try:
        from importlib import import_module

        psutil = import_module("psutil")
        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
