"""Stage 4.6 scaling benchmark: workers x MCTS budget (brief S28-S30, S44).

Measures *collection* throughput for a controlled matrix of collector counts
and MCTS simulation budgets, keeping every other parameter fixed (brief S46:
apples-to-apples).  Two modes:

* ``--sequential``: the Stage 4.5 single-process collector (baseline).
* default: the Stage 4.6 collector pool (workers x collectors-per-worker)
  with the central batched inference service.

Reported per configuration: env steps/s, searches/s, simulations/s, inference
batch statistics, worker CPU utilisation and RAM.

Usage (repository root)::

    python -m tools.benchmark_muzero_scaling --workers 1 2 4 8
    python -m tools.benchmark_muzero_scaling --sequential
    python -m tools.benchmark_muzero_scaling --workers 4 --simulations 8 16 32 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SplitConfig
from forexmind.muzero.collector import CollectorConfig, MuZeroCollector
from forexmind.muzero.config import MuZeroConfig
from forexmind.muzero.inference import build_muzero_network
from forexmind.muzero.parallel_collector import (
    CollectorPoolConfig,
    MuZeroCollectorPool,
    WorkerDatasetSpec,
)
from forexmind.muzero.replay import ReplayConfig, TrajectoryReplayBuffer
from forexmind.muzero.trajectory import model_version
from forexmind.observation.encoder import EncoderConfig
from forexmind.training.dataset_mmap import resolve_dataset
from forexmind.training.runtime_diagnostics import (
    ProcessTreeCpuSampler,
    collect_runtime_report,
    memory_report,
)

from tools.common import PROCESSED_DATA_DIR, REPORTS_DIR


def _collector_config(args: argparse.Namespace, simulations: int) -> CollectorConfig:
    return CollectorConfig(
        split="train",
        horizon=args.horizon,
        num_simulations=simulations,
        discount=args.discount,
        training=True,
        temperature=args.temperature,
        seed=args.seed,
        capture_diagnostics=False,
        per_decision_seed=True,
        boundary_search=not args.no_boundary_search,
    )


def _run_sequential(
    args: argparse.Namespace,
    dataset: Any,
    env_config: EnvironmentConfig,
    encoder_config: EncoderConfig,
    model: Any,
    simulations: int,
) -> dict[str, Any]:
    collector = MuZeroCollector(
        dataset,
        env_config,
        encoder_config,
        model,
        _collector_config(args, simulations),
        instruments=tuple(args.instruments),
    )
    # Match the workers' threading configuration so the comparison is
    # apples-to-apples (brief S26: per-call BLAS thread overhead is real).
    torch.set_num_threads(int(args.torch_threads_per_worker))
    # Warm-up is not timed: it removes dataset loading, first env creation and
    # first-search costs from the steady-state throughput (same for every mode).
    if args.warmup_episodes:
        collector.collect(args.warmup_episodes)
        collector.stats = type(collector.stats)()
    sampler = ProcessTreeCpuSampler()
    sampler.start()
    start = time.perf_counter()
    collector.collect(args.episodes)
    elapsed = time.perf_counter() - start
    cpu = sampler.stop()
    stats = collector.stats
    return {
        "mode": "sequential",
        "workers": 0,
        "collectors": 1,
        "simulations": simulations,
        "episodes": stats.trajectories,
        "env_steps": stats.env_steps,
        "searches": stats.searches,
        "seconds": elapsed,
        "env_steps_per_sec": stats.env_steps / elapsed,
        "searches_per_sec": stats.searches / elapsed,
        "simulations_per_sec": stats.searches * simulations / elapsed,
        "recurrent_inference_calls": stats.recurrent_inference_calls,
        "warmup_episodes": args.warmup_episodes,
        "batch": {
            "mean_batch_size": 1.0,
            "p90_batch_size": 1.0,
            "max_batch_size": 1.0,
            "mean_batch_wait_ms": 0.0,
            "mean_inference_latency_ms": (
                1e3 * elapsed / max(stats.recurrent_inference_calls, 1)
            ),
        },
        "cpu": cpu,
        "memory": memory_report(),
        "shutdown": {"alive_workers": 0},
    }


def _run_pool(
    args: argparse.Namespace,
    dataset_paths: dict[str, Any],
    env_config: EnvironmentConfig,
    encoder_config: EncoderConfig,
    model: Any,
    model_config: MuZeroConfig,
    simulations: int,
) -> dict[str, Any]:
    workers = int(dataset_paths["workers"])
    pool_config = CollectorPoolConfig(
        num_workers=workers,
        collectors_per_worker=args.collectors_per_worker,
        max_inference_batch_size=args.max_inference_batch_size,
        max_batch_wait_ms=args.max_batch_wait_ms,
        inference_mode=args.inference_mode,
        trajectory_queue_size=args.queue_size,
        torch_threads_per_worker=args.torch_threads_per_worker,
        drop_when_full=args.drop_when_full,
    )
    replay = TrajectoryReplayBuffer(
        ReplayConfig(max_trajectories=max(args.episodes * 2, 64), seed=args.seed)
    )
    # No episode quota: collectors keep producing right through the measured
    # window (workers that expire mid-window would silently deflate the rate).
    episodes_per_worker: int | None = None
    pool = MuZeroCollectorPool(
        config=pool_config,
        dataset_spec=WorkerDatasetSpec(
            processed_dir=str(args.processed_dir),
            split_config=dataset_paths["split_config"].to_dict(),
            instruments=tuple(args.instruments),
            backend=args.dataset_backend,
        ),
        env_config=env_config,
        encoder_config=encoder_config,
        collector_config=_collector_config(args, simulations),
        model_factory=lambda: build_muzero_network(model_config),
        model_config=model_config,
        model=model,
        replay=replay,
        model_version_string=model_version(model),
        episodes_per_worker=episodes_per_worker,
    )
    try:
        if args.warmup_episodes:
            pool.collect_trajectories(args.warmup_episodes, timeout_s=args.timeout)
        before = pool.aggregate_stats()
        window_start = time.time()
        sampler = ProcessTreeCpuSampler(worker_pids=pool.worker_pids)
        sampler.start()
        start = time.perf_counter()
        collected = _collect_produced_after(pool, args.episodes, window_start, args.timeout)
        elapsed = time.perf_counter() - start
        cpu = sampler.stop()
        diagnostics = pool.diagnostics(fresh=True)
        memory = memory_report(worker_pids=pool.worker_pids)
    finally:
        shutdown = pool.close()
    aggregate = diagnostics["aggregate"]
    inference = diagnostics.get("inference_server") or aggregate.get("inference") or {}
    produced_at = np.asarray(
        [float(item.trajectory.extra.get("produced_at", 0.0)) for item in collected]
    )
    # Production window: from just before the measured window opened to the last
    # measured trajectory.  This is immune to worker run-ahead and to how quickly
    # the parent drained the queue (brief S28 methodology note).
    production_seconds = (
        float(max(produced_at.max() - window_start, 1e-9)) if len(produced_at) else 0.0
    )
    transitions = sum(len(item.trajectory) for item in collected)

    def delta(key: str) -> int:
        return int(aggregate.get(key, 0)) - int(before.get(key, 0))

    searches = delta("searches")
    env_steps_produced = delta("env_steps")
    recurrent_calls = delta("recurrent_inference_calls")
    calls = int(inference.get("calls", 0)) - int(before.get("inference", {}).get("calls", 0))
    items = int(inference.get("items", 0)) - int(before.get("inference", {}).get("items", 0))
    batch = {
        "mean_batch_size": (items / calls) if calls else 0.0,
        "median_batch_size": inference.get("median_batch_size", 0.0),
        "p90_batch_size": inference.get("p90_batch_size", 0.0),
        "max_batch_size": inference.get("max_batch_size", 0.0),
        "mean_batch_wait_ms": inference.get("mean_batch_wait_ms", 0.0),
        "p90_batch_wait_ms": inference.get("p90_batch_wait_ms", 0.0),
        "mean_inference_latency_ms": inference.get("mean_inference_latency_ms", 0.0),
        "calls": calls,
        "items": items,
    }
    return {
        "mode": "parallel",
        "workers": workers,
        "collectors": pool_config.num_collectors,
        "simulations": simulations,
        "episodes": len(collected),
        "env_steps": transitions,
        "env_steps_produced": env_steps_produced,
        "searches": searches,
        "seconds": elapsed,
        "production_seconds": production_seconds,
        "env_steps_per_sec": transitions / production_seconds if production_seconds else 0.0,
        "searches_per_sec": searches / production_seconds if production_seconds else 0.0,
        "simulations_per_sec": (
            searches * simulations / production_seconds if production_seconds else 0.0
        ),
        "take_seconds": elapsed,
        "recurrent_inference_calls": recurrent_calls,
        "batch": batch,
        "warmup_episodes": args.warmup_episodes,
        "queue": {
            "size": diagnostics["trajectory_queue"]["size"],
            "max_size": diagnostics["trajectory_queue"]["max_size"],
            "blocked_seconds": aggregate.get("blocked_seconds", 0.0),
            "dropped": aggregate.get("dropped", 0),
        },
        "writer": diagnostics["writer"],
        "dataset_backends": aggregate.get("dataset_backends", {}),
        "cpu": cpu,
        "memory": memory,
        "shutdown": {
            "alive_workers": shutdown.get("alive_workers"),
            "queue_drained": shutdown.get("queue_drained"),
            "workers_terminated": shutdown.get("workers_terminated"),
        },
    }


def _collect_produced_after(
    pool: MuZeroCollectorPool,
    count: int,
    window_start: float,
    timeout_s: float,
) -> list[Any]:
    """Collect ``count`` trajectories that were *produced* after ``window_start``.

    Trajectories the workers had already queued before the measured window are
    discarded instead of being counted as throughput.
    """
    collected: list[Any] = []
    deadline = time.perf_counter() + timeout_s
    while len(collected) < count:
        remaining = max(1.0, deadline - time.perf_counter())
        batch = pool.collect_trajectories(count - len(collected), timeout_s=remaining)
        for item in batch:
            produced_at = float(item.trajectory.extra.get("produced_at", 0.0))
            if produced_at >= window_start:
                collected.append(item)
    return collected


def _format_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'workers':>7} | {'coll':>4} | {'sims':>4} | {'env steps/s':>11} | "
        f"{'searches/s':>10} | {'sims/s':>9} | {'batch':>6} | {'cpu':>6} | {'RAM MB':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        batch = row["batch"].get("mean_batch_size", 0.0)
        cpu = row["cpu"].get("effective_cores_utilized", 0.0)
        memory = row["memory"].get("worker_rss_aggregate_mb")
        if not memory:  # single-process mode has no worker RSS to aggregate
            memory = row["memory"].get("process_tree_rss_mb", 0.0)
        lines.append(
            f"{row['workers']:>7} | {row['collectors']:>4} | {row['simulations']:>4} | "
            f"{row['env_steps_per_sec']:>11,.1f} | {row['searches_per_sec']:>10,.1f} | "
            f"{row['simulations_per_sec']:>9,.0f} | {batch:>6.2f} | {cpu:>6.2f} | "
            f"{memory:>8.1f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--collectors-per-worker", type=int, default=2)
    parser.add_argument("--simulations", type=int, nargs="+", default=[16])
    parser.add_argument("--episodes", type=int, default=16, help="trajectories per configuration")
    parser.add_argument(
        "--warmup-episodes",
        type=int,
        default=4,
        help="untimed trajectories collected before the measured window (S28)",
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--instruments", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY"])
    parser.add_argument("--sequential", action="store_true", help="run the Stage 4.5 baseline only")
    parser.add_argument("--inference-mode", default="server", choices=["server", "local"])
    parser.add_argument("--max-inference-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument("--drop-when-full", action="store_true")
    parser.add_argument("--no-boundary-search", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dataset-backend", default="auto")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    instruments = tuple(instrument.upper() for instrument in args.instruments)
    split_config = SplitConfig.default()
    dataset, backend = resolve_dataset(
        processed_dir=args.processed_dir,
        split_config=split_config,
        instruments=instruments,
        backend=args.dataset_backend,
    )
    encoder_config = EncoderConfig()
    env_config = EnvironmentConfig()
    model_config = MuZeroConfig(
        obs_dim=encoder_config.spec.encoded_shape[0],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    np.random.seed(args.seed)
    model = build_muzero_network(model_config)
    model.eval()

    print("=" * 96)
    print("Stage 4.6 - MuZero collection scaling benchmark")
    print("=" * 96)
    print(f"dataset backend     : {backend}")
    print(f"instruments         : {list(instruments)}")
    print(f"horizon             : {args.horizon}")
    print(f"episodes/config     : {args.episodes}")
    print(f"collectors/worker   : {args.collectors_per_worker}")
    print(f"inference mode      : {args.inference_mode}")
    print(f"max batch / wait    : {args.max_inference_batch_size} / {args.max_batch_wait_ms} ms")
    print(f"torch threads/worker: {args.torch_threads_per_worker}")
    print(
        f"model               : latent {args.latent_dim}, "
        f"hidden {args.hidden_dim} x {args.num_layers}"
    )
    print("-" * 96)

    rows: list[dict[str, Any]] = []
    configurations: list[tuple[str, int, int]] = []
    if args.sequential:
        for simulations in args.simulations:
            configurations.append(("sequential", 0, simulations))
    else:
        for workers in args.workers:
            for simulations in args.simulations:
                configurations.append(("parallel", int(workers), int(simulations)))

    for mode, workers, simulations in configurations:
        collector_count = (
            workers * args.collectors_per_worker if mode == "parallel" else 1
        )
        print(
            f"[run] mode={mode} workers={workers} collectors={collector_count} "
            f"sims={simulations}",
            flush=True,
        )
        if mode == "sequential":
            row = _run_sequential(args, dataset, env_config, encoder_config, model, simulations)
        else:
            row = _run_pool(
                args,
                {"workers": workers, "split_config": split_config},
                env_config,
                encoder_config,
                model,
                model_config,
                simulations,
            )
        row.update(
            {
                "inference_mode": args.inference_mode,
                "horizon": args.horizon,
                "model": {
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                },
                "dataset_backend": backend,
            }
        )
        rows.append(row)
        print(
            f"      env steps/s {row['env_steps_per_sec']:,.1f} | searches/s "
            f"{row['searches_per_sec']:,.1f} | sims/s {row['simulations_per_sec']:,.0f} | "
            f"batch {row['batch'].get('mean_batch_size', 0.0):.2f} | "
            f"cores {row['cpu'].get('effective_cores_utilized', 0.0):.2f}",
            flush=True,
        )

    print("-" * 96)
    print(_format_table(rows))
    print("-" * 96)
    print(json.dumps(collect_runtime_report(sample_seconds=0.0), indent=2, default=str))

    if args.json_out is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / "stage46_muzero_scaling.json"
    else:
        out = args.json_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": {
                    "episodes": args.episodes,
                    "horizon": args.horizon,
                    "simulations": args.simulations,
                    "workers": args.workers,
                    "collectors_per_worker": args.collectors_per_worker,
                    "inference_mode": args.inference_mode,
                    "max_inference_batch_size": args.max_inference_batch_size,
                    "max_batch_wait_ms": args.max_batch_wait_ms,
                    "queue_size": args.queue_size,
                    "torch_threads_per_worker": args.torch_threads_per_worker,
                    "instruments": list(instruments),
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dataset_backend": backend,
                    "seed": args.seed,
                },
                "rows": rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
