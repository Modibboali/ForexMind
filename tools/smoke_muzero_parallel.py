"""Stage 4.6 smoke test: parallel MuZero collection with batched inference.

Starts ``num_workers`` collector processes, each driving
``collectors_per_worker`` episodes in lock-step through one batched MCTS, and
collects a small number of real TRAIN trajectories on the configured
instruments.  Prints worker, queue, inference-batch and replay diagnostics.

Usage (repository root)::

    python -m tools.smoke_muzero_parallel --workers 2 --collectors-per-worker 2 \
        --simulations 4 --horizon 4 --trajectories 4
    python -m tools.smoke_muzero_parallel --inference-mode local --workers 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SplitConfig
from forexmind.muzero.collector import CollectorConfig
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
    collect_runtime_report,
    print_memory_report,
    print_process_tree_report,
)

from tools.common import PROCESSED_DATA_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--collectors-per-worker", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--instruments", nargs="+", default=["EURUSD", "GBPUSD"])
    parser.add_argument("--inference-mode", default="server", choices=["server", "local"])
    parser.add_argument("--max-inference-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--queue-size", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dataset-backend", default="auto")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0, help="collection timeout (s)")
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
    model = build_muzero_network(model_config)
    model.eval()

    pool_config = CollectorPoolConfig(
        num_workers=args.workers,
        collectors_per_worker=args.collectors_per_worker,
        max_inference_batch_size=args.max_inference_batch_size,
        max_batch_wait_ms=args.max_batch_wait_ms,
        inference_mode=args.inference_mode,
        trajectory_queue_size=args.queue_size,
    )
    collector_config = CollectorConfig(
        split="train",
        horizon=args.horizon,
        num_simulations=args.simulations,
        seed=args.seed,
        training=True,
        temperature=1.0,
        capture_diagnostics=True,
        per_decision_seed=True,
    )
    replay = TrajectoryReplayBuffer(ReplayConfig(max_trajectories=64, seed=args.seed))
    dataset_spec = WorkerDatasetSpec.from_dataset(
        dataset, processed_dir=args.processed_dir, backend=args.dataset_backend
    )
    pool = MuZeroCollectorPool(
        config=pool_config,
        dataset_spec=dataset_spec,
        env_config=env_config,
        encoder_config=encoder_config,
        collector_config=collector_config,
        model_factory=lambda: build_muzero_network(model_config),
        model_config=model_config,
        model=model,
        replay=replay,
        model_version_string=model_version(model),
    )

    print("=" * 72, flush=True)
    print("Stage 4.6 parallel MuZero collection smoke test", flush=True)
    print("=" * 72, flush=True)
    print(f"dataset backend   : {backend}", flush=True)
    print(f"workers           : {pool_config.num_workers}", flush=True)
    print(f"collectors/worker : {pool_config.collectors_per_worker}", flush=True)
    print(f"total collectors  : {pool_config.num_collectors}", flush=True)
    print(f"inference mode    : {pool_config.inference_mode}", flush=True)
    print(f"trajectories      : {args.trajectories}", flush=True)
    print(f"worker pids       : {pool.worker_pids}", flush=True)
    print("-" * 72, flush=True)

    diagnostics: dict = {}
    shutdown: dict = {}
    memory_before: dict = {}
    memory_after: dict = {}
    runtime: dict = {}
    try:
        memory_before = print_memory_report(
            worker_pids=pool.worker_pids,
            workers_configured=pool_config.num_workers,
            label="MEMORY (start)",
        )
        collected = pool.collect_trajectories(args.trajectories, timeout_s=args.timeout)
        for item in collected:
            metadata = item.trajectory.metadata
            print(
                f"trajectory {metadata.trajectory_id:>8} | network version "
                f"{metadata.network_version} | steps {len(item.trajectory):>3} | "
                f"reward {float(item.trajectory.rewards.sum()):+.6f} | "
                f"records {len(item.records)}",
                flush=True,
            )
        diagnostics = pool.diagnostics()
        print("-" * 72, flush=True)
        print(json.dumps(diagnostics, indent=2, default=str), flush=True)
        print_process_tree_report(
            worker_pids=pool.worker_pids, workers_configured=pool_config.num_workers
        )
        memory_after = print_memory_report(
            worker_pids=pool.worker_pids,
            workers_configured=pool_config.num_workers,
            label="MEMORY (end)",
        )
        runtime = collect_runtime_report(worker_pids=pool.worker_pids, sample_seconds=0.0)
    finally:
        shutdown = pool.close()
    print("-" * 72)
    print(f"shutdown: {json.dumps(shutdown, indent=2, default=str)}")

    payload = {
        "config": pool_config.to_dict(),
        "collector": collector_config.to_dict(),
        "dataset_backend": backend,
        "runtime": runtime,
        "memory_start": memory_before,
        "memory_end": memory_after,
        "diagnostics": diagnostics,
        "shutdown": shutdown,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
