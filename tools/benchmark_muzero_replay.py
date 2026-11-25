"""Replay benchmark: sampling, insertion and store throughput (Stage 4.6 S20-S22)
plus the Stage 4.7 reference-vs-vectorized comparison (brief S2, S3, S13).

Measures, on real-shaped trajectories (no environment involved):

* the Stage 4.6 per-sample reference sampler and the Stage 4.7 packed/vectorized
  sampler side by side (same batch sizes, same trajectories, same positions),
* a phase breakdown of the reference path (trajectory selection, index
  construction, per-field gathers, value-target construction, mask/padding,
  batch stacking),
* replay insertion cost per trajectory,
* sampling throughput and latency for batch sizes 32 / 64 / 128 / 256,
* the shard replay store's write and read throughput,
* the payload (serialization) size and pickling cost used by the parallel
  collectors.

Usage (repository root)::

    python -m tools.benchmark_muzero_replay
    python -m tools.benchmark_muzero_replay --episodes 512 --horizon 64
    python -m tools.benchmark_muzero_replay --skip-reference
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from forexmind.muzero.actions import PlanningState
from forexmind.muzero.parallel_collector import payload_nbytes, trajectory_to_payload
from forexmind.muzero.profiling import PhaseTimer
from forexmind.muzero.replay import ReplayConfig, TrajectoryReplayBuffer
from forexmind.muzero.replay_store import load_replay, save_replay
from forexmind.muzero.targets import TargetConfig
from forexmind.muzero.trajectory import MuZeroTrajectory, TrajectoryMetadata

from tools.common import REPORTS_DIR

BATCH_SIZES = (32, 64, 128, 256)


def _synthetic_trajectory(
    index: int, *, steps: int, obs_dim: int, rng: np.random.Generator
) -> MuZeroTrajectory:
    """A realistic trajectory: float32 observations, six-action policies."""
    # A flat account's mask (FLAT is redundant) keeps the trajectory valid
    # without simulating the whole environment.
    mask = PlanningState.flat().action_mask()
    actions = rng.choice(np.flatnonzero(mask), size=steps).astype(np.int64)
    policies = rng.dirichlet(np.ones(mask.sum()), size=steps).astype(np.float32)
    root_policies = np.zeros((steps, 6), dtype=np.float32)
    root_policies[:, mask] = policies
    return MuZeroTrajectory(
        observations=rng.normal(size=(steps + 1, obs_dim)).astype(np.float32),
        actions=actions,
        rewards=rng.normal(scale=1e-3, size=steps).astype(np.float32),
        root_policies=root_policies,
        root_values=rng.normal(scale=1e-3, size=steps).astype(np.float32),
        action_masks=np.tile(mask, (steps, 1)),
        terminated=np.zeros(steps, dtype=bool),
        truncated=np.r_[np.zeros(steps - 1, dtype=bool), [True]],
        planning_exposure=np.zeros(steps + 1, dtype=np.float32),
        planning_is_flat=np.ones(steps + 1, dtype=bool),
        boundary_value=0.0,
        metadata=TrajectoryMetadata(
            trajectory_id=index,
            instrument="EURUSD",
            split="train",
            start_index=0,
            horizon=steps,
            episode_seed=index,
            search_seed=index,
            model_version="benchmark",
            num_simulations=16,
            discount=0.99,
            temperature=1.0,
            num_steps=steps,
            training=True,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--obs-dim", type=int, default=351)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="skip the slow Stage 4.6 reference sampler (it is ~100x slower)",
    )
    parser.add_argument("--profile-references", type=int, default=4)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    trajectories = [
        _synthetic_trajectory(index, steps=args.horizon, obs_dim=args.obs_dim, rng=rng)
        for index in range(args.episodes)
    ]
    config = ReplayConfig(max_trajectories=args.episodes * 2, seed=args.seed)
    target_config = TargetConfig(num_unroll_steps=5, td_steps=5, discount=0.99)

    print("=" * 84)
    print("Stage 4.6 - MuZero replay benchmark")
    print("=" * 84)
    print(f"episodes / horizon : {args.episodes} x {args.horizon}")
    print(f"observation dim    : {args.obs_dim}")
    print(f"total transitions  : {args.episodes * args.horizon}")

    # -- insertion ------------------------------------------------------------
    buffer = TrajectoryReplayBuffer(config)
    insertion_start = time.perf_counter()
    for trajectory in trajectories:
        buffer.add(trajectory)
    insertion_seconds = time.perf_counter() - insertion_start
    memory = buffer.memory_report()
    print("-" * 84)
    print(
        f"insertion          : {1000 * insertion_seconds / len(trajectories):.3f} ms/trajectory "
        f"({insertion_seconds:.3f} s total, {memory['total_mb']} MB)"
    )

    # -- sampling -------------------------------------------------------------
    sampling: dict[str, Any] = {}
    reference: dict[str, Any] = {}
    profile: dict[str, Any] = {}
    print("-" * 84)
    print(
        f"{'batch':>6} | {'old samples/s':>13} | {'new samples/s':>13} | "
        f"{'speedup':>8} | {'old ms/batch':>12} | {'new ms/batch':>12}"
    )
    print("-" * 84)
    for batch_size in BATCH_SIZES:
        generator = np.random.default_rng(args.seed)
        buffer.sample(batch_size, target_config=target_config, rng=generator)
        start = time.perf_counter()
        for _ in range(args.repeats):
            buffer.sample(batch_size, target_config=target_config, rng=generator)
        elapsed = time.perf_counter() - start
        row = {
            "batches_per_sec": args.repeats / elapsed,
            "samples_per_sec": args.repeats * batch_size / elapsed,
            "ms_per_batch": 1000.0 * elapsed / args.repeats,
            "us_per_sample": 1e6 * elapsed / (args.repeats * batch_size),
        }
        sampling[str(batch_size)] = row

        old_row: dict[str, Any] | None = None
        if not args.skip_reference:
            # The reference sampler is ~100x slower, so it gets fewer repeats -
            # but more than one so a single slow batch cannot dominate.
            reference_repeats = max(2, args.repeats // 4)
            generator = np.random.default_rng(args.seed)
            buffer.sample(
                batch_size,
                target_config=target_config,
                rng=generator,
                batch_backend="reference",
            )
            old_start = time.perf_counter()
            for _ in range(reference_repeats):
                buffer.sample(
                    batch_size,
                    target_config=target_config,
                    rng=generator,
                    batch_backend="reference",
                )
            old_elapsed = time.perf_counter() - old_start
            old_row = {
                "batches_per_sec": reference_repeats / old_elapsed,
                "samples_per_sec": reference_repeats * batch_size / old_elapsed,
                "ms_per_batch": 1000.0 * old_elapsed / reference_repeats,
                "us_per_sample": 1e6 * old_elapsed / (reference_repeats * batch_size),
                "repeats": reference_repeats,
            }
            reference[str(batch_size)] = old_row
        speedup = (
            (row["samples_per_sec"] / old_row["samples_per_sec"])
            if old_row is not None
            else float("nan")
        )
        print(
            f"{batch_size:>6} | {(old_row or {}).get('samples_per_sec', float('nan')):>13,.0f} | "
            f"{row['samples_per_sec']:>13,.0f} | {speedup:>8.1f}x | "
            f"{(old_row or {}).get('ms_per_batch', float('nan')):>12.2f} | "
            f"{row['ms_per_batch']:>12.2f}"
        )

    # -- reference phase breakdown (S3) --------------------------------------
    if not args.skip_reference:
        print("-" * 84)
        print("reference sampler phase breakdown (per batch of 32)")
        timer = PhaseTimer(enabled=True)
        generator = np.random.default_rng(args.seed)
        for _ in range(args.profile_references):
            buffer.sample(
                32,
                target_config=target_config,
                rng=generator,
                batch_backend="reference",
                timer=timer,
            )
        profile = timer.report()
        total = profile["total_seconds"] or 1e-9
        for name, payload in profile["phases"].items():
            print(
                f"  {name:<28} | {1000.0 * payload['seconds'] / args.profile_references:>9.3f} "
                f"ms/batch | {100.0 * payload['seconds'] / total:>5.1f}%"
            )

    # -- serialization --------------------------------------------------------
    payload = trajectory_to_payload(trajectories[0], worker_id=0)
    pickled = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    print("-" * 84)
    print(
        f"trajectory payload : {payload_nbytes(payload) / 1024:.1f} KB arrays, "
        f"{len(pickled) / 1024:.1f} KB pickled"
    )

    # -- shard store ----------------------------------------------------------
    store_dir = args.store_dir or (REPORTS_DIR / "_replay_benchmark")
    save_start = time.perf_counter()
    save_replay(buffer, store_dir)
    save_seconds = time.perf_counter() - save_start
    load_start = time.perf_counter()
    restored, store_report = load_replay(store_dir, config=config)
    load_seconds = time.perf_counter() - load_start
    print(
        f"shard store        : write {save_seconds:.2f} s, read {load_seconds:.2f} s "
        f"({store_report.bytes_on_disk / 1e6:.1f} MB, "
        f"{store_report.trajectories_loaded} trajectories)"
    )
    print("=" * 84)

    payload_json = {
        "config": {
            "episodes": args.episodes,
            "horizon": args.horizon,
            "obs_dim": args.obs_dim,
            "repeats": args.repeats,
        },
        "replay_memory": memory,
        "insertion_seconds": insertion_seconds,
        "insertion_ms_per_trajectory": 1000.0 * insertion_seconds / len(trajectories),
        "sampling": sampling,
        "reference_sampling": reference,
        "speedup_samples_per_sec": {
            key: (
                sampling[key]["samples_per_sec"] / reference[key]["samples_per_sec"]
                if key in reference
                else None
            )
            for key in sampling
        },
        "reference_profile_per_32_batch": profile,
        "payload": {
            "array_bytes": payload_nbytes(payload),
            "pickled_bytes": len(pickled),
        },
        "store": {
            "write_seconds": save_seconds,
            "read_seconds": load_seconds,
            "bytes": store_report.bytes_on_disk,
            "trajectories": store_report.trajectories_loaded,
            "trajectories_dropped_on_load": store_report.trajectories_dropped_on_load,
            "directory": str(store_dir),
        },
        "restored_trajectories_count": len(restored),
        "restored_trajectories": len(restored),
    }
    out = args.json_out or (REPORTS_DIR / "stage46_muzero_replay_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload_json, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
