"""Stage 4.6 phase profiler: where the wall time actually goes (brief S2, S45).

Runs the integrated MuZero loop with phase instrumentation enabled and prints
the wall-time breakdown that has to be measured *before* any optimisation::

    environment stepping / observation encoding
    initial inference / recurrent inference / MCTS tree logic
    replay insertion / replay sampling / learner forward+backward
    validation / checkpointing

The same tool profiles the single-process (Stage 4.5) and the parallel
(Stage 4.6) collector so the two can be compared directly.

Usage (repository root)::

    python -m tools.profile_muzero_phases --max-env-steps 512
    python -m tools.profile_muzero_phases --max-env-steps 512 --num-collectors 8
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from forexmind.muzero.train_muzero import _env_config, build_config, build_parser
from forexmind.muzero.trainer import MuZeroTrainer
from forexmind.observation.encoder import EncoderConfig
from forexmind.training.dataset_mmap import resolve_dataset

from tools.common import REPORTS_DIR


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    args.profile = True
    config = build_config(args)

    dataset, dataset_backend = resolve_dataset(
        processed_dir=args.processed_dir,
        instruments=tuple(args.instruments),
        backend=args.dataset_backend,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = MuZeroTrainer(
        dataset,
        _env_config(args.spread, args.leverage),
        EncoderConfig(context_length=args.context_length),
        config,
        device=device,
    )
    print("=" * 78)
    print("MuZero phase profile")
    print("=" * 78)
    print(f"dataset backend     : {dataset_backend}")
    print(f"device              : {device}")
    print(f"max env steps       : {config.max_env_steps}")
    print(f"simulations         : {config.num_simulations}")
    print(f"horizon             : {config.horizon}")
    print(
        "collection          : "
        + (
            f"parallel {config.num_workers}w x {config.collectors_per_worker}c "
            f"({config.inference_mode})"
            if config.parallel_collection
            else "single process"
        )
    )
    print("-" * 78)
    report = trainer.train()

    phases = report["profiling"]["phases"]
    total = report["profiling"]["total_seconds"]
    print("-" * 78)
    print(f"{'phase':<28} | {'seconds':>9} | {'share':>7} | {'calls':>8} | {'ms/call':>9}")
    print("-" * 78)
    for name, payload in phases.items():
        print(
            f"{name:<28} | {payload['seconds']:>9.3f} | {100.0 * payload['fraction']:>6.1f}% | "
            f"{payload['calls']:>8} | {payload['ms_per_call']:>9.3f}"
        )
    print("-" * 78)
    print(f"instrumented total: {total:.3f} s")
    timing = report["timing"]
    print(
        f"wrapper total     : {timing['total_seconds']:.3f} s "
        f"(collection {100 * timing['collection_fraction']:.1f}%, "
        f"learning {100 * timing['learning_fraction']:.1f}%, "
        f"evaluation {100 * timing['evaluation_fraction']:.1f}%)"
    )
    print(f"counters          : {report['counters']}")
    print(f"throughput        : {json.dumps(report['throughput'], indent=2)}")
    collection = report["collection"]
    if collection["mode"] == "parallel":
        print(f"inference batches : {json.dumps(collection['inference'], indent=2)}")

    destination = args.json_out
    if destination is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        destination = REPORTS_DIR / "stage46_muzero_phase_profile.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "profiling": report["profiling"],
                "timing": timing,
                "throughput": report["throughput"],
                "counters": report["counters"],
                "collection": collection,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
