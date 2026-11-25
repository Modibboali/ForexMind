"""Inspect ForexMind training runtime and worker process state.

Example:
    python -m tools.inspect_runtime --config configs/sac_cpu.yaml --workers 204 \
        --probe-steps 204
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from forexmind.episodes.config import EpisodeConfig
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.collector import ProcessCollector
from forexmind.training.config import ExperimentConfig, default_config
from forexmind.training.data import (
    DEFAULT_INSTRUMENT_ORDER,
    DEFAULT_PROCESSED_DIR,
)
from forexmind.training.runtime_diagnostics import print_process_tree_report
from forexmind.training.trainer import build_env_config


def _load_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(args.config) if args.config else default_config("sac")
    if args.workers is not None:
        cfg = replace(cfg, compute=replace(cfg.compute, num_workers=args.workers))
    if args.backend is not None:
        cfg = replace(cfg, compute=replace(cfg.compute, collect_backend=args.backend))
    return cfg


def _resolved_workers(cfg: ExperimentConfig) -> int:
    """Concrete worker count, resolving ``num_workers: auto`` from RAM."""
    if cfg.compute.num_workers != "auto":
        return int(cfg.compute.num_workers)
    from forexmind.training.memory_planning import estimate_auto_worker_count

    est_worker_mb = 400.0 if cfg.compute.dataset_backend == "mmap" else 4500.0
    return estimate_auto_worker_count(worker_rss_mb=est_worker_mb)


def _build_process_collector(cfg: ExperimentConfig) -> ProcessCollector:
    from forexmind.training.dataset_mmap import resolve_dataset

    instruments = (
        tuple(cfg.environment.instruments)
        if cfg.environment.instruments
        else DEFAULT_INSTRUMENT_ORDER
    )
    dataset, _backend = resolve_dataset(
        processed_dir=DEFAULT_PROCESSED_DIR,
        split_config=None,
        instruments=instruments,
        backend=cfg.compute.dataset_backend,
    )
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
    return ProcessCollector(
        processed_dir=str(DEFAULT_PROCESSED_DIR),
        split_config=dataset.split_config,
        instruments=dataset.instruments,
        env_config=env_config,
        encoder_config=encoder.config,
        window_config=window_config,
        episode_config=episode_config,
        algorithm=cfg.algorithm.name,
        model=cfg.model,
        obs_dim=encoder.config.spec.encoded_shape[0],
        action_dim=1,
        global_seed=cfg.compute.seed,
        num_workers=_resolved_workers(cfg),
        dataset_backend=cfg.compute.dataset_backend,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ForexMind runtime diagnostics.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--backend", choices=("sync", "process"), default=None)
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=0,
        help="Optional tiny collection probe after worker startup.",
    )
    parser.add_argument("--sample-seconds", type=float, default=1.0, help="CPU sampling window.")
    args = parser.parse_args()

    cfg = _load_config(args)
    collector: ProcessCollector | None = None
    worker_pids: list[int] = []
    workers = _resolved_workers(cfg)
    try:
        if cfg.compute.collect_backend == "process":
            collector = _build_process_collector(cfg)
            worker_pids = collector.worker_pids
            if args.probe_steps > 0:
                batch = collector.collect(args.probe_steps, random_action=True)
                producers = len({t.worker_id for t in batch})
                print(
                    f"Probe transitions: {len(batch)} "
                    f"(configured_workers={collector.num_workers}, "
                    f"worker_ids_seen={producers})"
                )
        print_process_tree_report(
            worker_pids=worker_pids,
            workers_configured=workers if cfg.compute.collect_backend == "process" else 0,
            sample_seconds=args.sample_seconds,
        )
    finally:
        if collector is not None:
            collector.close()


if __name__ == "__main__":
    main()
