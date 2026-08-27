"""Shared CLI helpers for training launchers (Phase 3).

Provides run-directory naming, seed handling, YAML/config overrides, and the
multi-seed training driver used by both ``train_sac`` and ``train_ppo``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from forexmind.training.config import ExperimentConfig
from forexmind.training.ppo import PPOTrainer
from forexmind.training.sac import SACTrainer

TRAINER_REGISTRY = {
    "sac": SACTrainer,
    "ppo": PPOTrainer,
}


def add_common_args(parser: argparse.ArgumentParser, algorithm: str) -> None:
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML experiment config (overrides defaults).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="One or more seeds (each runs an independent training).")
    parser.add_argument("--run-id", type=str, default="",
                        help="Short label embedded in the run directory name.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override environment worker processes.")
    parser.add_argument("--total-env-steps", type=int, default=None,
                        help="Override total environment steps.")
    parser.add_argument("--backend", type=str, default=None,
                        help="Override collect backend ('sync' | 'process').")
    parser.add_argument("--run-root", type=str, default=None,
                        help="Root directory for runs (default: config run_dir).")
    parser.add_argument("--algorithm", type=str, default=algorithm,
                        help="Algorithm name (sac|ppo); must match module.")


def load_config(args: argparse.Namespace, algorithm: str) -> ExperimentConfig:
    if args.config:
        cfg = ExperimentConfig.from_yaml(args.config)
    else:
        from forexmind.training.config import default_config

        cfg = default_config(algorithm)
    if cfg.algorithm.name != algorithm:
        cfg = replace(cfg, algorithm=type(cfg.algorithm)(name=algorithm))
    if args.workers is not None:
        cfg = replace(cfg, compute=replace(cfg.compute, num_workers=args.workers))
    if args.total_env_steps is not None:
        cfg = replace(cfg, training=replace(cfg.training, total_env_steps=args.total_env_steps))
    if args.backend:
        cfg = replace(cfg, compute=replace(cfg.compute, collect_backend=args.backend))
    return cfg


def run_dir_for(config: ExperimentConfig, seed: int, run_root: str | None = None) -> Path:
    root = Path(run_root) if run_root else Path(config.logging.run_dir)
    label = config.run_id or "run"
    return root / f"{config.algorithm.name}_{label}_seed{seed}"


def train_one(
    config: ExperimentConfig,
    seed: int,
    *,
    resume: str | None = None,
    run_root: str | None = None,
) -> dict[str, object]:
    """Run one training and persist its summary as ``training_summary.json``."""
    cfg = replace(config, compute=replace(config.compute, seed=seed))
    run_dir = run_dir_for(cfg, seed, run_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")

    trainer_cls = TRAINER_REGISTRY[cfg.algorithm.name]
    trainer = trainer_cls(cfg, run_dir)
    summary = trainer.train(resume=resume)
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    summary["run_dir"] = str(run_dir)
    summary["checkpoints"] = sorted(str(p) for p in (run_dir / "checkpoints").glob("*.pt"))
    print(f"\nRun directory : {run_dir}")
    print(f"Checkpoints   : {summary['checkpoints']}")
    return summary


def run_multiseed(
    args: argparse.Namespace, algorithm: str
) -> list[dict[str, object]]:
    config = load_config(args, algorithm)
    seeds = args.seeds if args.seeds else [config.compute.seed]
    summaries: list[dict[str, object]] = []
    t0 = time.perf_counter()
    for i, seed in enumerate(seeds):
        print(f"\n>>> {algorithm.upper()} training | seed {seed} "
              f"({i + 1}/{len(seeds)})")
        summary = train_one(config, seed, resume=args.resume, run_root=args.run_root)
        summaries.append(summary)
        print(json.dumps(summary, indent=2, default=str))
    print(f"\nAll {len(seeds)} seed(s) finished in {time.perf_counter() - t0:.1f}s")
    return summaries
