"""Integrated MuZero training launcher (Stage 4.5, scaled in Stage 4.6).

Usage (from the repository root)::

    python -m forexmind.muzero.train_muzero --max-env-steps 4096
    python -m forexmind.muzero.train_muzero --max-env-steps 200000 \
        --num-simulations 32 --batch-size 64 --output-dir runs/muzero
    python -m forexmind.muzero.train_muzero --resume runs/muzero/latest.pt
    python -m forexmind.muzero.train_muzero --max-env-steps 200000 \
        --num-collectors 16 --collectors-per-worker 2 --inference-mode server

Everything economic is frozen: the same processed dataset, the same TRAIN /
VALIDATION / TEST splits, the same ten-action environment and the same
``r_t = log(equity[t+1] / equity[t])`` reward that PPO uses.  MuZero only
projects that environment onto its own six actions.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

import torch

from forexmind.config import (
    EnvironmentConfig,
    ExecutionConfig,
    MarginConfig,
    PositionSizingConfig,
)
from forexmind.muzero.trainer import MuZeroTrainer, MuZeroTrainingConfig
from forexmind.observation.encoder import EncoderConfig
from forexmind.training.data import DEFAULT_PROCESSED_DIR
from forexmind.training.dataset_mmap import resolve_dataset

DEFAULT_INSTRUMENTS = ("EURUSD", "GBPUSD", "USDJPY")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MuZero on ForexMind.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--instruments", nargs="+", default=list(DEFAULT_INSTRUMENTS))
    parser.add_argument("--output-dir", type=Path, default=Path("data/reports/muzero"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # collection / search
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--num-simulations", type=int, default=16)
    parser.add_argument("--trajectories-per-iteration", type=int, default=2)
    parser.add_argument(
        "--temperature-schedule", default="constant", choices=["constant", "linear"]
    )
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.25)
    parser.add_argument("--temperature-decay-steps", type=int, default=20_000)

    # warm-up / ratio
    parser.add_argument("--min-replay-transitions-before-training", type=int, default=64)
    parser.add_argument("--learner-updates-per-iteration", type=int, default=4)

    # learner
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--td-steps", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--latent-gradient-scale", type=float, default=0.5)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)

    # model
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--support-size", type=int, default=21)
    parser.add_argument("--scalar", action="store_true", help="scalar value/reward heads")

    # replay
    parser.add_argument("--max-trajectories", type=int, default=128)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument(
        "--batch-backend",
        default="vectorized",
        choices=["vectorized", "reference"],
        help="replay target construction backend (Stage 4.7)",
    )

    # validation / checkpointing
    parser.add_argument("--max-env-steps", type=int, default=2_048)
    parser.add_argument("--eval-every-env-steps", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-horizon", type=int, default=512)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--checkpoint-every-env-steps", type=int, default=1_024)
    parser.add_argument("--progress-every-iterations", type=int, default=1)

    # evaluation tiers (Stage 4.7)
    parser.add_argument(
        "--full-eval-every-env-steps",
        type=int,
        default=0,
        help="Tier B (full) validation interval; 0 disables it during training",
    )
    parser.add_argument("--full-eval-episodes", type=int, default=100)
    parser.add_argument("--full-eval-horizon", type=int, default=512)
    parser.add_argument("--full-eval-seed", type=int, default=4_242)
    parser.add_argument(
        "--full-eval-at-end",
        action="store_true",
        help="run one full validation when training finishes (Stage 4.7 S35)",
    )
    parser.add_argument("--training-log-name", default="training_log")

    # parallel collection / batched inference (Stage 4.6)
    parser.add_argument("--num-collectors", type=int, default=1)
    parser.add_argument("--collectors-per-worker", type=int, default=2)
    parser.add_argument("--inference-mode", default="server", choices=["server", "local"])
    parser.add_argument("--inference-device", default=None)
    parser.add_argument("--max-inference-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--trajectory-queue-size", type=int, default=8)
    parser.add_argument("--sync-every-learner-updates", type=int, default=1)
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument("--dataset-backend", default="auto", choices=["auto", "parquet", "mmap"])
    parser.add_argument("--target-updates-per-env-step", type=float, default=None)
    parser.add_argument("--profile", action="store_true", help="enable phase profiling (S2)")

    # environment (unchanged economics)
    parser.add_argument("--spread", type=float, default=0.0002)
    parser.add_argument("--leverage", type=float, default=50.0)
    return parser


def build_config(args: argparse.Namespace) -> MuZeroTrainingConfig:
    return MuZeroTrainingConfig(
        instruments=tuple(args.instruments),
        horizon=args.horizon,
        num_simulations=args.num_simulations,
        trajectories_per_iteration=args.trajectories_per_iteration,
        temperature_schedule=args.temperature_schedule,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        temperature_decay_steps=args.temperature_decay_steps,
        min_replay_transitions_before_training=args.min_replay_transitions_before_training,
        learner_updates_per_iteration=args.learner_updates_per_iteration,
        batch_size=args.batch_size,
        unroll_steps=args.unroll_steps,
        td_steps=args.td_steps,
        discount=args.discount,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        latent_gradient_scale=args.latent_gradient_scale,
        policy_loss_weight=args.policy_loss_weight,
        value_loss_weight=args.value_loss_weight,
        reward_loss_weight=args.reward_loss_weight,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        support_size=args.support_size,
        use_support=not args.scalar,
        max_trajectories=args.max_trajectories,
        max_transitions=args.max_transitions,
        batch_backend=args.batch_backend,
        max_env_steps=args.max_env_steps,
        eval_every_env_steps=args.eval_every_env_steps,
        eval_episodes=args.eval_episodes,
        eval_horizon=args.eval_horizon,
        eval_seed=args.eval_seed,
        checkpoint_every_env_steps=args.checkpoint_every_env_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        progress_every_iterations=args.progress_every_iterations,
        num_collectors=args.num_collectors,
        collectors_per_worker=args.collectors_per_worker,
        inference_mode=args.inference_mode,
        inference_device=args.inference_device,
        max_inference_batch_size=args.max_inference_batch_size,
        max_batch_wait_ms=args.max_batch_wait_ms,
        trajectory_queue_size=args.trajectory_queue_size,
        sync_every_learner_updates=args.sync_every_learner_updates,
        torch_threads_per_worker=args.torch_threads_per_worker,
        dataset_backend=args.dataset_backend,
        processed_dir=args.processed_dir,
        target_updates_per_env_step=args.target_updates_per_env_step,
        profile=args.profile,
        full_eval_every_env_steps=args.full_eval_every_env_steps,
        full_eval_episodes=args.full_eval_episodes,
        full_eval_horizon=args.full_eval_horizon,
        full_eval_seed=args.full_eval_seed,
        full_eval_at_end=args.full_eval_at_end,
        training_log_name=args.training_log_name,
    )


def _env_config(spread: float, leverage: float) -> EnvironmentConfig:
    return EnvironmentConfig(
        execution=ExecutionConfig(spread_mode="fixed", spread_value=spread),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal(str(leverage))),
        sizing=PositionSizingConfig(mode="equity_fraction"),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = build_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, dataset_backend = resolve_dataset(
        processed_dir=args.processed_dir,
        instruments=tuple(args.instruments),
        backend=args.dataset_backend,
    )
    trainer = MuZeroTrainer(
        dataset,
        _env_config(args.spread, args.leverage),
        EncoderConfig(context_length=args.context_length),
        config,
        device=device,
    )
    mode = (
        f"parallel {config.num_workers}w x {config.collectors_per_worker}c "
        f"({config.inference_mode} inference)"
        if config.parallel_collection
        else "single process"
    )
    print(
        f"MuZero integrated training | device={device} | dataset={dataset_backend} | "
        f"collection={mode} | {trainer.model.parameter_report()}"
    )
    if args.resume is not None:
        print(f"resuming from {args.resume}: {trainer.resume(args.resume)}")
    report = trainer.train()
    path = trainer.write_report()
    print(f"report written to {path}")
    print(f"training log written to {trainer.write_training_log()}")
    print(f"final: {report['counters']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
