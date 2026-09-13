"""Integrated MuZero training launcher (Stage 4.5).

Usage (from the repository root)::

    python -m forexmind.muzero.train_muzero --max-env-steps 4096
    python -m forexmind.muzero.train_muzero --max-env-steps 200000 \
        --num-simulations 32 --batch-size 64 --output-dir runs/muzero
    python -m forexmind.muzero.train_muzero --resume runs/muzero/latest.pt

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
from forexmind.training.data import DEFAULT_PROCESSED_DIR, make_training_dataset

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

    # validation / checkpointing
    parser.add_argument("--max-env-steps", type=int, default=2_048)
    parser.add_argument("--eval-every-env-steps", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-horizon", type=int, default=512)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--checkpoint-every-env-steps", type=int, default=1_024)
    parser.add_argument("--progress-every-iterations", type=int, default=1)

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
        max_env_steps=args.max_env_steps,
        eval_every_env_steps=args.eval_every_env_steps,
        eval_episodes=args.eval_episodes,
        eval_horizon=args.eval_horizon,
        eval_seed=args.eval_seed,
        checkpoint_every_env_steps=args.checkpoint_every_env_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        progress_every_iterations=args.progress_every_iterations,
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
    dataset = make_training_dataset(args.processed_dir, instruments=tuple(args.instruments))
    trainer = MuZeroTrainer(
        dataset,
        _env_config(args.spread, args.leverage),
        EncoderConfig(context_length=args.context_length),
        config,
        device=device,
    )
    print(f"MuZero integrated training | device={device} | {trainer.model.parameter_report()}")
    if args.resume is not None:
        print(f"resuming from {args.resume}: {trainer.resume(args.resume)}")
    report = trainer.train()
    path = trainer.write_report()
    print(f"report written to {path}")
    print(f"final: {report['counters']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
