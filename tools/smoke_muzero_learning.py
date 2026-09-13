"""Stage 4.4 MuZero real-replay learning smoke test (§32).

End-to-end integration check on the *real* processed dataset:

1. collect a small TRAIN replay with the Stage 4.3 collector (untrained network),
2. calibrate the support scales to the observed reward / value targets (§10),
3. run a few hundred joint learner updates on batches sampled from that replay,
4. report the loss curve, the diagnostics, and a checkpoint/resume check.

It is a smoke test, not a training run: the replay is tiny, the model is small,
and the search is shallow.  The point is that the whole Stage 4.4 path works on
real Forex data and that the objective decreases.

Usage (from the repository root)::

    python -m tools.smoke_muzero_learning
    python -m tools.smoke_muzero_learning --updates 300 --batch-size 32
    python -m tools.smoke_muzero_learning --reward-scale 2e-4 --value-scale 0.3
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import torch
from forexmind.config import (
    EnvironmentConfig,
    ExecutionConfig,
    MarginConfig,
    PositionSizingConfig,
)
from forexmind.muzero import (
    MUZERO_ACTION_NAMES,
    CollectorConfig,
    LearnerConfig,
    MuZeroCollector,
    MuZeroConfig,
    MuZeroLearner,
    OptimizerConfig,
    ReplayConfig,
    TargetConfig,
    TrajectoryReplayBuffer,
    build_muzero_network,
)
from forexmind.muzero.calibration import calibration_report
from forexmind.observation.encoder import EncoderConfig
from forexmind.training.data import DEFAULT_PROCESSED_DIR, make_training_dataset

from tools.common import REPORTS_DIR

DEFAULT_INSTRUMENTS = ("EURUSD", "GBPUSD", "USDJPY")


def _env_config(spread: float, leverage: float) -> EnvironmentConfig:
    return EnvironmentConfig(
        execution=ExecutionConfig(spread_mode="fixed", spread_value=spread),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal(str(leverage))),
        sizing=PositionSizingConfig(mode="equity_fraction"),
    )


def _head_mean(history: list[dict[str, float]], key: str, fraction: float = 0.1) -> float:
    """Mean of the first ``fraction`` of a metric's history."""
    values = [row[key] for row in history if key in row]
    if not values:
        return float("nan")
    span = max(1, int(len(values) * fraction))
    return float(np.mean(values[:span]))


def _tail_mean(history: list[dict[str, float]], key: str, count: int = 5) -> float:
    """Mean of the last ``count`` values of a metric's history."""
    values = [row[key] for row in history if key in row]
    if not values:
        return float("nan")
    return float(np.mean(values[-min(count, len(values)) :]))


def _curve(history: list[dict[str, float]], key: str, buckets: int = 10) -> list[float]:
    values = [row[key] for row in history if key in row]
    if not values:
        return []
    chunks = np.array_split(np.asarray(values, dtype=float), min(buckets, len(values)))
    return [float(chunk.mean()) for chunk in chunks]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # collection
    parser.add_argument("--trajectories", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--instruments", nargs="+", default=list(DEFAULT_INSTRUMENTS))
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--context-length", type=int, default=64)
    # targets / model
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--td-steps", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--support-size", type=int, default=21)
    parser.add_argument("--scalar", action="store_true", help="scalar value/reward heads")
    parser.add_argument("--reward-scale", type=float, default=None, help="default: calibrated")
    parser.add_argument("--value-scale", type=float, default=None, help="default: calibrated")
    parser.add_argument("--target-u", type=float, default=0.6)
    # learning
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--latent-gradient-scale", type=float, default=0.5)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = make_training_dataset(args.processed_dir, instruments=tuple(args.instruments))
    encoder_config = EncoderConfig(context_length=args.context_length)
    obs_dim = encoder_config.spec.encoded_shape[0]

    collection_model = build_muzero_network(
        MuZeroConfig(
            obs_dim=obs_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            use_support=not args.scalar,
            value_support_size=args.support_size,
            reward_support_size=args.support_size,
        )
    ).to(device)
    collection_model.eval()

    collector = MuZeroCollector(
        dataset,
        _env_config(0.0002, 50.0),
        encoder_config,
        collection_model,
        CollectorConfig(
            split="train",
            horizon=args.horizon,
            num_simulations=args.simulations,
            discount=args.discount,
            training=True,
            temperature=args.temperature,
            seed=args.seed,
        ),
    )

    print("=" * 78)
    print("MuZero Stage 4.4 — real-replay learning smoke test")
    print("=" * 78)
    print(f"device            : {device} ({platform.machine()}, torch {torch.__version__})")
    print(f"instruments       : {', '.join(args.instruments)}")
    print(f"obs_dim           : {obs_dim}")
    print(f"unroll steps      : {args.unroll_steps} (td_steps={args.td_steps})")
    print(
        f"collection        : {args.trajectories} x horizon {args.horizon}, "
        f"{args.simulations} simulations"
    )
    print("-" * 78)

    target_config = TargetConfig(
        num_unroll_steps=args.unroll_steps,
        td_steps=args.td_steps,
        discount=args.discount,
    )
    replay = TrajectoryReplayBuffer(
        ReplayConfig(max_trajectories=max(8, args.trajectories), seed=args.seed)
    )
    start = time.perf_counter()
    collector.collect_into(replay, args.trajectories)
    collection_seconds = time.perf_counter() - start
    stats = collector.stats.to_dict()
    print(
        f"collected         : {stats['trajectories']} trajectories, "
        f"{stats['env_steps']} env steps in {collection_seconds:.1f}s"
    )
    print(f"mean reward       : {stats['mean_reward']:+.8f}")

    # ---- calibration -------------------------------------------------------
    calibration_batch = replay.sample(
        64, target_config=target_config, rng=np.random.default_rng(args.seed)
    )
    calibration = calibration_report(calibration_batch, target_u=args.target_u)
    reward_scale = (
        args.reward_scale
        if args.reward_scale is not None
        else float(calibration["reward"]["proposed_scale"])
    )
    value_scale = (
        args.value_scale
        if args.value_scale is not None
        else float(calibration["value"]["proposed_scale"])
    )
    print("-" * 78)
    print("target calibration (§10)")
    for name, scale in (("reward", reward_scale), ("value", value_scale)):
        block = calibration[name]
        print(
            f"  {name:<6} |p99|={block['statistics']['abs_p99']:.4e}  "
            f"scale={scale:.4e}  saturation={block['saturation_fraction']:.2%}"
        )
    print("-" * 78)

    # ---- learner -----------------------------------------------------------
    model = build_muzero_network(
        MuZeroConfig(
            obs_dim=obs_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            use_support=not args.scalar,
            value_support_size=args.support_size,
            reward_support_size=args.support_size,
            value_scale=value_scale,
            reward_scale=reward_scale,
        )
    ).to(device)
    learner = MuZeroLearner(
        model,
        LearnerConfig.for_model(
            model.config,
            optimizer=OptimizerConfig(
                name=args.optimizer,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
            ),
            latent_gradient_scale=args.latent_gradient_scale,
            device=str(device),
            seed=args.seed,
            loss_overrides={
                "policy_loss_weight": args.policy_loss_weight,
                "value_loss_weight": args.value_loss_weight,
                "reward_loss_weight": args.reward_loss_weight,
            },
        ),
        replay=replay,
    )

    print(f"parameters        : {learner.parameter_report()['total']:,}")
    print(
        f"optimizer         : {args.optimizer} lr={args.learning_rate} "
        f"wd={args.weight_decay} clip={args.max_grad_norm}"
    )
    print(
        f"latent grad scale : {args.latent_gradient_scale} "
        f"(1.0 = no scaling; damped after every step except the last)"
    )
    print("-" * 78)

    rng = np.random.default_rng(args.seed + 1)
    history: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    for _ in range(args.updates):
        batch = replay.sample(args.batch_size, target_config=target_config, rng=rng)
        history.append(learner.train_step(batch))
    learning_seconds = time.perf_counter() - wall_start

    header = f"{'window':>8} | {'total':>10} | {'policy':>10} | {'value':>10} | {'reward':>10}"
    print(header)
    print("-" * len(header))
    for index, (total, policy, value, reward) in enumerate(
        zip(
            _curve(history, "total_loss", 10),
            _curve(history, "policy_loss", 10),
            _curve(history, "value_loss", 10),
            _curve(history, "reward_loss", 10),
            strict=True,
        )
    ):
        print(f"{index:>8} | {total:>10.6f} | {policy:>10.6f} | {value:>10.6f} | {reward:>10.6f}")
    print("-" * len(header))

    first_total = float(
        np.mean([row["total_loss"] for row in history][: max(1, len(history) // 10)])
    )
    last_total = float(
        np.mean([row["total_loss"] for row in history][-max(1, len(history) // 10) :])
    )
    best = float(np.min([row["total_loss"] for row in history]))
    diagnostics = {
        "total_loss_head_mean": first_total,
        "total_loss_tail_mean": last_total,
        "total_loss_best": best,
        "total_loss_reduction": (first_total - last_total) / first_total if first_total else 0.0,
        "policy_kl_head": _head_mean(history, "policy_kl"),
        "policy_kl_tail": _tail_mean(history, "policy_kl"),
        "value_mae_head": _head_mean(history, "value_mae"),
        "value_mae_tail": _tail_mean(history, "value_mae"),
        "reward_mae_head": _head_mean(history, "reward_mae"),
        "reward_mae_tail": _tail_mean(history, "reward_mae"),
        "gradient_norm_mean": float(np.mean([row["gradient_norm"] for row in history])),
        "gradient_norm_max": float(np.max([row["gradient_norm"] for row in history])),
        "policy_top1_agreement_last": float(history[-1]["policy_top1_agreement"]),
        "pred_argmax_hold_fraction_last": float(history[-1]["pred_argmax_hold_fraction"]),
        "target_argmax_hold_fraction_last": float(history[-1]["target_argmax_hold_fraction"]),
        "pred_hold_prob_last": float(history[-1]["pred_hold_prob"]),
        "target_hold_prob_last": float(history[-1]["target_hold_prob"]),
    }
    print("learning summary")
    print(
        f"  total loss        : head {first_total:.6f} -> tail {last_total:.6f} "
        f"({diagnostics['total_loss_reduction']:+.1%}), best {best:.6f}"
    )
    print(
        f"  policy KL         : {diagnostics['policy_kl_head']:.6f} -> "
        f"{diagnostics['policy_kl_tail']:.6f}"
    )
    print(
        f"  value MAE         : {diagnostics['value_mae_head']:.6f} -> "
        f"{diagnostics['value_mae_tail']:.6f}"
    )
    print(
        f"  reward MAE        : {diagnostics['reward_mae_head']:.6e} -> "
        f"{diagnostics['reward_mae_tail']:.6e}"
    )
    print(
        f"  gradient norm     : mean {diagnostics['gradient_norm_mean']:.4f}, "
        f"max {diagnostics['gradient_norm_max']:.4f}"
    )
    print(f"  policy top1 agree : {diagnostics['policy_top1_agreement_last']:.2%}")
    print("-" * 78)
    print("HOLD diagnostics (§35)")
    print(
        f"  pred   hold prob  : {diagnostics['pred_hold_prob_last']:.2%}  "
        f"argmax {diagnostics['pred_argmax_hold_fraction_last']:.2%}"
    )
    print(
        f"  target hold prob  : {diagnostics['target_hold_prob_last']:.2%}  "
        f"argmax {diagnostics['target_argmax_hold_fraction_last']:.2%}"
    )
    predicted = np.array(
        [history[-1][f"pred_group_{name}"] for name in ("hold", "flat", "short", "long")]
    )
    target = np.array(
        [history[-1][f"target_group_{name}"] for name in ("hold", "flat", "short", "long")]
    )
    print(
        "  groups            : "
        + "  ".join(
            f"{name}={predicted[i]:.1%}/{target[i]:.1%}"
            for i, name in enumerate(("HOLD", "FLAT", "SHORT", "LONG"))
        )
    )
    print("-" * 78)
    print(f"actions           : {', '.join(MUZERO_ACTION_NAMES)}")
    print(
        f"throughput        : {args.updates / learning_seconds:,.1f} updates/s "
        f"({args.updates * args.batch_size / learning_seconds:,.0f} samples/s)"
    )
    print("-" * 78)

    # ---- checkpoint / resume ----------------------------------------------
    probe = replay.sample(
        min(args.batch_size, 8), target_config=target_config, rng=np.random.default_rng(7)
    )
    with torch.no_grad():
        before = float(learner.unroll(probe).policy_logits.sum().item())
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = REPORTS_DIR / "stage44_muzero_smoke_checkpoint.pt"
    learner.save_checkpoint(checkpoint_path, target_config=target_config)

    resumed_model = build_muzero_network(
        MuZeroConfig(
            obs_dim=obs_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            use_support=not args.scalar,
            value_support_size=args.support_size,
            reward_support_size=args.support_size,
            value_scale=value_scale,
            reward_scale=reward_scale,
        )
    )
    resumed = MuZeroLearner(resumed_model, LearnerConfig.for_model(resumed_model.config))
    payload = resumed.load_checkpoint(checkpoint_path)
    with torch.no_grad():
        after = float(resumed.unroll(probe).policy_logits.sum().item())
    resumed_metrics = resumed.train_step(probe)
    print("checkpoint / resume")
    print(f"  path              : {checkpoint_path}")
    print(f"  update_count      : {payload['update_count']}")
    print(
        f"  probe logit sum   : {before:.6f} -> {after:.6f} "
        f"({'identical' if abs(before - after) < 1e-6 else 'MISMATCH'})"
    )
    print(f"  resumed step loss : {resumed_metrics['total_loss']:.6f}")
    print("=" * 78)

    payload_out: dict[str, Any] = {
        "config": {
            "instruments": args.instruments,
            "trajectories": args.trajectories,
            "horizon": args.horizon,
            "num_simulations": args.simulations,
            "unroll_steps": args.unroll_steps,
            "td_steps": args.td_steps,
            "discount": args.discount,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "max_grad_norm": args.max_grad_norm,
            "latent_gradient_scale": args.latent_gradient_scale,
            "loss_weights": {
                "policy": args.policy_loss_weight,
                "value": args.value_loss_weight,
                "reward": args.reward_loss_weight,
            },
            "reward_scale": reward_scale,
            "value_scale": value_scale,
            "support_size": args.support_size,
            "representation": "scalar" if args.scalar else "categorical_support",
            "seed": args.seed,
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "parameters": learner.parameter_report(),
        "collector_stats": stats,
        "collection_seconds": collection_seconds,
        "calibration": calibration,
        "diagnostics": diagnostics,
        "loss_curve": {
            key: _curve(history, key, 10)
            for key in (
                "total_loss",
                "policy_loss",
                "value_loss",
                "reward_loss",
                "policy_kl",
                "value_mae",
                "reward_mae",
            )
        },
        "metrics_summary": learner.metrics_summary(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "probe_before": before,
            "probe_after": after,
            "update_count": int(payload["update_count"]),
            "resumed_step_loss": float(resumed_metrics["total_loss"]),
        },
        "learning_seconds": learning_seconds,
        "updates_per_second": args.updates / learning_seconds,
    }
    out = args.json_out
    if out is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / "stage44_muzero_learning_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
