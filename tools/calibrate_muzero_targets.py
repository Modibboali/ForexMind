"""Stage 4.4 MuZero target-scale calibration on the real processed dataset (§10).

Collects a small TRAIN replay with the Stage 4.3 collector, then reports the
observed reward / n-step value target distributions, the proposed support scales
:func:`~forexmind.muzero.calibration.propose_scale` derives, and how much of the
window the *currently configured* scales actually use.

This is an inspection tool: it never trains and never mutates the model.

Usage (from the repository root)::

    python -m tools.calibrate_muzero_targets
    python -m tools.calibrate_muzero_targets --trajectories 6 --horizon 24
    python -m tools.calibrate_muzero_targets --target-u 0.7 --value-scale 0.5
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
    CollectorConfig,
    MuZeroCollector,
    MuZeroConfig,
    ReplayConfig,
    TargetConfig,
    TrajectoryReplayBuffer,
    build_muzero_network,
)
from forexmind.muzero.calibration import (
    calibration_report,
    reward_targets,
    value_targets,
)
from forexmind.muzero.support import (
    SUPPORT_RANGE,
    saturation_fraction,
    transform_to_scalar,
)
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


def _fmt(value: float) -> str:
    return f"{value:+.6e}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--td-steps", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--target-u", type=float, default=0.6)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--value-support-size", type=int, default=21)
    parser.add_argument("--reward-support-size", type=int, default=21)
    parser.add_argument(
        "--value-scale",
        type=float,
        default=1.0,
        help="scale currently configured on the model (for the saturation comparison)",
    )
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--spread", type=float, default=0.0002)
    parser.add_argument("--leverage", type=float, default=50.0)
    parser.add_argument("--instruments", nargs="+", default=list(DEFAULT_INSTRUMENTS))
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = make_training_dataset(args.processed_dir, instruments=tuple(args.instruments))
    encoder_config = EncoderConfig(context_length=args.context_length)
    model = build_muzero_network(
        MuZeroConfig(
            obs_dim=encoder_config.spec.encoded_shape[0],
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            value_support_size=args.value_support_size,
            reward_support_size=args.reward_support_size,
            value_scale=args.value_scale,
            reward_scale=args.reward_scale,
        )
    ).to(device)
    model.eval()

    collector = MuZeroCollector(
        dataset,
        _env_config(args.spread, args.leverage),
        encoder_config,
        model,
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
    print("MuZero Stage 4.4 — target-scale calibration (real TRAIN replay)")
    print("=" * 78)
    print(f"device            : {device} ({platform.machine()}, torch {torch.__version__})")
    print(f"instruments       : {', '.join(args.instruments)}")
    print(f"obs_dim           : {model.config.obs_dim}")
    print(f"trajectories      : {args.trajectories} x horizon {args.horizon}")
    print(f"mcts simulations  : {args.simulations}")
    print(f"target_u          : {args.target_u}")
    print(f"support range     : +/-{SUPPORT_RANGE:.0f} x scale")
    print("-" * 78)

    replay = TrajectoryReplayBuffer(
        ReplayConfig(max_trajectories=max(8, args.trajectories), seed=args.seed)
    )
    start = time.perf_counter()
    collector.collect_into(replay, args.trajectories)
    elapsed = time.perf_counter() - start

    target_config = TargetConfig(
        num_unroll_steps=args.unroll_steps,
        td_steps=args.td_steps,
        discount=args.discount,
    )
    # a single big batch so every valid target in the replay is represented
    batch = replay.sample(64, target_config=target_config, rng=np.random.default_rng(args.seed))
    report = calibration_report(batch, target_u=args.target_u)

    print(f"collector wall    : {elapsed:.1f}s")
    print(f"valid rewards     : {report['reward']['statistics']['count']}")
    print(f"valid values      : {report['value']['statistics']['count']}")
    print("-" * 78)
    header = (
        f"{'target':<8} | {'mean':>13} | {'std':>11} | {'p01':>13} | {'p50':>13} | "
        f"{'p99':>13} | {'|p99|':>13} | {'max':>13}"
    )
    print(header)
    print("-" * len(header))
    for name in ("reward", "value"):
        stats = report[name]["statistics"]
        print(
            f"{name:<8} | {_fmt(stats['mean'])} | {stats['std']:>11.4e} | "
            f"{_fmt(stats['p01'])} | {_fmt(stats['p50'])} | {_fmt(stats['p99'])} | "
            f"{stats['abs_p99']:>13.4e} | {_fmt(stats['maximum'])}"
        )
    print("-" * len(header))

    print("proposed scales")
    for name in ("reward", "value"):
        proposed = report[name]["proposed_scale"]
        print(
            f"  {name:<6} scale={proposed:.6e}  "
            f"saturation={report[name]['saturation_fraction']:.2%}  "
            f"|z|/scale at p99={report[name]['statistics']['abs_p99'] / proposed:.3f}"
        )
    print(f"  headroom factor h^-1({args.target_u}) = {report['headroom_factor']:.4f}")

    configured = {"reward": args.reward_scale, "value": args.value_scale}
    print("configured scales (what the model currently uses)")
    configured_saturation: dict[str, float] = {}
    for name, scale in configured.items():
        sat = saturation_fraction(
            (batch.target_rewards if name == "reward" else batch.target_values).reshape(-1),
            scale=scale,
        )
        configured_saturation[name] = float(sat)
        print(f"  {name:<6} scale={scale:.6e}  saturation={sat:.2%}")
    print("-" * 78)

    # per-unroll-step detail: rewards shrink into the horizon, values do not
    per_step: dict[str, dict[str, float]] = {"reward": {}, "value": {}}
    for step in range(args.unroll_steps + 1):
        if step < args.unroll_steps:
            rewards = (
                batch.target_rewards[:, step][batch.reward_masks[:, step] > 0.5].detach().numpy()
            )
            per_step["reward"][str(step)] = float(np.abs(rewards).mean()) if rewards.size else 0.0
        values = batch.target_values[:, step][batch.value_masks[:, step] > 0.5].detach().numpy()
        per_step["value"][str(step)] = float(np.abs(values).mean()) if values.size else 0.0
    print("mean |target| per unroll step")
    print("  step : " + "  ".join(f"{step:>11}" for step in range(args.unroll_steps + 1)))
    print(
        "  rwd  : "
        + "  ".join(
            f"{per_step['reward'].get(str(s), 0.0):>11.4e}" for s in range(args.unroll_steps)
        )
    )
    print(
        "  val  : "
        + "  ".join(f"{per_step['value'][str(s)]:>11.4e}" for s in range(args.unroll_steps + 1))
    )
    print("-" * 78)

    # how many bins of the proposed window are actually used
    bin_usage: dict[str, Any] = {}
    for name in ("reward", "value"):
        scale = report[name]["proposed_scale"]
        values = (
            reward_targets(batch).tolist() if name == "reward" else value_targets(batch).tolist()
        )
        encoded = transform_to_scalar(torch.tensor(values, dtype=torch.float64) / scale).numpy()
        size = args.reward_support_size if name == "reward" else args.value_support_size
        half = (size - 1) // 2
        bins = np.clip(np.round(encoded * half).astype(int), -half, half)
        occupancy = np.bincount(bins + half, minlength=size)
        used = int((occupancy > 0).sum())
        bin_usage[name] = {
            "support_size": int(size),
            "bins_used": used,
            "bins_used_fraction": used / size,
            "centre_bin_fraction": float(occupancy[half] / max(1, occupancy.sum())),
            "occupancy": occupancy.tolist(),
        }
        print(
            f"{name:<6} support bins used : {used}/{size} ({used / size:.1%}), "
            f"centre bin {bin_usage[name]['centre_bin_fraction']:.1%}"
        )
    print("=" * 78)

    payload: dict[str, Any] = {
        "config": {
            "instruments": args.instruments,
            "trajectories": args.trajectories,
            "horizon": args.horizon,
            "num_simulations": args.simulations,
            "unroll_steps": args.unroll_steps,
            "td_steps": args.td_steps,
            "discount": args.discount,
            "target_u": args.target_u,
            "seed": args.seed,
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "obs_dim": model.config.obs_dim,
        "report": report,
        "configured_scales": configured,
        "configured_saturation": configured_saturation,
        "mean_abs_target_per_step": per_step,
        "support_bin_usage": bin_usage,
        "collector_stats": collector.stats.to_dict(),
        "collector_wall_clock_seconds": elapsed,
    }
    out = args.json_out
    if out is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / "stage44_muzero_target_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
