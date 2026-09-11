"""Stage 4.3 MuZero trajectory collection on the real processed dataset.

Collects a few short TRAIN trajectories with an untrained MuZero model and a
small MCTS budget, then reports collection statistics, replay memory, sampling
diagnostics, and an example training batch.  This is an integration check, not
a performance measurement: the network has not been trained.

Usage (from the repository root)::

    python -m tools.collect_muzero_trajectories
    python -m tools.collect_muzero_trajectories --trajectories 4 --horizon 16 --simulations 8
    python -m tools.collect_muzero_trajectories --split validation --training off
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
    MuZeroCollector,
    MuZeroConfig,
    ReplayConfig,
    TargetConfig,
    TrajectoryReplayBuffer,
    build_muzero_network,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--td-steps", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--training", default="on", choices=["on", "off"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-trajectories", type=int, default=32)
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
        )
    ).to(device)
    model.eval()

    collector = MuZeroCollector(
        dataset,
        _env_config(args.spread, args.leverage),
        encoder_config,
        model,
        CollectorConfig(
            split=args.split,
            horizon=args.horizon,
            num_simulations=args.simulations,
            discount=args.discount,
            training=args.training == "on",
            temperature=args.temperature,
            seed=args.seed,
        ),
    )

    print("=" * 78)
    print("MuZero Stage 4.3 - real trajectory collection (untrained model)")
    print("=" * 78)
    print(f"device            : {device} ({platform.machine()}, torch {torch.__version__})")
    print(f"instruments       : {', '.join(args.instruments)}")
    print(f"split             : {args.split}")
    print(f"obs_dim           : {model.config.obs_dim}")
    print(f"latent_dim        : {model.config.latent_dim}")
    print(f"training search   : {args.training} (temperature={args.temperature})")
    print(f"mcts simulations  : {args.simulations}")
    print(f"horizon           : {args.horizon} decisions per trajectory")
    print(f"model fingerprint : {collector.model_version}")
    print("-" * 78)

    replay = TrajectoryReplayBuffer(ReplayConfig(max_trajectories=args.max_trajectories))
    start = time.perf_counter()
    collector.collect_into(replay, args.trajectories)
    elapsed = time.perf_counter() - start

    stats = collector.stats.to_dict()
    print(f"trajectories      : {stats['trajectories']}")
    print(f"environment steps : {stats['env_steps']}")
    print(f"mean length       : {stats['mean_length']:.1f}")
    print(f"terminated/trunc. : {stats['terminated']}/{stats['truncated']}")
    print(f"MCTS calls        : {stats['searches']}")
    print(f"recurrent calls   : {stats['recurrent_inference_calls']}")
    print(f"mean reward       : {stats['mean_reward']:+.8f}")
    print(f"reward sum        : {stats['reward_sum']:+.8f}")
    print(f"wall clock        : {elapsed:.1f}s")
    print("-" * 78)
    frequencies = stats["action_frequencies"]
    for index, name in enumerate(MUZERO_ACTION_NAMES):
        print(f"  {index} {name:10} {stats['action_counts'][index]:>6}  {frequencies[index]:6.2%}")
    print("-" * 78)

    memory = replay.memory_report()
    print(f"replay memory     : {memory['total_mb']:.4f} MB")
    print(f"bytes/transition  : {memory['bytes_per_transition']:.1f}")
    print(f"capacity          : {memory['num_trajectories']}/{memory['max_trajectories']}")

    sampling = replay.sampling_diagnostics()
    print(
        "action mix        : "
        f"hold={sampling['actions']['pct_hold']:.1%} "
        f"flat={sampling['actions']['pct_flat']:.1%} "
        f"short={sampling['actions']['pct_short']:.1%} "
        f"long={sampling['actions']['pct_long']:.1%}"
    )
    print(
        "decision mix      : "
        f"hold={sampling['events']['pct_hold']:.1%} "
        f"entry={sampling['events']['pct_entry']:.1%} "
        f"exit={sampling['events']['pct_exit']:.1%} "
        f"resize={sampling['events']['pct_resize']:.1%}"
    )

    target_config = TargetConfig(
        num_unroll_steps=args.unroll_steps,
        td_steps=args.td_steps,
        discount=args.discount,
    )
    batch = replay.sample(
        args.batch_size, target_config=target_config, rng=np.random.default_rng(args.seed)
    )
    print("-" * 78)
    print("example training batch")
    for name, shape in batch.shapes().items():
        print(f"  {name:16} {shape}")
    finite = all(
        torch.isfinite(tensor).all()
        for tensor in (
            batch.observation,
            batch.target_rewards,
            batch.target_values,
            batch.target_policies,
        )
    )
    print(f"  all targets finite : {finite}")

    payload: dict[str, Any] = {
        "config": {
            "instruments": args.instruments,
            "split": args.split,
            "horizon": args.horizon,
            "num_simulations": args.simulations,
            "training": args.training == "on",
            "temperature": args.temperature,
            "discount": args.discount,
            "context_length": args.context_length,
            "obs_dim": model.config.obs_dim,
            "seed": args.seed,
        },
        "model_version": collector.model_version,
        "device": str(device),
        "torch_version": torch.__version__,
        "stats": stats,
        "memory": memory,
        "sampling": sampling,
        "sample_batch_shapes": batch.shapes(),
        "trajectories": [t.to_dict() for t in replay.trajectories],
        "wall_clock_seconds": elapsed,
    }
    out = args.json_out
    if out is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / f"stage43_muzero_collection_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("-" * 78)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
