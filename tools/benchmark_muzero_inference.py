"""Stage 4.1 MuZero inference benchmark and parameter report.

Builds the MuZero core network, prints the parameter-count breakdown, then
measures ``initial_inference`` / ``recurrent_inference`` throughput for a range
of batch sizes.  This establishes the baseline that Stage 4.2 MCTS will call
``recurrent_inference`` against; nothing here trains or uses the environment.

Usage (from the repository root)::

    python -m tools.benchmark_muzero_inference
    python -m tools.benchmark_muzero_inference --batch-sizes 1 16 64 256 --iters 200
    python -m tools.benchmark_muzero_inference --scalar
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import torch
from forexmind.muzero import MuZeroConfig, build_muzero_network, observation_dim

from tools.common import REPORTS_DIR


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _time_call(fn: Callable[[], object], *, iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _synchronize(device)
    return time.perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--iters", type=int, default=100, help="timed iterations per batch size")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-embedding-dim", type=int, default=16)
    parser.add_argument("--support-size", type=int, default=21)
    parser.add_argument("--scalar", action="store_true", help="use scalar value/reward heads")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)

    config = MuZeroConfig(
        obs_dim=observation_dim(),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        action_embedding_dim=args.action_embedding_dim,
        use_support=not args.scalar,
        value_support_size=args.support_size,
        reward_support_size=args.support_size,
    )
    model = build_muzero_network(config).to(device)
    model.eval()
    report = model.parameter_report()

    print("=" * 72)
    print("MuZero Stage 4.1 — parameter report and inference throughput")
    print("=" * 72)
    print(f"device            : {device} ({platform.machine()}, torch {torch.__version__})")
    print(f"obs_dim           : {config.obs_dim}")
    print(f"num_actions       : {config.num_actions}")
    print(f"latent_dim        : {config.latent_dim}")
    print(f"hidden_dim        : {config.hidden_dim} x {config.num_layers} layers")
    print(f"action_embed_dim  : {config.action_embedding_dim}")
    support = (
        f"categorical (value={config.value_support_size}, reward={config.reward_support_size})"
        if config.use_support
        else "scalar regression heads"
    )
    print(f"value/reward head : {support}")
    print("-" * 72)
    for name in ("representation", "dynamics", "prediction", "total"):
        print(f"parameters {name:>14}: {report[name]:>9,}")
    print("-" * 72)

    results: dict[str, dict[int, dict[str, float]]] = {
        "initial_inference": {},
        "recurrent_inference": {},
    }
    header = (
        f"{'batch':>6} | {'initial calls/s':>16} | {'initial states/s':>17} | "
        f"{'recurrent calls/s':>18} | {'recurrent states/s':>19}"
    )
    print(header)
    print("-" * len(header))

    for batch in args.batch_sizes:
        obs = torch.randn(batch, config.obs_dim, device=device)
        latent = torch.randn(batch, config.latent_dim, device=device)
        action = torch.randint(0, config.num_actions, (batch,), device=device)
        mask = torch.ones(batch, config.num_actions, dtype=torch.bool, device=device)

        with torch.inference_mode():
            initial_s = _time_call(
                partial(model.initial_inference, obs, mask),
                iters=args.iters,
                warmup=args.warmup,
                device=device,
            )
            recurrent_s = _time_call(
                partial(model.recurrent_inference, latent, action, mask),
                iters=args.iters,
                warmup=args.warmup,
                device=device,
            )
        initial_calls = args.iters / initial_s
        recurrent_calls = args.iters / recurrent_s
        results["initial_inference"][batch] = {
            "calls_per_sec": initial_calls,
            "states_per_sec": initial_calls * batch,
            "seconds_per_call": initial_s / args.iters,
        }
        results["recurrent_inference"][batch] = {
            "calls_per_sec": recurrent_calls,
            "states_per_sec": recurrent_calls * batch,
            "seconds_per_call": recurrent_s / args.iters,
        }
        print(
            f"{batch:>6} | {initial_calls:>16,.1f} | {initial_calls * batch:>17,.1f} | "
            f"{recurrent_calls:>18,.1f} | {recurrent_calls * batch:>19,.1f}"
        )

    payload = {
        "config": config.to_dict(),
        "parameters": report,
        "device": str(device),
        "torch_version": torch.__version__,
        "iters": args.iters,
        "warmup": args.warmup,
        "results": results,
    }
    out = args.json_out
    if out is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / "stage41_muzero_inference_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("-" * 72)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
