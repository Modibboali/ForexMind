"""Stage 4.2 MuZero MCTS / PUCT search-cost benchmark.

Measures search throughput and search cost for one root observation across a
range of simulation budgets, using the real MuZero network (no environment, no
training).  Nothing here is optimised: the point is to establish the cost curve
that Stage 4.3 trajectory collection will pay.

Usage (from the repository root)::

    python -m tools.benchmark_muzero_search
    python -m tools.benchmark_muzero_search --simulations 16 32 64 128 --repeats 20
    python -m tools.benchmark_muzero_search --latent-dim 128 --hidden-dim 256
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from forexmind.muzero import (
    MUZERO_NUM_ACTIONS,
    MuZeroConfig,
    MuZeroMCTS,
    SearchConfig,
    build_muzero_network,
    observation_dim,
)
from forexmind.muzero.actions import PlanningState

from tools.common import REPORTS_DIR


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_balance(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--repeats", type=int, default=20, help="searches timed per budget")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_muzero_network(
        MuZeroConfig(
            obs_dim=observation_dim(),
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        )
    ).to(device)
    model.eval()

    observation = torch.randn(1, model.config.obs_dim, device=device)
    mask = PlanningState.flat().action_mask()

    print("=" * 78)
    print("MuZero Stage 4.2 — MCTS / PUCT search cost")
    print("=" * 78)
    print(f"device            : {device} ({platform.machine()}, torch {torch.__version__})")
    print(f"obs_dim           : {model.config.obs_dim}")
    print(f"num_actions       : {MUZERO_NUM_ACTIONS}")
    print(f"latent_dim        : {model.config.latent_dim}")
    print(f"hidden_dim        : {model.config.hidden_dim} x {model.config.num_layers} layers")
    print(f"discount          : {args.discount}")
    print(f"repeats/budget    : {args.repeats}")
    print("-" * 78)
    header = (
        f"{'sims':>6} | {'searches/s':>11} | {'sims/s':>10} | {'ms/search':>10} | "
        f"{'recurrent/search':>16} | {'nodes/search':>13} | {'depth':>7}"
    )
    print(header)
    print("-" * len(header))

    results: dict[str, Any] = {}
    for simulations in args.simulations:
        config = SearchConfig(num_simulations=simulations, discount=args.discount).evaluation()
        search = MuZeroMCTS(model, config, rng=np.random.default_rng(args.seed))

        for _ in range(args.warmup):
            search.search(observation, mask)
        _load_balance(device)

        recurrent_calls: list[int] = []
        expanded: list[int] = []
        depths: list[int] = []
        start = time.perf_counter()
        for _ in range(args.repeats):
            result = search.search(observation, mask)
            recurrent_calls.append(result.diagnostics.recurrent_inference_calls)
            expanded.append(result.diagnostics.expanded_nodes)
            depths.append(result.diagnostics.tree_depth)
        _load_balance(device)
        elapsed = time.perf_counter() - start

        searches_per_sec = args.repeats / elapsed
        row = {
            "searches_per_sec": searches_per_sec,
            "simulations_per_sec": searches_per_sec * simulations,
            "ms_per_search": 1000.0 * elapsed / args.repeats,
            "recurrent_calls_per_search": statistics.fmean(recurrent_calls),
            "expanded_nodes_per_search": statistics.fmean(expanded),
            "tree_depth": statistics.fmean(depths),
        }
        results[str(simulations)] = row
        print(
            f"{simulations:>6} | {searches_per_sec:>11,.1f} | "
            f"{searches_per_sec * simulations:>10,.0f} | {row['ms_per_search']:>10,.1f} | "
            f"{row['recurrent_calls_per_search']:>16,.1f} | "
            f"{row['expanded_nodes_per_search']:>13,.1f} | {row['tree_depth']:>7,.2f}"
        )

    payload = {
        "config": {
            "latent_dim": model.config.latent_dim,
            "hidden_dim": model.config.hidden_dim,
            "num_layers": model.config.num_layers,
            "obs_dim": model.config.obs_dim,
            "num_actions": MUZERO_NUM_ACTIONS,
            "discount": args.discount,
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "results": results,
    }
    out = args.json_out
    if out is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / "stage42_muzero_search_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("-" * len(header))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
