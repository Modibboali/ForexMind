"""Stage 4.4 MuZero learner throughput and memory benchmark (§38).

Measures the cost of one joint MuZero update (initial inference + ``K``
recurrent steps + masked loss + backward + optimizer step) across batch sizes,
using synthetic batches with the *real* observation width and network size.  No
environment, no MCTS, no replay: this isolates the learner.

Reports, per batch size: updates/s, samples/s, unrolled states/s, forward-only
cost, and the process RSS growth over the timed run.

Usage (from the repository root)::

    python -m tools.benchmark_muzero_learning
    python -m tools.benchmark_muzero_learning --batch-sizes 16 32 64 128 --iters 20
    python -m tools.benchmark_muzero_learning --unroll-steps 10 --scalar
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
from forexmind.muzero import (
    MUZERO_NUM_ACTIONS,
    LearnerConfig,
    MuZeroConfig,
    MuZeroLearner,
    OptimizerConfig,
    build_muzero_network,
    observation_dim,
)
from forexmind.muzero.targets import MuZeroBatch

from tools.common import REPORTS_DIR


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _rss_mb() -> float | None:
    """Process RSS in MB, or ``None`` when it cannot be measured.

    ``psutil`` (the ``train`` extra) is preferred, but the benchmark must not
    print a fake ``0.0`` when it is absent, so a stdlib fallback measures the
    process working set directly.
    """
    try:
        psutil = import_module("psutil")
    except ImportError:  # pragma: no cover - depends on the installed extras
        return _fallback_rss_mb()
    return float(psutil.Process().memory_info().rss) / 1e6


def _fallback_rss_mb() -> float | None:  # pragma: no cover - platform dependent
    """RSS from the standard library (Windows ``psapi`` or POSIX ``getrusage``)."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            windll: Any = getattr(ctypes, "windll", None)
            if windll is None:
                return None
            # argtypes matter here: without them ctypes passes the 64-bit
            # process pseudohandle through a 32-bit int and the call fails.
            kernel32 = windll.kernel32
            psapi = windll.psapi
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.GetCurrentProcess.argtypes = []
            get_memory_info = psapi.GetProcessMemoryInfo
            get_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_memory_info.restype = wintypes.BOOL
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            handle = kernel32.GetCurrentProcess()
            ok = get_memory_info(handle, ctypes.byref(counters), counters.cb)
            if not ok:
                return None
            return float(counters.WorkingSetSize) / 1e6
        except Exception:
            return None
    try:
        import resource

        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    # Linux reports KiB, macOS reports bytes.
    return usage / (1024.0 if sys.platform.startswith("linux") else 1e6)


def _fmt_mb(value: float | None, *, signed: bool = False) -> str:
    """Format a memory column, printing ``n/a`` when the measurement is missing."""
    if value is None:
        return f"{'n/a':>8}"
    return f"{value:>+8,.2f}" if signed else f"{value:>8,.1f}"


def _synthetic_batch(
    batch_size: int,
    *,
    obs_dim: int,
    unroll: int,
    num_actions: int,
    value_scale: float,
    reward_scale: float,
    device: torch.device,
    seed: int = 0,
) -> MuZeroBatch:
    """A batch with realistic shapes and in-range targets (no environment needed)."""
    generator = torch.Generator().manual_seed(seed)
    steps = unroll + 1
    observation = torch.randn(batch_size, obs_dim, generator=generator)
    actions = torch.randint(0, num_actions, (batch_size, unroll), generator=generator)
    reward_units = torch.randn(batch_size, unroll, generator=generator) * (reward_scale * 0.3)
    value_units = torch.randn(batch_size, steps, generator=generator) * (value_scale * 0.3)
    policies = torch.rand(batch_size, steps, num_actions, generator=generator)
    policies = policies / policies.sum(dim=-1, keepdim=True)
    return MuZeroBatch(
        observation=observation.to(device),
        actions=actions.to(device),
        target_rewards=reward_units.to(device),
        target_values=value_units.to(device),
        target_policies=policies.to(device),
        policy_masks=torch.ones(batch_size, steps, device=device),
        value_masks=torch.ones(batch_size, steps, device=device),
        reward_masks=torch.ones(batch_size, unroll, device=device),
        action_masks=torch.ones(batch_size, steps, num_actions, dtype=torch.bool, device=device),
        trajectory_ids=torch.arange(batch_size, device=device),
        positions=torch.zeros(batch_size, dtype=torch.long, device=device),
        splits=("train",) * batch_size,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--iters", type=int, default=10, help="timed updates per batch size")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-embedding-dim", type=int, default=16)
    parser.add_argument("--support-size", type=int, default=21)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--latent-gradient-scale", type=float, default=0.5)
    parser.add_argument("--scalar", action="store_true", help="scalar value/reward heads")
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
    learner = MuZeroLearner(
        model,
        LearnerConfig.for_model(
            config,
            optimizer=OptimizerConfig(learning_rate=args.learning_rate),
            latent_gradient_scale=args.latent_gradient_scale,
            device=str(device),
            seed=args.seed,
        ),
    )
    report = learner.parameter_report()

    print("=" * 78)
    print("MuZero Stage 4.4 — joint learner throughput and memory")
    print("=" * 78)
    print(f"device            : {device} ({platform.machine()}, torch {torch.__version__})")
    print(f"obs_dim           : {config.obs_dim}")
    print(f"num_actions       : {MUZERO_NUM_ACTIONS}")
    print(f"latent_dim        : {config.latent_dim}")
    print(f"hidden_dim        : {config.hidden_dim} x {config.num_layers} layers")
    print(f"unroll steps      : {args.unroll_steps}")
    print(f"latent grad scale : {args.latent_gradient_scale}")
    print(f"representation    : {'scalar heads' if args.scalar else 'categorical support'}")
    print(
        f"parameters        : {report['total']:,} "
        f"(repr {report['representation']:,} / dyn {report['dynamics']:,} "
        f"/ pred {report['prediction']:,})"
    )
    print("-" * 78)
    header = (
        f"{'batch':>6} | {'updates/s':>10} | {'samples/s':>10} | {'states/s':>10} | "
        f"{'ms/update':>10} | {'ms/forward':>11} | {'RSS MB':>8} | {'dRSS MB':>8}"
    )
    print(header)
    print("-" * len(header))

    results: dict[str, Any] = {}
    for batch_size in args.batch_sizes:
        batch = _synthetic_batch(
            batch_size,
            obs_dim=config.obs_dim,
            unroll=args.unroll_steps,
            num_actions=config.num_actions,
            value_scale=config.value_scale,
            reward_scale=config.reward_scale,
            device=device,
            seed=args.seed,
        )
        for _ in range(args.warmup):
            learner.train_step(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        rss_before = _rss_mb()
        start = time.perf_counter()
        for _ in range(args.iters):
            learner.train_step(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        rss_after = _rss_mb()

        with torch.no_grad():
            forward_start = time.perf_counter()
            for _ in range(args.iters):
                learner.unroll(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_elapsed = time.perf_counter() - forward_start

        updates_per_sec = args.iters / elapsed
        unrolled_states = batch_size * (args.unroll_steps + 1)
        row: dict[str, Any] = {
            "updates_per_sec": updates_per_sec,
            "samples_per_sec": updates_per_sec * batch_size,
            "unrolled_states_per_sec": updates_per_sec * unrolled_states,
            "ms_per_update": 1000.0 * elapsed / args.iters,
            "ms_per_forward": 1000.0 * forward_elapsed / args.iters,
            "rss_mb": rss_after,
            "rss_delta_mb": (
                None if (rss_before is None or rss_after is None) else rss_after - rss_before
            ),
            "updates": learner.update_count,
        }
        results[str(batch_size)] = row
        delta = row["rss_delta_mb"]
        print(
            f"{batch_size:>6} | {row['updates_per_sec']:>10,.2f} | "
            f"{row['samples_per_sec']:>10,.0f} | "
            f"{row['unrolled_states_per_sec']:>10,.0f} | "
            f"{row['ms_per_update']:>10,.1f} | {row['ms_per_forward']:>11,.1f} | "
            f"{_fmt_mb(row['rss_mb'])} | "
            f"{_fmt_mb(delta, signed=True)}"
        )

    print("-" * len(header))
    if results:
        rows = sorted(results.items(), key=lambda item: int(item[0]))
        largest = rows[-1][1]
        print(
            f"note: 'states/s' counts unrolled latent states "
            f"= batch x (K+1) = batch x {args.unroll_steps + 1}"
        )
        print(
            f"note: ms/update includes loss + backward + clip + optimizer step; "
            f"ms/forward is the no-grad unroll only "
            f"({largest['ms_per_forward'] / largest['ms_per_update']:.0%} of a batch="
            f"{rows[-1][0]} update)"
        )
    print("=" * 78)

    payload: dict[str, Any] = {
        "config": {
            "obs_dim": config.obs_dim,
            "num_actions": config.num_actions,
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "action_embedding_dim": config.action_embedding_dim,
            "unroll_steps": args.unroll_steps,
            "latent_gradient_scale": args.latent_gradient_scale,
            "representation": "scalar" if args.scalar else "categorical_support",
            "support_size": args.support_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "platform": platform.machine(),
        "parameters": report,
        "iters": args.iters,
        "warmup": args.warmup,
        "results": results,
    }
    out = args.json_out
    if out is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / "stage44_muzero_learning_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
