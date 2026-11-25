"""Baseline benchmark runner (Phase 2, section 40).

Evaluates every baseline agent under identical episode conditions and writes a
machine-readable report plus a human summary.  Random is run with multiple
seeds and aggregated across them.

Usage:
    python -m tools.run_baselines --split validation --episodes 100 --seed 42
    python -m tools.run_baselines --split test --episodes 100 --seed 42 --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from forexmind.baselines import available_agents, make_agent
from forexmind.baselines.random import RandomAgent, RandomConfig
from forexmind.config import default_config
from forexmind.episodes.config import EpisodeConfig
from forexmind.episodes.sampler import EpisodeSampler
from forexmind.evaluation.report import build_report, human_summary, write_report
from forexmind.evaluation.runner import EvaluationRunner
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig

from tools.common import REPORTS_DIR, make_split_dataset

DEFAULT_RANDOM_SEEDS = (1, 2, 3, 4, 5)


def main() -> int:
    parser = argparse.ArgumentParser(description="ForexMind baseline benchmark")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(DEFAULT_RANDOM_SEEDS),
        help="random-agent seeds (aggregated)",
    )
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--leverage", type=float, default=50.0)
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--spread", type=float, default=0.0002)
    parser.add_argument("--commission", type=float, default=0.0)
    args = parser.parse_args()

    dataset = make_split_dataset()
    env_config = default_config(
        initial_balance=str(args.initial_balance),
        leverage=args.leverage,
        spread_value=args.spread,
        commission_per_unit=args.commission,
        sizing_mode="equity_fraction",
    )
    encoder_config = EncoderConfig(
        context_length=args.context_length,
        initial_balance=env_config.margin.initial_balance,
    )
    encoder = ObservationEncoder(encoder_config)
    window_config = WindowConfig(context_length=args.context_length)
    runner = EvaluationRunner(dataset, env_config, encoder, window_config)

    sampler = EpisodeSampler(
        dataset,
        EpisodeConfig(
            split=args.split,
            horizon=args.horizon,
            context_length=args.context_length,
            seed=args.seed,
        ),
    )
    specs = sampler.sample(args.episodes, seed=args.seed)
    print(
        f"Sampled {len(specs)} episode specs for split={args.split} seed={args.seed} "
        f"horizon={args.horizon}"
    )

    periods_per_year = runner.periods_per_year(args.split)
    print(f"periods/year (auto, split={args.split}): {periods_per_year:.0f}")

    agent_evaluations = []
    for name in available_agents():
        if name == "random":
            for s in args.seeds:
                random_agent = RandomAgent(RandomConfig(seed=s))
                ev = runner.run_agent(random_agent, specs)
                ev.seed = s
                agent_evaluations.append(ev)
                print(
                    f"  random[seed={s}] done: {ev.wall_seconds:.1f}s "
                    f"({ev.steps_per_second:.0f} steps/s)"
                )
        else:
            base_agent = make_agent(name)
            ev = runner.run_agent(base_agent, specs)
            agent_evaluations.append(ev)
            print(f"  {name:<16} done: {ev.wall_seconds:.1f}s ({ev.steps_per_second:.0f} steps/s)")

    report = build_report(
        agent_name="baselines",
        agent_evaluations=agent_evaluations,
        dataset=dataset,
        env_config=env_config,
        encoder_config=encoder_config,
        split=args.split,
        episodes=args.episodes,
        seed=args.seed,
        horizon=args.horizon,
        periods_per_year=periods_per_year,
        window_context_length=args.context_length,
        extra_metadata={"run_at": datetime.utcnow().isoformat(timespec="seconds")},
    )

    out_dir = args.outdir or (REPORTS_DIR / "baselines" / args.split)
    path = write_report(
        report, out_dir, f"baselines_{args.split}", human_summary=human_summary(report)
    )
    print("\n" + human_summary(report))
    print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
