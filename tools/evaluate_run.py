"""Single-agent evaluation re-run tool (Phase 2).

Re-runs one agent under an explicit configuration (either defaults or a
previously saved report JSON) and writes a run report.

Usage:
    python -m tools.evaluate_run --agent momentum --split validation --episodes 50 --seed 7
    python -m tools.evaluate_run --config reports/baselines/test/baselines_test.json \
        --agent momentum --episodes 50 --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from forexmind.baselines import make_agent
from forexmind.config import default_config
from forexmind.episodes.config import EpisodeConfig
from forexmind.episodes.sampler import EpisodeSampler
from forexmind.evaluation.report import build_report, human_summary, write_report
from forexmind.evaluation.runner import EvaluationRunner
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig

from tools.common import (
    REPORTS_DIR,
    encoder_config_from_dict,
    environment_config_from_dict,
    make_split_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-evaluate a single ForexMind agent")
    parser.add_argument("--agent", default="momentum")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="a saved report JSON to inherit environment/encoder config",
    )
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    dataset = make_split_dataset()

    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as fh:
            saved = json.load(fh)
        env_config = environment_config_from_dict(saved["environment"])
        encoder_config = encoder_config_from_dict(saved["encoder"])
        split = str(saved.get("split", args.split))
        horizon = int(saved.get("horizon", args.horizon))
        context_length = int(saved.get("context_length", args.context_length))
        print(f"Inherited config from {args.config} (split={split}, horizon={horizon})")
    else:
        env_config = default_config(
            initial_balance="10000",
            leverage=50,
            spread_value=0.0002,
            sizing_mode="equity_fraction",
        )
        encoder_config = EncoderConfig(
            context_length=args.context_length,
            initial_balance=env_config.margin.initial_balance,
        )
        split, horizon, context_length = args.split, args.horizon, args.context_length

    encoder = ObservationEncoder(encoder_config)
    window_config = WindowConfig(context_length=context_length)
    runner = EvaluationRunner(dataset, env_config, encoder, window_config)

    sampler = EpisodeSampler(
        dataset,
        EpisodeConfig(split=split, horizon=horizon, context_length=context_length, seed=args.seed),
    )
    specs = sampler.sample(args.episodes, seed=args.seed)
    agent = make_agent(args.agent)
    evaluation = runner.run_agent(agent, specs)
    print(
        f"  {args.agent} done: {evaluation.wall_seconds:.1f}s "
        f"({evaluation.steps_per_second:.0f} steps/s)"
    )

    periods_per_year = runner.periods_per_year(split)
    report = build_report(
        agent_name=args.agent,
        agent_evaluations=[evaluation],
        dataset=dataset,
        env_config=env_config,
        encoder_config=encoder_config,
        split=split,
        episodes=args.episodes,
        seed=args.seed,
        horizon=horizon,
        periods_per_year=periods_per_year,
        window_context_length=context_length,
        extra_metadata={
            "run_at": datetime.utcnow().isoformat(timespec="seconds"),
            "re_run_of": str(args.config),
        },
    )
    out_dir = args.outdir or (REPORTS_DIR / "runs")
    path = write_report(
        report, out_dir, f"{args.agent}_{split}", human_summary=human_summary(report)
    )
    print("\n" + human_summary(report))
    print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
