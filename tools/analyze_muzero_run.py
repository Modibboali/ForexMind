"""Analyse a MuZero run: curves, search budgets, checkpoints, baselines (S26-S45).

Answers the Stage 4.7 scientific questions from persisted artifacts:

* learning curves from ``training_log.csv`` (losses, errors, entropies, HOLD
  behaviour, staleness, replay composition),
* **does more search help?** - the same fixed validation episodes evaluated with
  0 / 8 / 16 / 32 (optionally 64) simulations at every checkpoint,
* **does performance improve with training?** - checkpoint progression,
* per-instrument results, concentration (top-5 episodes, best-instrument
  exclusion) and overlap diagnostics,
* FLAT / fixed-exposure baselines on the identical episode specifications.

Usage (repository root)::

    python -m tools.analyze_muzero_run --run-dir data/reports/stage47_muzero_run
    python -m tools.analyze_muzero_run --run-dir <run> --budgets 0 8 16 32 64
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from forexmind.config import EnvironmentConfig
from forexmind.muzero.config import MuZeroConfig, SearchConfig
from forexmind.muzero.evaluation import MuZeroEvaluator
from forexmind.muzero.inference import build_muzero_network
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.training.data import DEFAULT_PROCESSED_DIR
from forexmind.training.dataset_mmap import resolve_dataset
from forexmind.training.runtime_diagnostics import memory_report

from tools.common import REPORTS_DIR

CURVE_METRICS = (
    "train_total_loss",
    "train_policy_loss",
    "train_value_loss",
    "train_reward_loss",
    "train_policy_kl",
    "train_policy_entropy",
    "train_target_policy_entropy",
    "train_reward_mae",
    "train_reward_rmse",
    "train_value_mae",
    "train_value_rmse",
    "train_gradient_norm",
    "train_pred_hold_prob",
    "train_target_hold_prob",
    "train_k0_value_mae",
    "train_k1_value_mae",
    "train_latent_k0_norm",
    "search_search_changed_argmax_fraction",
    "search_root_visit_entropy_mean",
    "search_mcts_network_kl_mean",
    "search_tree_depth_mean",
    "search_root_value_abs_delta_mean",
    "search_selected_hold_fraction",
    "search_prior_argmax_hold_fraction",
    "search_search_argmax_hold_fraction",
    "search_group_flat_fraction",
    "search_group_short_fraction",
    "search_group_long_fraction",
    "staleness_mean_staleness",
    "staleness_max_staleness",
    "sampled_action_0_fraction",
    "sampled_action_1_fraction",
    "replay_event_pct_entry",
    "replay_event_pct_exit",
    "replay_event_pct_resize",
    "replay_event_pct_hold",
    "updates_per_env_step",
    "env_steps",
    "gradient_updates",
)


def _load_model(path: Path, device: torch.device):
    """Rebuild the network exactly as checkpointed and load its weights."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = payload.get("model_config")
    if model_config is None:
        architecture = payload["architecture"]
        model_config = {
            "obs_dim": architecture["obs_dim"],
            "latent_dim": architecture["latent_dim"],
            "hidden_dim": architecture["hidden_dim"],
            "num_layers": architecture["num_layers"],
            "action_embedding_dim": architecture["action_embedding_dim"],
            "use_support": architecture["use_support"],
            "value_support_size": architecture["value_support_size"],
            "reward_support_size": architecture["reward_support_size"],
        }
    model = build_muzero_network(MuZeroConfig(**model_config))
    state = payload.get("model_state") or payload.get("model") or payload.get("model_state_dict")
    if state is None:
        raise KeyError(f"checkpoint {path} has no model state")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, payload


def _trade_behaviour(rows: list[dict[str, Any]]) -> dict[str, float]:
    """S25 trade-behaviour summary over evaluated episode rows."""
    def collect(key: str) -> list[float]:
        return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]

    executions = collect("actual_executions")
    changes = collect("position_changes")
    turnover = collect("turnover")
    entry_delay = collect("time_to_first_position_steps")
    holding = collect("mean_position_holding_duration")
    reversals = collect("sign_reversals")
    final_units = collect("final_position_units")

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "mean_executions": mean(executions),
        "mean_position_changes": mean(changes),
        "mean_turnover": mean(turnover),
        "mean_sign_reversals": mean(reversals),
        "mean_first_entry_delay": mean(entry_delay),
        "mean_holding_duration": mean(holding),
        "mean_final_position_units": mean(final_units),
        "episodes": float(len(rows)),
    }


def _concentration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """S45: is the headline dominated by a few episodes or one instrument?"""
    gains = np.asarray([float(row["total_return"]) for row in rows], dtype=np.float64)
    instruments = [str(row["instrument"]) for row in rows]
    order = np.argsort(-gains)
    positive_total = float(gains[gains > 0].sum())
    top5 = float(gains[order[:5]].sum())
    per_instrument: dict[str, list[float]] = {}
    for instrument, value in zip(instruments, gains, strict=True):
        per_instrument.setdefault(instrument, []).append(float(value))
    means = {name: float(np.mean(values)) for name, values in per_instrument.items()}
    best = max(means, key=lambda name: means[name]) if means else None
    without_best = (
        float(np.mean([value for name, value in means.items() if name != best]))
        if len(means) > 1
        else 0.0
    )
    return {
        "episodes": len(rows),
        "overall_mean_return": float(gains.mean()) if gains.size else 0.0,
        "overall_mean_log_return": float(
            np.mean([float(row["cumulative_log_return"]) for row in rows])
        )
        if rows
        else 0.0,
        "share_of_aggregate_gains_from_top5_episodes": (
            top5 / positive_total if positive_total > 0 else 0.0
        ),
        "best_instrument": best,
        "mean_excluding_best_instrument": without_best,
        "per_instrument_mean_return": means,
        "per_instrument_episodes": {
            name: len(values) for name, values in per_instrument.items()
        },
    }


def _curve_summary(log_path: Path) -> dict[str, Any]:
    """First-quarter vs last-quarter means for every logged curve (S23)."""
    with log_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"rows": 0}
    summary: dict[str, Any] = {"rows": len(rows)}
    quarter = max(1, len(rows) // 4)
    for key in CURVE_METRICS:
        values = []
        for row in rows:
            raw = row.get(key)
            if raw in (None, ""):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "first_quarter_mean": float(array[:quarter].mean()),
            "last_quarter_mean": float(array[-quarter:].mean()),
            "min": float(array.min()),
            "max": float(array.max()),
            "final": float(array[-1]),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/reports/stage47_muzero_run"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 8, 16, 32])
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--max-checkpoints", type=int, default=0, help="0 = all")
    parser.add_argument("--eval-episodes", type=int, default=0, help="0 = use the run's full tier")
    parser.add_argument("--eval-horizon", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--instruments", nargs="+", default=None)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--dataset-backend", default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    report_path = run_dir / "training_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"no training report at {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = report.get("config", {})
    episodes = args.eval_episodes or int(config.get("full_eval_episodes", 32))
    horizon = args.eval_horizon or int(config.get("full_eval_horizon", 256))
    seed = args.eval_seed or int(config.get("full_eval_seed", 4_242))
    instruments = (
        tuple(args.instruments)
        if args.instruments
        else tuple(config.get("instruments") or ())
    )
    device = torch.device(args.device)

    print("=" * 88)
    print("Stage 4.7 MuZero run analysis")
    print("=" * 88)
    print(f"run dir        : {run_dir}")
    print(f"eval episodes  : {episodes} (horizon {horizon}, seed {seed})")
    print(f"budgets        : {args.budgets}")
    counters = report.get("counters", {})
    print(f"env steps      : {counters.get('env_steps')}")
    print(f"updates        : {counters.get('gradient_updates')}")
    timing = report.get("timing", {})
    print(
        f"timing         : collection {100 * timing.get('collection_fraction', 0):.1f}% | "
        f"learning {100 * timing.get('learning_fraction', 0):.1f}% | "
        f"validation {100 * timing.get('validation_fraction', 0):.1f}% | "
        f"checkpoints {100 * timing.get('checkpoint_fraction', 0):.1f}%"
    )
    print(f"throughput     : {json.dumps(report.get('throughput', {}), default=str)}")

    curves = _curve_summary(run_dir / f"{config.get('training_log_name', 'training_log')}.csv")

    checkpoints = args.checkpoints
    if checkpoints is None:
        steps = sorted(
            run_dir.glob("step_*.pt"),
            key=lambda path: (
                int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else 0
            ),
        )
        if args.max_checkpoints:
            steps = steps[: args.max_checkpoints]
        checkpoints = [str(path) for path in steps]
        latest = run_dir / "latest.pt"
        if latest.is_file() and str(latest) not in checkpoints:
            checkpoints.append(str(latest))
    print(f"checkpoints    : {[Path(c).name for c in checkpoints]}")

    dataset, dataset_backend = resolve_dataset(
        processed_dir=args.processed_dir,
        instruments=instruments,
        backend=args.dataset_backend,
    )
    encoder_config = EncoderConfig()
    env_config = EnvironmentConfig()

    matrix: dict[str, dict[str, Any]] = {}
    specs = None
    baselines: dict[str, Any] = {}
    for checkpoint in checkpoints:
        path = Path(checkpoint)
        if not path.is_file():
            print(f"[skip] missing checkpoint {path}")
            continue
        model, payload = _load_model(path, device)
        extra = payload.get("extra", {})
        for budget in args.budgets:
            evaluator = MuZeroEvaluator(
                dataset,
                env_config,
                ObservationEncoder(encoder_config),
                search_config=SearchConfig(
                    num_simulations=max(1, budget),
                    discount=float(config.get("discount", 0.99)),
                    add_root_noise=False,
                    temperature=0.0,
                    seed=seed,
                ),
                eval_horizon=horizon,
                eval_seed=seed,
                context_length=encoder_config.context_length,
                device=device,
            )
            if specs is None:
                specs = evaluator.selection_episode_specs("validation", episodes, seed)
            label = f"{path.stem}|sims={budget}"
            print(f"[eval] {label} ...", flush=True)
            evaluation = evaluator.evaluate(
                model,
                "validation",
                episodes,
                episode_specs=specs,
                network_only=budget == 0,
                num_simulations=budget,
            )
            episode_rows = evaluation.metrics.get("episodes", [])
            matrix[label] = {
                "checkpoint": path.name,
                "simulations": budget,
                "env_steps": int(extra.get("env_steps", 0)),
                "network_version": int(extra.get("network_version", 0)),
                "headline": evaluation.headline(),
                "search": evaluation.search,
                "per_instrument": evaluation.metrics.get("per_instrument", {}),
                "overlap": evaluation.metrics.get("overlap_diagnostics", {}),
                "concentration": _concentration(evaluation.metrics.get("episodes", [])),
                "trade_behaviour": _trade_behaviour(episode_rows),
            }
            if not args.skip_baselines and not baselines and specs is not None:
                reports = evaluator.evaluate_matched_baselines(specs)
                baselines = {
                    name: {
                        "mean_episode_return": entry["mean_episode_return"],
                        "mean_episode_log_return": entry.get("mean_episode_log_return"),
                        "median_episode_return": entry["median_episode_return"],
                        "profitable_episode_fraction": entry["profitable_episode_fraction"],
                        "p10_episode_return": entry["episode_statistics"]["total_return"]["p10"],
                        "p90_episode_return": entry["episode_statistics"]["total_return"]["p90"],
                        "mean_turnover": entry.get("mean_turnover_per_episode"),
                        "mean_executions": entry.get("mean_executions_per_episode"),
                    }
                    for name, entry in reports.items()
                }
                print(f"[baselines] {sorted(baselines)}", flush=True)

    memory = memory_report()
    payload_out = {
        "run_dir": str(run_dir),
        "config": config,
        "counters": counters,
        "timing": timing,
        "throughput": report.get("throughput", {}),
        "dataset_backend": dataset_backend,
        "eval_specs": {"episodes": episodes, "horizon": horizon, "seed": seed},
        "curves": curves,
        "matrix": matrix,
        "baselines": baselines,
        "memory": memory,
        "ppo_reference": _ppo_reference(),
    }
    out = args.json_out or (REPORTS_DIR / "stage47_muzero_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload_out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")

    markdown = _render_markdown(payload_out)
    destination = args.markdown_out or REPORTS_DIR / "stage47_muzero_analysis.md"
    destination.write_text(markdown, encoding="utf-8")
    print(f"wrote {destination}")
    print(markdown)
    return 0


def _ppo_reference() -> dict[str, Any] | None:
    """The audited frozen PPO result, if present (S42)."""
    path = REPORTS_DIR / "ppo_evaluation_audit" / "PPO.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": str(path),
        "note": (
            "independent sampled validation episodes, 100 x 512 steps, same protocol "
            "but different episode seeds - a protocol-level comparison, not a paired one"
        ),
        "episode_count": data.get("episode_count"),
        "mean_episode_return": data.get("episode_statistics", {})
        .get("total_return", {})
        .get("mean"),
        "median_episode_return": data.get("episode_statistics", {})
        .get("total_return", {})
        .get("median"),
        "mean_cumulative_log_return": data.get("episode_statistics", {})
        .get("cumulative_log_return", {})
        .get("mean"),
        "profitable_episode_fraction": data.get("profitable_episode_fraction"),
        "p10_episode_return": data.get("episode_statistics", {})
        .get("total_return", {})
        .get("p10"),
        "p90_episode_return": data.get("episode_statistics", {})
        .get("total_return", {})
        .get("p90"),
        "mean_turnover": data.get("mean_turnover_per_episode"),
        "mean_executions": data.get("mean_executions_per_episode"),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("## Search-budget / checkpoint matrix\n")
    lines.append(
        "| checkpoint | sims | mean log return | mean return | profitable | p10 | p90 | "
        "turnover | executions | argmax changed | KL(search||prior) | depth |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["matrix"].values():
        headline = row["headline"]
        search = row["search"]
        lines.append(
            f"| {row['checkpoint']} | {row['simulations']} | "
            f"{_num(headline.get('mean_episode_log_return'))} | "
            f"{_num(headline.get('mean_episode_return'))} | "
            f"{_pct(headline.get('profitable_episode_fraction'))} | "
            f"{_num(headline.get('p10_episode_return'))} | "
            f"{_num(headline.get('p90_episode_return'))} | "
            f"{_num(headline.get('mean_turnover'))} | "
            f"{_num(headline.get('mean_executions'))} | "
            f"{_pct(search.get('search_changed_argmax_fraction'))} | "
            f"{_num(search.get('mcts_network_kl_mean'))} | "
            f"{_num(search.get('tree_depth_mean'))} |"
        )
    lines.append("\n## Curves (first vs last quarter of training)\n")
    lines.append("| metric | first | last | min | max | final |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key, stats in payload["curves"].items():
        if not isinstance(stats, dict):
            continue
        lines.append(
            f"| {key} | {stats['first_quarter_mean']:.6g} | {stats['last_quarter_mean']:.6g} | "
            f"{stats['min']:.6g} | {stats['max']:.6g} | {stats['final']:.6g} |"
        )
    if payload.get("baselines"):
        lines.append("\n## Baselines on the identical episode specs\n")
        lines.append("| agent | mean log return | mean return | median | profitable | turnover |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, entry in sorted(payload["baselines"].items()):
            lines.append(
                f"| {name} | {_num(entry.get('mean_episode_log_return'))} | "
                f"{_num(entry.get('mean_episode_return'))} | "
                f"{_num(entry.get('median_episode_return'))} | "
                f"{_pct(entry.get('profitable_episode_fraction'))} | "
                f"{_num(entry.get('mean_turnover'))} |"
            )
    if payload.get("ppo_reference"):
        ppo = payload["ppo_reference"]
        lines.append("\n## Frozen PPO reference (protocol-level comparison)\n")
        lines.append(
            f"- episodes: {ppo['episode_count']}, mean return {_num(ppo['mean_episode_return'])}, "
            f"median {_num(ppo['median_episode_return'])}, "
            f"mean log return {_num(ppo['mean_cumulative_log_return'])}, "
            f"profitable {_pct(ppo['profitable_episode_fraction'])}, "
            f"turnover {_num(ppo['mean_turnover'])}, executions {_num(ppo['mean_executions'])}"
        )
        lines.append(f"- {ppo['note']}")
    return "\n".join(lines) + "\n"


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
