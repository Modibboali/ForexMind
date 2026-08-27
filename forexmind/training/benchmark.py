"""Final test-split benchmark tables (Phase 3, §42).

After training, freeze the best checkpoint (deterministic policy) and evaluate
it on the *untouched* test split alongside the seven Phase-2 baselines using
the identical episode specs and evaluation runner.  Produces:

* an agent-level table (Return / Sharpe / Sortino / MaxDD / Turnover / PnL),
* a per-instrument table,
* a per-year table,
saved as JSON, CSV, and a readable text table.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from forexmind.baselines.base import TradingAgent, make_agent
from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SplitDataset
from forexmind.episodes.config import EpisodeConfig
from forexmind.episodes.sampler import EpisodeSampler, EpisodeSpec
from forexmind.evaluation.aggregation import (
    aggregate_across_instruments,
    per_period_report,
)
from forexmind.evaluation.metrics import compute_series_metrics
from forexmind.evaluation.runner import AgentEvaluation, EvaluationRunner
from forexmind.observation.encoder import ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.policies import PolicyAgent

BASELINE_AGENTS: tuple[str, ...] = (
    "flat",
    "long",
    "short",
    "random",
    "momentum",
    "mean_reversion",
    "sma_crossover",
)

_METRIC_COLUMNS = (
    "total_return",
    "annualized_return",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown_pct",
    "annualized_volatility",
    "turnover",
    "mean_reward",
)


def _floatify(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Mapping):
        return {k: _floatify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_floatify(v) for v in value]
    return value


def build_test_episode_specs(
    dataset: SplitDataset,
    *,
    split: str = "test",
    n_episodes: int = 100,
    horizon: int = 512,
    context_length: int = 64,
    seed: int = 42,
) -> list[EpisodeSpec]:
    """Deterministic episode specs shared by every agent in the benchmark."""
    cfg = EpisodeConfig(
        split=split, horizon=horizon, context_length=context_length, seed=seed
    )
    return EpisodeSampler(dataset, cfg).sample(n_episodes, seed=seed)


def evaluate_specs(
    runner: EvaluationRunner,
    agent: TradingAgent,
    specs: list[EpisodeSpec],
    split: str,
) -> AgentEvaluation:
    """Run one agent over the shared specs and return its raw evaluation."""
    return runner.run_agent(agent, specs)


def summarize_evaluation(
    evaluation: AgentEvaluation,
    periods_per_year: float,
) -> dict[str, Any]:
    """Aggregate metrics, per-instrument metrics, and per-year metrics."""
    grouped = evaluation.trajectories_by_instrument
    _ts, agg_log, per_instrument_series = aggregate_across_instruments(grouped)
    metrics = compute_series_metrics(agg_log, periods_per_year)
    metrics["turnover"] = _pooled_turnover(evaluation)
    metrics["mean_reward"] = _pooled_mean_reward(evaluation)

    per_instrument: dict[str, dict[str, object]] = {}
    for instr, series in per_instrument_series.items():
        per_instrument[instr] = compute_series_metrics(series, periods_per_year)
        per_instrument[instr]["turnover"] = _pooled_turnover_for(grouped.get(instr, []))

    # Per-year metrics pooled over all trajectories (reset each year).
    all_ts = np.concatenate([t.timestamps for t in _trajs(grouped)])
    all_lr = np.concatenate([t.log_returns for t in _trajs(grouped)])
    per_year = per_period_report(all_ts, all_lr, periods_per_year)

    per_instrument_year: dict[str, dict[str, dict[str, object]]] = {}
    for instr, trajs in grouped.items():
        ts = np.concatenate([t.timestamps for t in trajs])
        lr = np.concatenate([t.log_returns for t in trajs])
        per_instrument_year[instr] = per_period_report(ts, lr, periods_per_year)

    return {
        "agent": evaluation.agent_name,
        "n_episodes": evaluation.n_episodes(),
        "metrics": _floatify(metrics),
        "per_instrument": _floatify(per_instrument),
        "per_year": _floatify(per_year),
        "per_instrument_year": _floatify(per_instrument_year),
        "wall_seconds": round(evaluation.wall_seconds, 4),
        "steps_per_second": round(evaluation.steps_per_second, 1),
    }


def _pooled_turnover(evaluation: AgentEvaluation) -> float:
    total_notional = 0.0
    capital = 0.0
    for trajs in evaluation.trajectories_by_instrument.values():
        for t in trajs:
            capital += _f(t.info.get("initial_balance"))
            for trade in t.trade_log:
                units = trade.get("units_delta", 0.0)
                price = trade.get("execution_price") or 0.0
                total_notional += abs(_f(units)) * _f(price)
    return total_notional / capital if capital > 0 else 0.0


def _pooled_turnover_for(trajs: list[Any]) -> float:
    total_notional = 0.0
    capital = 0.0
    for t in trajs:
        capital += _f(t.info.get("initial_balance"))
        for trade in t.trade_log:
            units = trade.get("units_delta", 0.0)
            price = trade.get("execution_price") or 0.0
            total_notional += abs(_f(units)) * _f(price)
    return total_notional / capital if capital > 0 else 0.0


def _pooled_mean_reward(evaluation: AgentEvaluation) -> float:
    rewards = [
        float(r)
        for trajs in evaluation.trajectories_by_instrument.values()
        for t in trajs
        for r in t.rewards
    ]
    return float(np.mean(rewards)) if rewards else 0.0


def _f(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _trajs(grouped: Mapping[str, list[Any]]) -> list[Any]:
    out: list[Any] = []
    for v in grouped.values():
        out.extend(v)
    return out


def benchmark_test_split(
    *,
    dataset: SplitDataset,
    env_config: EnvironmentConfig,
    encoder: ObservationEncoder,
    window_config: WindowConfig,
    policy: nn.Module,
    algorithm: str = "sac",
    split: str = "test",
    n_episodes: int = 100,
    horizon: int = 512,
    seed: int = 42,
    baseline_agents: tuple[str, ...] = BASELINE_AGENTS,
) -> dict[str, Any]:
    """Freeze ``policy`` and compare it against all baselines on ``split``."""
    runner = EvaluationRunner(dataset, env_config, encoder, window_config)
    specs = build_test_episode_specs(
        dataset, split=split, n_episodes=n_episodes, horizon=horizon,
        context_length=window_config.context_length, seed=seed,
    )
    periods_per_year = runner.periods_per_year(split)

    results: list[dict[str, Any]] = []
    # The trained agent first (deterministic policy mean).
    policy_agent = PolicyAgent(policy, algorithm, name=f"{algorithm}_trained")
    ev = evaluate_specs(runner, policy_agent, specs, split)
    results.append(summarize_evaluation(ev, periods_per_year))

    for name in baseline_agents:
        ev = evaluate_specs(runner, make_agent(name), specs, split)
        results.append(summarize_evaluation(ev, periods_per_year))

    return {
        "dataset_version": "processed-parquet-2026",
        "split": split,
        "n_episodes": n_episodes,
        "horizon": horizon,
        "seed": seed,
        "baseline_agents": list(baseline_agents),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Tables (JSON / CSV / text)
# ---------------------------------------------------------------------------


def flatten_agent_row(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"agent": summary["agent"], "n_episodes": summary["n_episodes"]}
    for col in _METRIC_COLUMNS:
        row[col] = summary["metrics"].get(col)
    return row


def agent_table(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [flatten_agent_row(r) for r in result["results"]]


def per_instrument_table(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in result["results"]:
        for instr, m in r["per_instrument"].items():
            row: dict[str, Any] = {"agent": r["agent"], "instrument": instr}
            for col in _METRIC_COLUMNS:
                row[col] = m.get(col)
            rows.append(row)
    return rows


def per_year_table(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in result["results"]:
        for year, m in r["per_year"].items():
            row: dict[str, Any] = {"agent": r["agent"], "year": year}
            for col in _METRIC_COLUMNS:
                row[col] = m.get(col)
            rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _text_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = columns
    lines = [" | ".join(h.center(max(10, len(h))) for h in header)]
    lines.append("-+-".join("-" * max(10, len(h)) for h in header))
    for row in rows:
        lines.append(" | ".join(_fmt(row.get(c, "")).rjust(max(10, len(c))) for c in header))
    return "\n".join(lines)


def write_benchmark_results(
    result: dict[str, Any], out_dir: str | Path
) -> dict[str, Path]:
    """Write JSON, agent/instrument/year CSVs, and a text table."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "benchmark.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    paths: dict[str, Path] = {"json": json_path}
    tables = {
        "agent": (agent_table(result), ["agent", "n_episodes", *_METRIC_COLUMNS]),
        "per_instrument": (
            per_instrument_table(result),
            ["agent", "instrument", *_METRIC_COLUMNS],
        ),
        "per_year": (per_year_table(result), ["agent", "year", *_METRIC_COLUMNS]),
    }
    text_parts: list[str] = []
    for name, (rows, columns) in tables.items():
        csv_path = out / f"benchmark_{name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        paths[name] = csv_path
        text_parts.append(f"== {name.upper()} ==")
        text_parts.append(_text_table(rows, columns))
        text_parts.append("")

    text_path = out / "benchmark.txt"
    text_path.write_text("\n".join(text_parts), encoding="utf-8")
    paths["text"] = text_path
    return paths


# ---------------------------------------------------------------------------
# CLI entry point helpers
# ---------------------------------------------------------------------------


def load_checkpoint_policy(
    checkpoint_path: str | Path, obs_dim: int, model: Any
) -> tuple[nn.Module, str]:
    """Build the policy network and load weights from a checkpoint file."""
    from forexmind.training.networks import build_sac_networks

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    algorithm = state.get("algorithm", "sac")
    cfg = state.get("config", {})
    if isinstance(cfg, dict):
        from forexmind.training.config import ExperimentConfig

        exp_cfg = ExperimentConfig.from_dict(cfg)
        model = exp_cfg.model
    nets = build_sac_networks(obs_dim, 1, model)
    nets.actor.load_state_dict({k: torch.as_tensor(v) for k, v in state["policy"].items()})
    nets.actor.eval()
    return nets.actor, algorithm
