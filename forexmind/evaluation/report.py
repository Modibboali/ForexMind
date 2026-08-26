"""Evaluation reporting (Phase 2).

Builds machine-readable JSON reports and a concise human-readable summary,
with full reproducibility metadata (dataset version, split config, env /
execution / reward / episode / agent config, seeds, project version).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from forexmind import __version__
from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SplitDataset
from forexmind.episodes.trajectory import Trajectory
from forexmind.evaluation.aggregation import (
    aggregate_across_instruments,
    per_period_report,
)
from forexmind.evaluation.metrics import compute_series_metrics
from forexmind.evaluation.runner import AgentEvaluation
from forexmind.observation.encoder import EncoderConfig


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


def env_config_to_dict(config: EnvironmentConfig) -> dict[str, object]:
    return {
        "execution": {
            "spread_mode": config.execution.spread_mode,
            "spread_value": config.execution.spread_value,
            "slippage_mode": config.execution.slippage_mode,
            "slippage_value": config.execution.slippage_value,
            "commission_per_unit": config.execution.commission_per_unit,
        },
        "margin": {
            "initial_balance": str(config.margin.initial_balance),
            "leverage": str(config.margin.leverage),
            "margin_requirement": (
                str(config.margin.margin_requirement)
                if config.margin.margin_requirement is not None
                else None
            ),
            "maintenance_margin_ratio": str(config.margin.maintenance_margin_ratio),
            "max_leverage": (
                str(config.margin.max_leverage) if config.margin.max_leverage is not None else None
            ),
        },
        "reward": {"reward_type": config.reward.reward_type},
        "sizing": {
            "mode": config.sizing.mode,
            "fixed_units": str(config.sizing.fixed_units),
        },
        "decision_interval_minutes": config.decision_interval_minutes,
        "execution_timing": config.execution_timing,
        "mtm_price": config.mtm_price,
        "close_at_episode_end": config.close_at_episode_end,
        "horizon": config.horizon,
        "observation_window": config.observation_window,
    }


def _agent_report(
    evaluation: AgentEvaluation,
    *,
    dataset: SplitDataset,
    env_config: EnvironmentConfig,
    encoder_config: EncoderConfig,
    split: str,
    periods_per_year: float,
) -> dict[str, Any]:
    grouped = evaluation.trajectories_by_instrument
    per_instrument: dict[str, dict[str, object]] = {}
    for instr, trajs in grouped.items():
        timestamps, mean_log, n_ep = _mean_series(trajs)
        metrics = compute_series_metrics(mean_log, periods_per_year)
        per_instrument[instr] = {
            "n_episodes": n_ep,
            "metrics": _floatify(metrics),
            "periods": per_period_report(timestamps, mean_log, periods_per_year),
        }

    agg_ts, agg_log, _ = aggregate_across_instruments(grouped)
    aggregate = {
        "metrics": _floatify(compute_series_metrics(agg_log, periods_per_year)),
        "periods": per_period_report(agg_ts, agg_log, periods_per_year),
    }

    agent_cfg = _floatify(getattr(evaluation.config, "to_dict", lambda: {})())
    return {
        "agent": evaluation.agent_name,
        "random_seed": evaluation.seed,
        "config": agent_cfg,
        "n_episodes": evaluation.n_episodes(),
        "wall_seconds": round(evaluation.wall_seconds, 4),
        "steps_per_second": round(evaluation.steps_per_second, 1),
        "per_instrument": per_instrument,
        "aggregate": aggregate,
    }


def _mean_series(
    trajectories: list[Trajectory],
) -> tuple[np.ndarray, np.ndarray, int]:
    from forexmind.evaluation.aggregation import mean_log_return_series

    return mean_log_return_series(trajectories)


def build_report(
    *,
    agent_name: str,
    agent_evaluations: list[AgentEvaluation],
    dataset: SplitDataset,
    env_config: EnvironmentConfig,
    encoder_config: EncoderConfig,
    split: str,
    episodes: int,
    seed: int,
    horizon: int,
    periods_per_year: float,
    window_context_length: int,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a full machine-readable evaluation report."""
    reports: list[dict[str, Any]] = []
    for ev in agent_evaluations:
        reports.append(
            _agent_report(
                ev,
                dataset=dataset,
                env_config=env_config,
                encoder_config=encoder_config,
                split=split,
                periods_per_year=periods_per_year,
            )
        )

    # Aggregate random baselines across seeds into one block.
    random_runs = [r for r in reports if r["agent"] == "random"]
    other_runs = [r for r in reports if r["agent"] != "random"]

    report: dict[str, Any] = {
        "project": "ForexMind",
        "phase": 2,
        "project_version": __version__,
        "split": split,
        "split_config": dataset.split_config.to_dict(),
        "episodes": episodes,
        "seed": seed,
        "horizon": horizon,
        "periods_per_year": periods_per_year,
        "environment": env_config_to_dict(env_config),
        "encoder": encoder_config.to_dict(),
        "context_length": window_context_length,
        "instruments": list(dataset.instruments),
        "agents": [r["agent"] for r in reports],
        "random_aggregate": _aggregate_random(random_runs),
        "results": other_runs + random_runs,
        "metadata": extra_metadata or {},
    }
    return _floatify(report)


def _aggregate_random(random_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate random-baseline results across seeds (equal weight per seed)."""
    if not random_runs:
        return {}
    by_instrument: dict[str, list[float]] = {}
    for run in random_runs:
        per_instrument = run.get("per_instrument") or {}
        for instr, block in per_instrument.items():
            by_instrument.setdefault(str(instr), []).append(float(block["metrics"]["total_return"]))
    agg: dict[str, Any] = {
        "n_seeds": len(random_runs),
        "mean_total_return": {k: float(np.mean(v)) for k, v in by_instrument.items()},
        "mean_total_return_aggregate": float(np.mean([np.mean(v) for v in by_instrument.values()]))
        if by_instrument
        else 0.0,
    }
    return _floatify(agg)


def write_report(
    report: dict[str, Any],
    out_dir: str | Path,
    name: str,
    *,
    human_summary: str | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if human_summary:
        (out_dir / f"{name}.summary.txt").write_text(human_summary, encoding="utf-8")
    return path


def human_summary(report: dict[str, Any]) -> str:
    """Concise human-readable summary of a report dict."""
    lines = [
        f"ForexMind Phase 2 report  (split={report['split']}, "
        f"episodes={report['episodes']}, seed={report['seed']})",
        f"periods/year = {report['periods_per_year']:.0f}",
    ]
    for result in report["results"]:
        agent = result["agent"]
        agg = result["aggregate"]
        m = agg["metrics"]
        lines.append(
            f"  {agent:<16} return={m['total_return']:+.4f}  sharpe={m['sharpe']:+.3f}  "
            f"sortino={m['sortino']:+.3f}  maxDD={m['max_drawdown_pct']:.3f}  "
            f"steps/s={result['steps_per_second']:.0f}"
        )
    rand = report.get("random_aggregate") or {}
    if rand:
        lines.append(
            f"  random (seeds={rand['n_seeds']}): mean aggregate return "
            f"{rand['mean_total_return_aggregate']:+.4f}"
        )
    return "\n".join(lines)
