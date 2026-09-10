"""Continuous single-account validation paths, evaluated per instrument.

Each instrument is a legitimate chronological strategy path. Multiple
instruments are intentionally not averaged into a fictional portfolio.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch import nn

from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SplitDataset
from forexmind.episodes.sampler import EpisodeSpec
from forexmind.episodes.trajectory import Trajectory
from forexmind.evaluation.metrics import compute_series_metrics
from forexmind.evaluation.runner import EvaluationRunner
from forexmind.evaluation.sampled import action_summary
from forexmind.observation.encoder import ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.policies import PolicyAgent


def chronological_spec(
    dataset: SplitDataset,
    instrument: str,
    split: str,
    context_length: int,
    seed: int,
) -> EpisodeSpec:
    """Cover every usable observation in a split with one account lifecycle."""
    bounds = dataset.split(instrument, split)
    start = bounds.first_index + context_length
    horizon = bounds.last_index - start
    if horizon < 1:
        raise ValueError(f"insufficient chronological data for {instrument}/{split}")
    return EpisodeSpec(
        instrument=instrument,
        split=split,
        start_index=start,
        end_index=bounds.last_index,
        horizon=horizon,
        context_length=context_length,
        seed=seed,
    )


def chronological_periods_per_year(trajectory: Trajectory) -> float:
    """Annualization from decisions divided by the path's elapsed calendar time."""
    initial = np.datetime64(str(trajectory.info["initial_timestamp"]), "ns")
    final = trajectory.timestamps[-1]
    span_seconds = float((final - initial) / np.timedelta64(1, "s"))
    if span_seconds <= 0:
        raise ValueError("chronological path has non-positive elapsed time")
    years = span_seconds / (365.25 * 24 * 3600)
    return trajectory.n_steps / years


def chronological_instrument_report(trajectory: Trajectory) -> dict[str, Any]:
    """Validate and summarize one real continuous account path."""
    if trajectory.n_steps != trajectory.spec.horizon:
        raise ValueError("chronological path ended before its requested horizon")
    if len(trajectory.timestamps) != trajectory.n_steps:
        raise ValueError("chronological timestamp and decision counts differ")
    diffs = np.diff(trajectory.timestamps)
    if np.isnat(trajectory.timestamps).any() or np.any(diffs <= np.timedelta64(0, "ns")):
        raise ValueError("chronological timestamps must be unique and strictly increasing")
    ppy = chronological_periods_per_year(trajectory)
    metrics = compute_series_metrics(trajectory.log_returns, ppy)
    actions = action_summary([trajectory])
    turnover = actions.pop("turnover")
    return {
        "evaluation_type": "chronological",
        "is_portfolio_path": True,
        "account_lifecycle_count": 1,
        "instrument": trajectory.spec.instrument,
        "split": trajectory.spec.split,
        "start_timestamp": trajectory.info["initial_timestamp"],
        "end_timestamp": str(trajectory.timestamps[-1]),
        "timestamps_strictly_increasing": True,
        "duplicate_timestamp_count": 0,
        "overlapping_period_count": 0,
        "periods_per_year": ppy,
        **metrics,
        "portfolio_sharpe": metrics["sharpe"],
        "portfolio_sortino": metrics["sortino"],
        "portfolio_calmar": metrics["calmar"],
        "turnover": turnover,
        "actual_executions": actions["actual_executions"],
        "position_changes": actions["position_changes"],
        "sign_reversals": actions["sign_reversals"],
        "actions": actions,
        "terminal_equity_includes_unrealized_pnl": True,
        "terminal_liquidation_requested": trajectory.info[
            "terminal_liquidation_requested"
        ],
    }


class ChronologicalEvaluator:
    """Run one continuous validation account per instrument."""

    def __init__(
        self,
        dataset: SplitDataset,
        env_config: EnvironmentConfig,
        encoder: ObservationEncoder,
        window_config: WindowConfig,
    ) -> None:
        self.dataset = dataset
        self.runner = EvaluationRunner(
            dataset,
            env_config,
            encoder,
            window_config,
            capture_account_state=True,
        )
        self.context_length = window_config.context_length

    def evaluate(
        self,
        policy: nn.Module,
        algorithm: str,
        *,
        split: str = "validation",
        instruments: tuple[str, ...] | list[str] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        selected = list(instruments or self.dataset.instruments)
        reports: dict[str, dict[str, Any]] = {}
        for offset, instrument in enumerate(selected):
            spec = chronological_spec(
                self.dataset, instrument, split, self.context_length, seed + offset
            )
            agent = PolicyAgent(policy, algorithm, name=f"{algorithm}_chronological")
            trajectory = self.runner.run_episode(agent, spec)
            reports[instrument] = chronological_instrument_report(trajectory)
        return {
            "evaluation_type": "chronological_per_instrument",
            "is_portfolio_path": False,
            "combined_portfolio_metrics": None,
            "combined_portfolio_metrics_unavailable_reason": (
                "Independent instrument accounts are not a multi-instrument portfolio"
            ),
            "split": split,
            "seed": seed,
            "per_instrument": reports,
        }
