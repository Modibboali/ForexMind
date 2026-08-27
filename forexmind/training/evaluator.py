"""Policy evaluation for validation/test (Phase 3).

Uses the existing Phase-2 :class:`EvaluationRunner` — no separate evaluation
engine.  The frozen policy is wrapped in a :class:`PolicyAgent` with
*deterministic* action selection (policy mean; no exploration noise).

Also implements validation-based checkpoint selection
(``Score = Sharpe - lambda * max_drawdown_pct`` by default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch import nn

from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SplitDataset
from forexmind.episodes.config import EpisodeConfig
from forexmind.episodes.sampler import EpisodeSampler, EpisodeSpec
from forexmind.evaluation.runner import AgentEvaluation, EvaluationRunner
from forexmind.observation.encoder import ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.policies import PolicyAgent


def _f(value: object, default: float = 0.0) -> float:
    """Coerce an unknown value to float."""
    if isinstance(value, (int, float)):
        return float(value)
    return default


def selection_score(metrics: dict[str, object], metric: str, lambda_dd: float = 1.0) -> float:
    """Configurable validation selection score."""
    if metric == "sharpe":
        return _f(metrics.get("sharpe"))
    if metric == "total_return":
        return _f(metrics.get("total_return"))
    if metric == "sharpe_drawdown":
        sharpe = _f(metrics.get("sharpe"))
        mdd = _f(metrics.get("max_drawdown_pct"))
        return sharpe - lambda_dd * mdd
    raise ValueError(f"unsupported selection metric {metric!r}")


@dataclass
class PolicyEvaluation:
    """Result of evaluating a frozen policy on a split."""

    split: str
    metrics: dict[str, object] = field(default_factory=dict)
    per_instrument: dict[str, dict[str, object]] = field(default_factory=dict)
    periods: dict[str, dict[str, object]] = field(default_factory=dict)
    trajectories: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return _f(self.metrics.get("_selection_score"))


class PolicyEvaluator:
    """Evaluates a policy on validation/test via the Phase-2 runner."""

    def __init__(
        self,
        dataset: SplitDataset,
        env_config: EnvironmentConfig,
        encoder: ObservationEncoder,
        window_config: WindowConfig | None = None,
        selection_metric: str = "sharpe_drawdown",
        lambda_drawdown: float = 1.0,
        eval_horizon: int = 512,
        eval_seed: int = 42,
        context_length: int = 64,
    ) -> None:
        self.dataset = dataset
        self.env_config = env_config
        self.encoder = encoder
        self.window_config = window_config or WindowConfig(context_length=context_length)
        self.selection_metric = selection_metric
        self.lambda_drawdown = lambda_drawdown
        self.eval_horizon = eval_horizon
        self.eval_seed = eval_seed
        self._runner = EvaluationRunner(
            dataset, env_config, self.encoder, self.window_config
        )

    def _episode_specs(self, split: str, n_episodes: int, seed: int) -> list[EpisodeSpec]:
        cfg = EpisodeConfig(
            split=split, horizon=self.eval_horizon,
            context_length=self.window_config.context_length, seed=seed,
        )
        return EpisodeSampler(self.dataset, cfg).sample(n_episodes, seed=seed)

    def evaluate(
        self,
        policy: nn.Module,
        algorithm: str,
        split: str,
        n_episodes: int,
        *,
        seed: int | None = None,
    ) -> PolicyEvaluation:
        """Run deterministic episodes on ``split`` and compute metrics."""
        agent = PolicyAgent(policy, algorithm, name=f"{algorithm}_eval")
        seed = seed if seed is not None else self.eval_seed
        specs = self._episode_specs(split, n_episodes, seed)
        ev = self._runner.run_agent(agent, specs)

        # Aggregate metrics from pooled log returns (equal weight per step).
        from forexmind.evaluation.aggregation import aggregate_across_instruments
        from forexmind.evaluation.metrics import compute_series_metrics

        grouped = ev.trajectories_by_instrument
        _ts, agg_log, per_instrument_series = aggregate_across_instruments(grouped)
        periods_per_year = self._runner.periods_per_year(split)
        metrics = compute_series_metrics(agg_log, periods_per_year)
        metrics["_selection_score"] = selection_score(
            metrics, self.selection_metric, self.lambda_drawdown
        )
        # Trading-style diagnostics from pooled trajectories.
        metrics["turnover"] = _pooled_turnover(ev)
        metrics["mean_reward"] = _pooled_mean_reward(ev)

        per_instrument: dict[str, dict[str, object]] = {}
        for instr, series in per_instrument_series.items():
            per_instrument[instr] = compute_series_metrics(series, periods_per_year)

        return PolicyEvaluation(
            split=split,
            metrics=metrics,
            per_instrument=per_instrument,
            trajectories=dict(grouped),
        )

    def score_of(self, evaluation: PolicyEvaluation) -> float:
        return evaluation.score


def _pooled_turnover(ev: AgentEvaluation) -> float:
    total_notional = 0.0
    capital = 0.0
    for trajs in ev.trajectories_by_instrument.values():
        for t in trajs:
            capital += _f(t.info.get("initial_balance"))
            for trade in t.trade_log:
                units = trade.get("units_delta", 0.0)
                price = trade.get("execution_price") or 0.0
                total_notional += abs(_f(units)) * _f(price)
    return total_notional / capital if capital > 0 else 0.0


def _pooled_mean_reward(ev: AgentEvaluation) -> float:
    rewards = [
        float(r)
        for trajs in ev.trajectories_by_instrument.values()
        for t in trajs
        for r in t.rewards
    ]
    return float(np.mean(rewards)) if rewards else 0.0
