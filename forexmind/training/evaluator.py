"""Policy evaluation for validation/test (Phase 3).

Uses the existing Phase-2 :class:`EvaluationRunner` — no separate evaluation
engine.  The frozen policy is wrapped in a :class:`PolicyAgent` with
*deterministic* action selection (policy mean; no exploration noise).

Checkpoint selection uses independent episode outcomes. Synthetic, step-aligned
diagnostic curves are never eligible for model selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import nn

from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SplitDataset
from forexmind.episodes.config import EpisodeConfig
from forexmind.episodes.sampler import EpisodeSampler, EpisodeSpec
from forexmind.evaluation.runner import EvaluationRunner
from forexmind.evaluation.sampled import (
    EnterAndHoldAgent,
    paired_comparison,
    sampled_report,
)
from forexmind.observation.encoder import ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.policies import PolicyAgent


def _f(value: object, default: float = 0.0) -> float:
    """Coerce an unknown value to float.

    Execution prices and PnL are stored in the trade log as strings (Decimal
    serialization, e.g. ``'1.20088'``), so numeric strings must be coerced too;
    otherwise notional/turnover silently becomes 0.  Non-numeric values fall
    back to ``default``.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


DEFAULT_SELECTION_METRIC = "mean_episode_log_return"
VALID_SELECTION_METRICS = {DEFAULT_SELECTION_METRIC, "mean_episode_return"}
LEGACY_SELECTION_METRICS = {"sharpe", "sharpe_drawdown", "total_return"}


def selection_score(
    metrics: dict[str, object], metric: str = DEFAULT_SELECTION_METRIC, lambda_dd: float = 1.0
) -> float:
    """Return an episode-level selection score; reject legacy synthetic inputs."""
    del lambda_dd  # retained only for source compatibility with older callers
    if metric not in VALID_SELECTION_METRICS:
        if metric in LEGACY_SELECTION_METRICS:
            raise ValueError(
                f"legacy selection metric {metric!r} is invalid for independently reset "
                f"episodes; use {DEFAULT_SELECTION_METRIC!r}"
            )
        raise ValueError(f"unsupported selection metric {metric!r}")
    value = metrics.get(metric)
    if not isinstance(value, (int, float)):
        raise ValueError(f"selection metric {metric!r} is unavailable")
    return float(value)


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
        selection_metric: str = DEFAULT_SELECTION_METRIC,
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
        if selection_metric not in VALID_SELECTION_METRICS:
            raise ValueError(
                f"selection_metric must be episode-level; got {selection_metric!r}"
            )
        self._runner = EvaluationRunner(
            dataset,
            env_config,
            self.encoder,
            self.window_config,
            capture_account_state=True,
        )

    def _episode_specs(self, split: str, n_episodes: int, seed: int) -> list[EpisodeSpec]:
        cfg = EpisodeConfig(
            split=split,
            horizon=self.eval_horizon,
            context_length=self.window_config.context_length,
            seed=seed,
        )
        return EpisodeSampler(self.dataset, cfg).sample(n_episodes, seed=seed)

    def selection_episode_specs(
        self, split: str, n_episodes: int, seed: int
    ) -> list[EpisodeSpec]:
        """Fixed, balanced, non-overlapping specifications for model selection."""
        cfg = EpisodeConfig(
            split=split,
            horizon=self.eval_horizon,
            context_length=self.window_config.context_length,
            seed=seed,
        )
        return EpisodeSampler(self.dataset, cfg).sample_non_overlapping(
            n_episodes, seed=seed, split=split
        )

    def evaluate(
        self,
        policy: nn.Module,
        algorithm: str,
        split: str,
        n_episodes: int,
        *,
        seed: int | None = None,
        episode_specs: list[EpisodeSpec] | None = None,
    ) -> PolicyEvaluation:
        """Run deterministic episodes and score their independent outcomes."""
        agent = PolicyAgent(policy, algorithm, name=f"{algorithm}_eval")
        seed = seed if seed is not None else self.eval_seed
        specs = episode_specs or self._episode_specs(split, n_episodes, seed)
        if len(specs) != n_episodes or any(spec.split != split for spec in specs):
            raise ValueError("episode_specs must match split and requested episode count")
        ev = self._runner.run_agent(agent, specs)
        grouped = ev.trajectories_by_instrument
        trajectories = [t for values in grouped.values() for t in values]
        metrics = sampled_report(trajectories, self._runner.periods_per_year(split))
        metrics["selection_metric_name"] = self.selection_metric
        metrics["_selection_score"] = selection_score(metrics, self.selection_metric)

        return PolicyEvaluation(
            split=split,
            metrics=metrics,
            per_instrument=metrics["per_instrument"],
            trajectories=dict(grouped),
        )

    def score_of(self, evaluation: PolicyEvaluation) -> float:
        return evaluation.score

    def evaluate_sampled(
        self,
        policy: nn.Module,
        algorithm: str,
        split: str,
        n_episodes: int,
        *,
        seed: int | None = None,
        episode_specs: list[EpisodeSpec] | None = None,
    ) -> dict[str, Any]:
        """Return the corrected sampled report used by standalone tools."""
        return self.evaluate(
            policy,
            algorithm,
            split,
            n_episodes,
            seed=seed,
            episode_specs=episode_specs,
        ).metrics

    def evaluate_matched_baselines(
        self, episode_specs: list[EpisodeSpec]
    ) -> dict[str, dict[str, Any]]:
        """Evaluate static exposures once for a fixed validation specification set."""
        reports: dict[str, dict[str, Any]] = {}
        for action in (0, 6, 7, 8, 9, 5, 4, 3, 2):
            agent = EnterAndHoldAgent(action)
            evaluation = self._runner.run_agent(agent, episode_specs)
            trajectories = [
                trajectory
                for values in evaluation.trajectories_by_instrument.values()
                for trajectory in values
            ]
            reports[agent.name] = sampled_report(
                trajectories, self._runner.periods_per_year(episode_specs[0].split)
            )
        return reports

    def matched_baseline_report(
        self,
        policy_evaluation: PolicyEvaluation,
        episode_specs: list[EpisodeSpec],
        *,
        baseline_reports: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compare current PPO outcomes with fixed enter-once/HOLD baselines."""
        comparisons: dict[str, Any] = {}
        baselines: dict[str, Any] = {}
        reports = baseline_reports or self.evaluate_matched_baselines(episode_specs)
        for name, report in reports.items():
            baselines[name] = {
                "mean_episode_return": report["mean_episode_return"],
                "median_episode_return": report["median_episode_return"],
                "profitable_episode_fraction": report["profitable_episode_fraction"],
            }
            comparisons[name] = paired_comparison(policy_evaluation.metrics, report)
            comparisons[name].pop("pairs")
        return {"diagnostic_only": True, "baselines": baselines, "paired": comparisons}
