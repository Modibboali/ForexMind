"""Deterministic MuZero evaluation (Stage 4.5 S23-S26).

MuZero is evaluated through the **same corrected independent-episode pipeline as
PPO**: :class:`forexmind.evaluation.runner.EvaluationRunner` runs fixed,
non-overlapping ``VALIDATION`` episode specifications, and
:func:`forexmind.evaluation.sampled.sampled_report` scores the independent
episode outcomes.  Nothing here re-implements evaluation, and synthetic
cross-episode Sharpe is never used for model selection.

* Dirichlet root noise is disabled and the search temperature is 0, so the
  agent's action is the MCTS visit argmax at every decision.
* Validation trajectories are returned to the caller and are never inserted into
  training replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SplitDataset
from forexmind.environment.actions import resolve_action
from forexmind.episodes.config import EpisodeConfig
from forexmind.episodes.sampler import EpisodeSampler, EpisodeSpec
from forexmind.evaluation.runner import EvaluationRunner
from forexmind.evaluation.sampled import EnterAndHoldAgent, sampled_report
from forexmind.muzero.actions import env_action_index, project_action_mask
from forexmind.muzero.config import SearchConfig
from forexmind.muzero.diagnostics import RootSearchRecord, search_summary
from forexmind.muzero.inference import apply_action_mask
from forexmind.muzero.search import MuZeroMCTS
from forexmind.observation.encoder import ObservationEncoder
from forexmind.observation.schema import EncodedObservation
from forexmind.observation.window import WindowConfig
from forexmind.training.evaluator import (
    DEFAULT_SELECTION_METRIC,
    VALID_SELECTION_METRICS,
    selection_score,
)

__all__ = [
    "DEFAULT_SELECTION_METRIC",
    "MuZeroAgent",
    "MuZeroEvaluation",
    "MuZeroEvaluator",
]


class MuZeroAgent:
    """A :class:`TradingAgent` that plays the deterministic MCTS policy."""

    def __init__(
        self,
        model: Any,
        search_config: SearchConfig | None = None,
        *,
        name: str = "muzero_eval",
        device: str | torch.device = "cpu",
        capture_diagnostics: bool = True,
    ) -> None:
        config = search_config or SearchConfig()
        self.model = model
        self.search_config = SearchConfig(
            num_simulations=config.num_simulations,
            discount=config.discount,
            add_root_noise=False,  # deterministic evaluation
            temperature=0.0,  # argmax over visit counts
            seed=config.seed,
            root_dirichlet_alpha=config.root_dirichlet_alpha,
            root_exploration_fraction=config.root_exploration_fraction,
        )
        self.name = name
        self.device = torch.device(device)
        self.capture_diagnostics = capture_diagnostics
        self.records: list[RootSearchRecord] = []
        self.action_mask: np.ndarray | None = None
        self.last_action_index: int | None = None
        self.searches = 0
        self._mcts = MuZeroMCTS(
            model, self.search_config, rng=np.random.default_rng(self.search_config.seed)
        )

    # -- TradingAgent protocol -------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        self.action_mask = None
        self.last_action_index = None

    def set_action_mask(self, mask: np.ndarray) -> None:
        # The runner supplies the ten-wide environment mask; MuZero reasons over
        # its own six actions, so project once here and use it everywhere.
        self.action_mask = project_action_mask(mask)

    def act(self, observation: EncodedObservation) -> Any:
        mask = self.action_mask
        if mask is None:
            raise ValueError("MuZero evaluation requires the current environment action mask")
        encoded = np.asarray(observation.encoded, dtype=np.float32)
        result = self._mcts.search(encoded, mask, add_root_noise=False)
        self.searches += 1
        action = int(result.action)
        self.last_action_index = action
        if self.capture_diagnostics:
            with torch.no_grad():
                output = self.model.initial_inference(encoded, mask)
                prior = torch.softmax(apply_action_mask(output.policy_logits, mask), dim=-1)[0]
            visits = np.asarray(result.visit_counts, dtype=np.float64)
            total = float(visits.sum())
            self.records.append(
                RootSearchRecord(
                    prior=prior.detach().cpu().numpy().astype(np.float64),
                    visits=visits,
                    policy=visits / total if total > 0 else np.zeros_like(visits),
                    q_values=np.asarray(result.diagnostics.q_values, dtype=np.float64),
                    predicted_rewards=np.asarray(
                        result.diagnostics.predicted_rewards, dtype=np.float64
                    ),
                    mask=np.asarray(result.root_action_mask, dtype=bool),
                    action=action,
                    network_value=float(result.diagnostics.root_predicted_value),
                    search_value=float(result.root_value),
                    tree_depth=int(result.diagnostics.tree_depth),
                )
            )
        return resolve_action(env_action_index(action))


@dataclass
class MuZeroEvaluation:
    """Result of evaluating one MuZero model on a split."""

    split: str
    metrics: dict[str, Any] = field(default_factory=dict)
    search: dict[str, float] = field(default_factory=dict)
    trajectories: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def score(self) -> float:
        value = self.metrics.get("_selection_score")
        return float(value) if isinstance(value, (int, float)) else 0.0

    def headline(self) -> dict[str, Any]:
        """The S23 report fields, in one flat dict."""
        metrics = self.metrics
        return {
            "episodes": metrics.get("episode_count"),
            "mean_episode_log_return": metrics.get("mean_episode_log_return"),
            "mean_episode_return": metrics.get("mean_episode_return"),
            "median_episode_return": metrics.get("median_episode_return"),
            "profitable_episode_fraction": metrics.get("profitable_episode_fraction"),
            "p10_episode_return": metrics.get("p10_episode_return"),
            "p90_episode_return": metrics.get("p90_episode_return"),
            "mean_turnover": metrics.get("mean_turnover_per_episode"),
            "mean_executions": metrics.get("mean_executions_per_episode"),
            "selection_metric_name": metrics.get("selection_metric_name"),
            "selection_score": self.score,
        }


class MuZeroEvaluator:
    """Deterministic, PPO-comparable MuZero validation."""

    def __init__(
        self,
        dataset: SplitDataset,
        env_config: EnvironmentConfig,
        encoder: ObservationEncoder,
        *,
        search_config: SearchConfig | None = None,
        selection_metric: str = DEFAULT_SELECTION_METRIC,
        eval_horizon: int = 512,
        eval_seed: int = 42,
        context_length: int = 64,
        device: str | torch.device = "cpu",
    ) -> None:
        if selection_metric not in VALID_SELECTION_METRICS:
            raise ValueError(f"selection_metric must be episode-level; got {selection_metric!r}")
        self.dataset = dataset
        self.env_config = env_config
        self.encoder = encoder
        self.search_config = search_config or SearchConfig()
        self.selection_metric = selection_metric
        self.eval_horizon = eval_horizon
        self.eval_seed = eval_seed
        self.device = device
        self.window_config = WindowConfig(context_length=context_length)
        self._runner = EvaluationRunner(
            dataset, env_config, encoder, self.window_config, capture_account_state=True
        )

    # -- episode specifications -------------------------------------------------

    def selection_episode_specs(
        self, split: str, n_episodes: int, seed: int | None = None
    ) -> list[EpisodeSpec]:
        """Fixed, balanced, non-overlapping specs - identical rules to PPO."""
        resolved = self.eval_seed if seed is None else seed
        config = EpisodeConfig(
            split=split,
            horizon=self.eval_horizon,
            context_length=self.window_config.context_length,
            seed=resolved,
        )
        return EpisodeSampler(self.dataset, config).sample_non_overlapping(
            n_episodes, seed=resolved, split=split
        )

    # -- evaluation -------------------------------------------------------------

    def evaluate(
        self,
        model: Any,
        split: str,
        n_episodes: int,
        *,
        seed: int | None = None,
        episode_specs: list[EpisodeSpec] | None = None,
        capture_diagnostics: bool = True,
    ) -> MuZeroEvaluation:
        specs = (
            episode_specs
            if episode_specs is not None
            else self.selection_episode_specs(split, n_episodes, seed)
        )
        if len(specs) != n_episodes or any(spec.split != split for spec in specs):
            raise ValueError("episode_specs must match split and requested episode count")
        agent = MuZeroAgent(
            model,
            self.search_config,
            name="muzero_eval",
            device=self.device,
            capture_diagnostics=capture_diagnostics,
        )
        evaluation = self._runner.run_agent(agent, specs)
        trajectories = [
            trajectory
            for values in evaluation.trajectories_by_instrument.values()
            for trajectory in values
        ]
        metrics = sampled_report(trajectories, self._runner.periods_per_year(split))
        metrics["selection_metric_name"] = self.selection_metric
        metrics["_selection_score"] = selection_score(metrics, self.selection_metric)
        search = search_summary(agent.records) if agent.records else {}
        return MuZeroEvaluation(
            split=split,
            metrics=metrics,
            search=search,
            trajectories=dict(evaluation.trajectories_by_instrument),
        )

    # -- baselines (S26) --------------------------------------------------------

    def evaluate_matched_baselines(
        self, episode_specs: list[EpisodeSpec]
    ) -> dict[str, dict[str, Any]]:
        """The same FLAT / enter-and-hold references PPO reports (diagnostic only)."""
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
        self, evaluation: MuZeroEvaluation, episode_specs: list[EpisodeSpec]
    ) -> dict[str, Any]:
        reports = self.evaluate_matched_baselines(episode_specs)
        baselines = {
            name: {
                "mean_episode_return": report["mean_episode_return"],
                "median_episode_return": report["median_episode_return"],
                "profitable_episode_fraction": report["profitable_episode_fraction"],
            }
            for name, report in reports.items()
        }
        return {
            "diagnostic_only": True,
            "muzero": {
                "mean_episode_return": evaluation.metrics.get("mean_episode_return"),
                "mean_episode_log_return": evaluation.metrics.get("mean_episode_log_return"),
                "profitable_episode_fraction": evaluation.metrics.get(
                    "profitable_episode_fraction"
                ),
            },
            "baselines": baselines,
        }
