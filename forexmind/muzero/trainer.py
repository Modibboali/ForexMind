"""The first complete MuZero training loop (Stage 4.5).

    real Forex environment
        -> MCTS with the current MuZero model      (Stage 4.2)
        -> trajectory collection                   (Stage 4.3)
        -> bounded trajectory replay               (Stage 4.3)
        -> masked joint learner update             (Stage 4.4)
        -> updated weights used by the next search (this module)

The trainer **orchestrates** the four earlier stages; it does not re-implement
inference, search, targets or the loss.  Collection and learning happen in one
process against **one model object**, so the searches of the next iteration see
the weights the learner just wrote - there is no second, stale copy to refresh.

Explicit lifecycle (S3-S5)::

    collect trajectories_per_iteration real TRAIN episodes
    add them to replay (TRAIN split enforced by the replay buffer itself)
    once the warm-up threshold is met, run learner_updates_per_iteration updates
    network_version += 1 after each update group
    periodically evaluate on fixed VALIDATION episodes and checkpoint

Diagnostics are separated on purpose (S39): network-prior quality, search
improvement, reward-model fit, value-model fit, mask legality and real
environment return are all reported independently and never collapsed.

Not implemented here (deferred to Stage 4.6+, per the brief): actor
multiprocessing, batched/parallel MCTS, reanalysis, prioritized or decision-rich
replay, HOLD downsampling, target refreshing, distributed learners and
Stochastic MuZero.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SPLIT_NAMES, SplitDataset
from forexmind.muzero.collector import (
    CollectedTrajectory,
    CollectorConfig,
    MuZeroCollector,
)
from forexmind.muzero.config import MuZeroConfig, SearchConfig
from forexmind.muzero.diagnostics import search_summary, staleness_summary
from forexmind.muzero.evaluation import DEFAULT_SELECTION_METRIC, MuZeroEvaluation, MuZeroEvaluator
from forexmind.muzero.inference import build_muzero_network
from forexmind.muzero.learner import LearnerConfig, MuZeroLearner, OptimizerConfig
from forexmind.muzero.replay import ReplayConfig, TrajectoryReplayBuffer
from forexmind.muzero.replay_store import load_replay, replay_store_report, save_replay
from forexmind.muzero.targets import TargetConfig
from forexmind.muzero.trajectory import model_version
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder

__all__ = [
    "MuZeroTrainer",
    "MuZeroTrainingConfig",
    "TrainingProgress",
]


@dataclass(frozen=True, slots=True)
class MuZeroTrainingConfig:
    """Everything that controls the integrated loop (nothing implicit)."""

    # -- collection -----------------------------------------------------------
    split: str = "train"
    instruments: tuple[str, ...] | None = None
    horizon: int = 32
    num_simulations: int = 16
    trajectories_per_iteration: int = 2
    #: Root visit temperature schedule: ``constant`` uses ``temperature_start``,
    #: ``linear`` decays it to ``temperature_end`` over ``temperature_decay_steps``
    #: environment steps.  Evaluation always uses temperature 0.
    temperature_schedule: str = "constant"
    temperature_start: float = 1.0
    temperature_end: float = 0.25
    temperature_decay_steps: int = 20_000

    # -- warm-up and ratio ----------------------------------------------------
    min_replay_transitions_before_training: int = 64
    learner_updates_per_iteration: int = 4

    # -- learner --------------------------------------------------------------
    batch_size: int = 32
    unroll_steps: int = 5
    td_steps: int = 5
    discount: float = 0.99
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 5.0
    latent_gradient_scale: float = 0.5
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    #: *None* means "calibrate from the first replay batch" (Stage 4.4 S10).
    reward_scale: float | None = None
    value_scale: float | None = None

    # -- model ----------------------------------------------------------------
    latent_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    support_size: int = 21
    use_support: bool = True

    # -- replay ---------------------------------------------------------------
    max_trajectories: int = 128
    max_transitions: int | None = None

    # -- validation / checkpointing -------------------------------------------
    max_env_steps: int = 2_048
    eval_every_env_steps: int = 512
    eval_episodes: int = 4
    eval_horizon: int = 512
    eval_seed: int = 42
    selection_metric: str = DEFAULT_SELECTION_METRIC
    checkpoint_every_env_steps: int = 1_024
    output_dir: Path = Path("data/reports/muzero")
    seed: int = 0
    progress_every_iterations: int = 1

    def __post_init__(self) -> None:
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {self.split!r}")
        if self.split != "train":
            raise ValueError("the integrated trainer collects TRAIN episodes only")
        if self.horizon < 1 or self.num_simulations < 1:
            raise ValueError("horizon and num_simulations must be >= 1")
        if self.trajectories_per_iteration < 1 or self.learner_updates_per_iteration < 0:
            raise ValueError("trajectories_per_iteration >= 1 and updates >= 0 are required")
        if self.min_replay_transitions_before_training < 0:
            raise ValueError("min_replay_transitions_before_training must be >= 0")
        if self.temperature_schedule not in ("constant", "linear"):
            raise ValueError("temperature_schedule must be 'constant' or 'linear'")
        if self.temperature_schedule == "linear" and self.temperature_decay_steps < 1:
            raise ValueError("temperature_decay_steps must be >= 1 for a linear schedule")
        if not 0.0 <= self.temperature_end <= self.temperature_start:
            raise ValueError("temperature must decay: 0 <= end <= start")
        if self.batch_size < 1 or self.max_env_steps < 1:
            raise ValueError("batch_size and max_env_steps must be >= 1")
        if self.max_trajectories < 1:
            raise ValueError("max_trajectories must be >= 1")
        if (
            self.learner_updates_per_iteration
            and self.max_transitions is not None
            and self.batch_size > self.max_transitions
        ):
            raise ValueError(
                "batch_size exceeds max_transitions; the replay can never serve a batch"
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {spec.name: getattr(self, spec.name) for spec in fields(self)}
        payload["output_dir"] = str(self.output_dir)
        payload["instruments"] = list(self.instruments) if self.instruments else None
        return payload


@dataclass(slots=True)
class TrainingProgress:
    """One iteration's measurable outcome (S5, S42)."""

    iteration: int
    env_steps: int
    trajectories_collected: int
    new_transitions: int
    replay_transitions: int
    learner_updates: int
    network_version: int
    collection_seconds: float = 0.0
    learning_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    mean_trajectory_return: float = 0.0
    learner: dict[str, float] = field(default_factory=dict)
    search: dict[str, float] = field(default_factory=dict)
    staleness: dict[str, float] = field(default_factory=dict)
    validation: dict[str, Any] | None = None
    checkpoints: dict[str, str] = field(default_factory=dict)

    @property
    def updates_per_env_step(self) -> float:
        return self.learner_updates / self.new_transitions if self.new_transitions else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "env_steps": self.env_steps,
            "trajectories_collected": self.trajectories_collected,
            "new_transitions": self.new_transitions,
            "replay_transitions": self.replay_transitions,
            "learner_updates": self.learner_updates,
            "updates_per_env_step": self.updates_per_env_step,
            "network_version": self.network_version,
            "collection_seconds": self.collection_seconds,
            "learning_seconds": self.learning_seconds,
            "evaluation_seconds": self.evaluation_seconds,
            "mean_trajectory_return": self.mean_trajectory_return,
            "training": dict(self.learner),
            "search": dict(self.search),
            "staleness": dict(self.staleness),
            "validation": self.validation,
            "checkpoints": dict(self.checkpoints),
        }


class MuZeroTrainer:
    """Synchronous integrated MuZero trainer (correction over scale)."""

    def __init__(
        self,
        dataset: SplitDataset,
        env_config: EnvironmentConfig,
        encoder_config: EncoderConfig,
        config: MuZeroTrainingConfig | None = None,
        *,
        model: Any | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or MuZeroTrainingConfig()
        self.device = torch.device(device)
        self.dataset = dataset
        self.env_config = env_config
        self.encoder_config = encoder_config
        self.output_dir = Path(self.config.output_dir)
        self.replay_dir = self.output_dir / "replay"

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.model = model or build_muzero_network(
            MuZeroConfig(
                obs_dim=encoder_config.spec.encoded_shape[0],
                latent_dim=self.config.latent_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                use_support=self.config.use_support,
                value_support_size=self.config.support_size,
                reward_support_size=self.config.support_size,
            )
        )
        self.model.to(self.device)
        self.model.eval()

        self.target_config = TargetConfig(
            num_unroll_steps=self.config.unroll_steps,
            td_steps=self.config.td_steps,
            discount=self.config.discount,
        )
        self.replay = TrajectoryReplayBuffer(
            ReplayConfig(
                max_trajectories=self.config.max_trajectories,
                max_transitions=self.config.max_transitions,
                sampling="uniform",  # S14: baseline stays uniform
                seed=self.config.seed,
            )
        )
        self.collector = MuZeroCollector(
            dataset,
            env_config,
            encoder_config,
            self.model,
            CollectorConfig(
                split=self.config.split,
                horizon=self.config.horizon,
                num_simulations=self.config.num_simulations,
                discount=self.config.discount,
                training=True,
                temperature=self.config.temperature_start,
                seed=self.config.seed,
                network_version=0,
                capture_diagnostics=True,
            ),
            instruments=self.config.instruments,
        )
        self.learner = MuZeroLearner(
            self.model,
            LearnerConfig.for_model(
                self.model.config,
                optimizer=OptimizerConfig(
                    learning_rate=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    max_grad_norm=self.config.max_grad_norm,
                ),
                latent_gradient_scale=self.config.latent_gradient_scale,
                device=str(self.device),
                seed=self.config.seed,
                loss_overrides={
                    "policy_loss_weight": self.config.policy_loss_weight,
                    "value_loss_weight": self.config.value_loss_weight,
                    "reward_loss_weight": self.config.reward_loss_weight,
                },
            ),
            replay=self.replay,
        )
        self.evaluator = MuZeroEvaluator(
            dataset,
            env_config,
            ObservationEncoder(encoder_config),
            search_config=SearchConfig(
                num_simulations=self.config.num_simulations,
                discount=self.config.discount,
                add_root_noise=False,
                temperature=0.0,
                seed=self.config.seed,
            ),
            selection_metric=self.config.selection_metric,
            eval_horizon=self.config.eval_horizon,
            eval_seed=self.config.eval_seed,
            context_length=encoder_config.context_length,
            device=self.device,
        )

        # -- counters ---------------------------------------------------------
        self.network_version = 0
        self.env_steps = 0
        self.trajectories_collected = 0
        self.collection_seconds = 0.0
        self.learning_seconds = 0.0
        self.evaluation_seconds = 0.0
        self.best_score: float | None = None
        self.best_env_steps: int | None = None
        self.best_checkpoint_path: str | None = None
        self.history: list[TrainingProgress] = []
        self.validation_specs: list[Any] | None = None
        self.resume_report: dict[str, Any] = {"resumed": False, "replay_restored": False}
        self._iteration = 0
        self._last_eval_env_steps = -1
        self._last_checkpoint_env_steps = -1
        self._rng = np.random.default_rng(self.config.seed)
        self._calibrated_scales: dict[str, float] | None = None

    # -- helpers --------------------------------------------------------------

    @property
    def gradient_updates(self) -> int:
        return int(self.learner.update_count)

    @property
    def collector_model_version(self) -> str:
        """Fingerprint of the weights the *collector* would search with."""
        return model_version(self.model)

    @property
    def learner_model_version(self) -> str:
        """Fingerprint of the weights the *learner* just optimized."""
        return model_version(self.model)

    def warmup_complete(self) -> bool:
        return self.replay.num_transitions >= self.config.min_replay_transitions_before_training

    def current_temperature(self) -> float:
        """Root visit temperature for the next collection phase (S11)."""
        if self.config.temperature_schedule == "constant":
            return float(self.config.temperature_start)
        fraction = min(1.0, self.env_steps / self.config.temperature_decay_steps)
        start = self.config.temperature_start
        end = self.config.temperature_end
        return float(start + (end - start) * fraction)

    # -- phases ---------------------------------------------------------------

    def collect_phase(self) -> tuple[list[CollectedTrajectory], dict[str, float]]:
        """Collect TRAIN trajectories with the current weights (S9-S13)."""
        temperature = self.current_temperature()
        self.collector.config = CollectorConfig(
            split=self.config.split,
            horizon=self.config.horizon,
            num_simulations=self.config.num_simulations,
            discount=self.config.discount,
            training=True,
            temperature=temperature,
            seed=self.config.seed,
            network_version=self.network_version,
            capture_diagnostics=True,
        )
        self.model.eval()
        collected: list[CollectedTrajectory] = [
            self.collector.collect_next_with_diagnostics()
            for _ in range(self.config.trajectories_per_iteration)
        ]
        records = [record for item in collected for record in item.records]
        for item in collected:
            trajectory = item.trajectory
            if trajectory.metadata.split != self.config.split:
                raise ValueError(
                    f"refusing to add a {trajectory.metadata.split!r} trajectory to TRAIN replay"
                )
            self.replay.add(trajectory, require_split=self.config.split)
        summary = search_summary(records) if records else {}
        return collected, summary

    def learn_phase(self) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Run the configured number of joint learner updates (S15)."""
        self.model.train()
        metrics: list[dict[str, Any]] = []
        staleness: list[float] = []
        version_by_id = {
            int(trajectory.metadata.trajectory_id): int(trajectory.metadata.network_version)
            for trajectory in self.replay.trajectories
        }
        for _ in range(self.config.learner_updates_per_iteration):
            batch = self.replay.sample(
                self.config.batch_size,
                target_config=self.target_config,
                rng=self._rng,
            )
            row = self.learner.train_step(batch)
            metrics.append(row)
            for trajectory_id in batch.trajectory_ids.tolist():
                staleness.append(self.network_version - version_by_id.get(int(trajectory_id), 0))
        if metrics:
            self.network_version += 1
        self.model.eval()
        stats = staleness_summary(np.asarray(staleness, dtype=np.float64))
        return metrics, stats

    # -- validation -----------------------------------------------------------

    def evaluate(self) -> MuZeroEvaluation:
        """Deterministic VALIDATION evaluation (S23-S25)."""
        self.model.eval()
        if self.validation_specs is None:
            self.validation_specs = self.evaluator.selection_episode_specs(
                "validation", self.config.eval_episodes, self.config.eval_seed
            )
        return self.evaluator.evaluate(
            self.model,
            "validation",
            self.config.eval_episodes,
            episode_specs=self.validation_specs,
        )

    # -- training loop --------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Run the integrated loop until ``max_env_steps`` is reached."""
        while self.env_steps < self.config.max_env_steps:
            self._iteration += 1
            progress = self.run_iteration()
            if (
                self.config.progress_every_iterations
                and self._iteration % self.config.progress_every_iterations == 0
            ):
                print(self.format_progress(progress), flush=True)
        final_evaluation = self._maybe_evaluate(force=True)
        self.save_checkpoint("latest")
        return self.report(final_evaluation=final_evaluation)

    def run_iteration(self) -> TrainingProgress:
        """One collect -> learn -> (evaluate/checkpoint) iteration."""
        previous_env_steps = self.env_steps
        previous_transitions = self.replay.num_transitions
        version_before = self.network_version

        start = time.perf_counter()
        collected, search = self.collect_phase()
        collection_seconds = time.perf_counter() - start
        self.collection_seconds += collection_seconds
        self.env_steps = int(self.collector.stats.env_steps)
        self.trajectories_collected += len(collected)
        new_transitions = self.replay.num_transitions - previous_transitions
        returns = [float(item.trajectory.rewards.sum()) for item in collected]

        learner_rows: list[dict[str, Any]] = []
        staleness: dict[str, float] = {}
        learning_seconds = 0.0
        if self.warmup_complete() and self.config.learner_updates_per_iteration:
            start = time.perf_counter()
            learner_rows, staleness = self.learn_phase()
            learning_seconds = time.perf_counter() - start
            self.learning_seconds += learning_seconds

        progress = TrainingProgress(
            iteration=self._iteration,
            env_steps=self.env_steps,
            trajectories_collected=self.trajectories_collected,
            new_transitions=new_transitions,
            replay_transitions=self.replay.num_transitions,
            learner_updates=len(learner_rows),
            network_version=self.network_version,
            collection_seconds=collection_seconds,
            learning_seconds=learning_seconds,
            mean_trajectory_return=float(np.mean(returns)) if returns else 0.0,
            learner=_mean_metrics(learner_rows),
            search=search,
            staleness=staleness,
        )
        if self.network_version != version_before:
            progress.learner["network_version_increment"] = float(
                self.network_version - version_before
            )

        evaluation = self._maybe_evaluate()
        if evaluation is not None:
            progress.validation = evaluation.headline()
        checkpoints = self._maybe_checkpoint(score=evaluation.score if evaluation else None)
        progress.checkpoints = checkpoints
        if self.env_steps == previous_env_steps and self.env_steps < self.config.max_env_steps:
            raise RuntimeError("collection made no environment progress; refusing to spin forever")
        self.history.append(progress)
        return progress

    def _maybe_evaluate(self, *, force: bool = False) -> MuZeroEvaluation | None:
        due = (
            self.env_steps - self._last_eval_env_steps >= self.config.eval_every_env_steps
            or self._last_eval_env_steps < 0
        )
        if not force and not due:
            return None
        start = time.perf_counter()
        evaluation = self.evaluate()
        self.evaluation_seconds += time.perf_counter() - start
        self._last_eval_env_steps = self.env_steps
        score = evaluation.score
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_env_steps = self.env_steps
        return evaluation

    def _maybe_checkpoint(self, *, score: float | None) -> dict[str, str]:
        written: dict[str, str] = {}
        due = (
            self.env_steps - self._last_checkpoint_env_steps
            >= self.config.checkpoint_every_env_steps
            or self._last_checkpoint_env_steps < 0
        )
        if due:
            written["latest"] = str(self.save_checkpoint("latest"))
            written[f"step_{self.env_steps}"] = str(self.save_checkpoint(f"step_{self.env_steps}"))
            self._last_checkpoint_env_steps = self.env_steps
        if score is not None and self.best_env_steps == self.env_steps:
            written["best"] = str(self.save_checkpoint("best"))
        return written

    # -- checkpointing --------------------------------------------------------

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "env_steps": self.env_steps,
            "gradient_updates": self.gradient_updates,
            "trajectories_collected": self.trajectories_collected,
            "network_version": self.network_version,
            "best_validation_score": self.best_score,
            "best_checkpoint_step": self.best_env_steps,
            "best_checkpoint_path": self.best_checkpoint_path,
            "selection_metric_name": self.config.selection_metric,
            "num_actions": self.model.config.num_actions,
            "num_simulations": self.config.num_simulations,
            "unroll_steps": self.config.unroll_steps,
            "td_steps": self.config.td_steps,
            "discount": self.config.discount,
            "reward_value_representation": (
                "categorical_support" if self.model.config.use_support else "scalar_regression"
            ),
            "training_config": self.config.to_dict(),
            "target_config": self.target_config.to_dict(),
            "replay_metadata": self.replay.memory_report(),
            "rng": {"numpy": _numpy_state_to_json(self._rng)},
            "collector_model_version": self.collector_model_version,
            "learner_model_version": self.learner_model_version,
        }

    def save_checkpoint(self, tag: str) -> Path:
        """Write ``<output_dir>/<tag>.pt``: model, optimizer, counters, config, RNG."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learner.env_steps = self.env_steps
        path = self.learner.save_checkpoint(
            self.output_dir / f"{tag}.pt",
            target_config=self.target_config,
            extra=self._checkpoint_payload(),
        )
        # Replay lives outside the checkpoint (S30) and is written separately.
        save_replay(
            self.replay,
            self.replay_dir,
            extra={"env_steps": self.env_steps, "network_version": self.network_version},
        )
        if tag == "best":
            self.best_checkpoint_path = str(path)
        return path

    def resume(self, path: str | Path) -> dict[str, Any]:
        """Restore a run; the replay is restored from the store when present (S29)."""
        payload = self.learner.load_checkpoint(path, strict=True)
        extra = payload.get("extra", {}) if isinstance(payload, dict) else {}
        self.env_steps = int(extra.get("env_steps", payload.get("env_steps", 0)))
        self.trajectories_collected = int(extra.get("trajectories_collected", 0))
        self.network_version = int(extra.get("network_version", 0))
        self.best_score = extra.get("best_validation_score")
        self.best_env_steps = extra.get("best_checkpoint_step")
        self.best_checkpoint_path = extra.get("best_checkpoint_path")
        rng_state = extra.get("rng", {}).get("numpy")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state

        replay_restored = False
        replay_dropped = 0
        if self.replay_dir.exists():
            self.replay, store_report = load_replay(
                self.replay_dir,
                config=self.replay.config,
                require_split=self.config.split,
            )
            self.learner.replay = self.replay
            replay_restored = True
            replay_dropped = store_report.trajectories_dropped_on_load
        self.resume_report = {
            "resumed": True,
            "checkpoint": str(path),
            "env_steps": self.env_steps,
            "gradient_updates": self.gradient_updates,
            "network_version": self.network_version,
            "best_validation_score": self.best_score,
            "replay_restored": replay_restored,
            "replay_trajectories": len(self.replay),
            "replay_dropped_on_load": replay_dropped,
            "replay_store": (replay_store_report(self.replay_dir) if replay_restored else None),
            "exact_continuation": bool(replay_restored and replay_dropped == 0),
        }
        return self.resume_report

    # -- reporting ------------------------------------------------------------

    def format_progress(self, progress: TrainingProgress) -> str:
        """Compact but complete progress block (S42)."""
        search = progress.search or {}
        learner = progress.learner or {}
        lines = [
            "MUZERO TRAINING PROGRESS",
            "=" * 56,
            f"Environment steps    : {progress.env_steps}",
            f"Trajectories         : {progress.trajectories_collected}",
            f"Replay transitions   : {progress.replay_transitions}",
            f"Gradient updates     : {self.gradient_updates}",
            f"Network version      : {progress.network_version}",
            f"Mean traj. return    : {progress.mean_trajectory_return:+.6f}",
            "",
            _fmt_row("total_loss", learner, "total_loss"),
            _fmt_row("policy_loss", learner, "policy_loss"),
            _fmt_row("value_loss", learner, "value_loss"),
            _fmt_row("reward_loss", learner, "reward_loss"),
            "",
            _fmt_row("policy_kl", learner, "policy_kl"),
            _fmt_row("reward_mae", learner, "reward_mae"),
            _fmt_row("value_mae", learner, "value_mae"),
            _fmt_row("grad_norm", learner, "gradient_norm"),
            "",
            f"MCTS simulations     : {self.config.num_simulations}",
            _fmt_row("search changed argmax", search, "search_changed_argmax_fraction"),
            _fmt_row("root visit entropy", search, "root_visit_entropy_mean"),
            _fmt_row("MCTS||prior KL", search, "mcts_network_kl_mean"),
            _fmt_row("tree depth (mean)", search, "tree_depth_mean"),
            _fmt_row("reward model MAE", search, "reward_model_mae"),
            _fmt_row("root value |delta|", search, "root_value_abs_delta_mean"),
            "",
            f"HOLD  : {_pct(search.get('selected_hold_fraction'))}"
            f"  (prior {_pct(search.get('prior_argmax_hold_fraction'))},"
            f" MCTS {_pct(search.get('search_argmax_hold_fraction'))})",
            f"FLAT  : {_pct(search.get('selected_flat_fraction'))}",
            f"SHORT : {_pct(search.get('group_short_fraction'))}",
            f"LONG  : {_pct(search.get('group_long_fraction'))}",
            "",
            f"env steps/sec        : {_rate(progress.env_steps, self.collection_seconds)}",
            f"updates/sec          : {_rate(self.gradient_updates, self.learning_seconds)}",
            f"updates/env step     : {progress.updates_per_env_step:.4f}",
        ]
        if progress.staleness:
            lines.append(
                f"replay staleness     : mean {progress.staleness.get('mean_staleness', 0.0):.2f}"
                f" max {progress.staleness.get('max_staleness', 0.0):.0f}"
            )
        if progress.validation:
            validation = progress.validation
            lines.extend(
                [
                    "",
                    f"validation log-return: {_num(validation.get('mean_episode_log_return'))}",
                    f"validation return    : {_num(validation.get('mean_episode_return'))}"
                    f"  (profitable {_pct(validation.get('profitable_episode_fraction'))})",
                    f"best score           : {_num(self.best_score)} @ {self.best_env_steps} steps",
                ]
            )
        lines.append("=" * 56)
        return "\n".join(lines)

    def timing_breakdown(self) -> dict[str, Any]:
        total = self.collection_seconds + self.learning_seconds + self.evaluation_seconds
        return {
            "collection_seconds": self.collection_seconds,
            "learning_seconds": self.learning_seconds,
            "evaluation_seconds": self.evaluation_seconds,
            "total_seconds": total,
            "collection_fraction": self.collection_seconds / total if total else 0.0,
            "learning_fraction": self.learning_seconds / total if total else 0.0,
            "evaluation_fraction": self.evaluation_seconds / total if total else 0.0,
        }

    def throughput(self) -> dict[str, float]:
        searches = int(self.collector.stats.searches)
        env_steps = max(self.env_steps, 1)
        return {
            "env_steps_per_sec": (
                self.env_steps / self.collection_seconds if self.collection_seconds else 0.0
            ),
            "searches_per_sec": searches / self.collection_seconds
            if self.collection_seconds
            else 0.0,
            "simulations_per_sec": (
                searches * self.config.num_simulations / self.collection_seconds
                if self.collection_seconds
                else 0.0
            ),
            "recurrent_inference_per_sec": (
                self.collector.stats.recurrent_inference_calls / self.collection_seconds
                if self.collection_seconds
                else 0.0
            ),
            "learner_updates_per_sec": (
                self.gradient_updates / self.learning_seconds if self.learning_seconds else 0.0
            ),
            "samples_per_sec": (
                self.gradient_updates * self.config.batch_size / self.learning_seconds
                if self.learning_seconds
                else 0.0
            ),
            "env_steps": float(env_steps),
        }

    def report(self, *, final_evaluation: MuZeroEvaluation | None = None) -> dict[str, Any]:
        """Full structured report of the integrated run."""
        validation = final_evaluation.headline() if final_evaluation else None
        return {
            "config": self.config.to_dict(),
            "counters": {
                "env_steps": self.env_steps,
                "trajectories_collected": self.trajectories_collected,
                "gradient_updates": self.gradient_updates,
                "network_version": self.network_version,
                "searches": int(self.collector.stats.searches),
            },
            "replay": self.replay.memory_report(),
            "replay_composition": self.replay.sampling_diagnostics(),
            "throughput": self.throughput(),
            "timing": self.timing_breakdown(),
            "best_validation_score": self.best_score,
            "best_env_steps": self.best_env_steps,
            "selection_metric_name": self.config.selection_metric,
            "validation": validation,
            "resume": self.resume_report,
            "iterations": [progress.to_dict() for progress in self.history],
        }

    def write_report(self, path: str | Path | None = None) -> Path:
        destination = Path(path) if path is not None else self.output_dir / "training_report.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.report(), indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return destination


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = (
        "total_loss",
        "policy_loss",
        "value_loss",
        "reward_loss",
        "gradient_norm",
        "learning_rate",
        "policy_kl",
        "value_mae",
        "reward_mae",
    )
    return {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
        if any(key in row for row in rows)
    }


def _fmt_row(label: str, metrics: dict[str, Any], key: str) -> str:
    if key not in metrics:
        return ""
    return f"{label:<21}: {float(metrics[key]):.6f}"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):5.1f}%"


def _num(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.6f}"


def _rate(count: int, seconds: float) -> str:
    return f"{count / seconds:.3f}" if seconds > 0 else "n/a"


def _numpy_state_to_json(rng: np.random.Generator) -> dict[str, Any]:
    return json.loads(json.dumps(rng.bit_generator.state, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")
