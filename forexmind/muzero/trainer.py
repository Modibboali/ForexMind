"""The first complete MuZero training loop (Stage 4.5, scaled in Stage 4.6).

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

Stage 4.6 adds an *optional* parallel collection mode
(``num_collectors > 1``): the same loop, but trajectories come from
:class:`~forexmind.muzero.parallel_collector.MuZeroCollectorPool` (collector
processes + central batched inference + bounded trajectory queue + one replay
writer thread).  With ``num_collectors <= 1`` every code path below is the Stage
4.5 single-process one.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, fields, replace
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
from forexmind.muzero.parallel_collector import (
    CollectorPoolConfig,
    LockedReplay,
    MuZeroCollectorPool,
    WorkerDatasetSpec,
)
from forexmind.muzero.profiling import PhaseTimer, merge_phase_reports
from forexmind.muzero.replay import ReplayConfig, TrajectoryReplayBuffer
from forexmind.muzero.replay_store import load_replay, replay_store_report, save_replay
from forexmind.muzero.targets import TargetConfig
from forexmind.muzero.trajectory import model_version
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.training.data import DEFAULT_PROCESSED_DIR

__all__ = [
    "MuZeroTrainer",
    "MuZeroTrainingConfig",
    "NumericalCorruptionError",
    "TrainingProgress",
]


class NumericalCorruptionError(RuntimeError):
    """Raised when a run must stop instead of continuing on corrupt numbers.

    Stage 4.7 S37: a 50k-200k step run must abort on NaN/Inf losses, gradients,
    latents, diagnostics or returns rather than silently finishing.
    """


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
    #: ``vectorized`` (Stage 4.7 packed sampler) or ``reference`` (Stage 4.6).
    batch_backend: str = "vectorized"

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

    # -- parallel collection (Stage 4.6, brief S3-S8, S26-S27) ----------------
    #: Total concurrent collectors.  ``<= 1`` keeps the Stage 4.5 single-process
    #: loop unchanged; ``> 1`` enables the collector pool.
    num_collectors: int = 1
    #: Episodes driven in lock-step inside one worker (batched MCTS roots).
    collectors_per_worker: int = 2
    max_inference_batch_size: int = 32
    max_batch_wait_ms: float = 2.0
    inference_mode: str = "server"
    #: Device for the inference server's model copies (``None`` -> trainer device).
    inference_device: str | None = None
    trajectory_queue_size: int = 8
    sync_every_learner_updates: int = 1
    collector_stop_timeout_s: float = 60.0
    torch_threads_per_worker: int = 1
    dataset_backend: str = "auto"
    processed_dir: Path | None = None
    queue_wait_timeout_s: float = 900.0
    #: Optional collection/learning ratio guard (brief S8): when set, each
    #: iteration runs ``round(target * new_transitions)`` learner updates so
    #: scaling collectors cannot change the effective update ratio.
    target_updates_per_env_step: float | None = None
    #: Wall-time phase profiling (brief S2).  Off by default: no overhead.
    profile: bool = False

    # -- evaluation tiers (Stage 4.7, S14-S18) --------------------------------
    #: Tier B (full validation).  ``0`` disables it; Tier A (quick evaluation)
    #: keeps using ``eval_every_env_steps`` / ``eval_episodes`` / ``eval_horizon``.
    full_eval_every_env_steps: int = 0
    full_eval_episodes: int = 100
    full_eval_horizon: int = 512
    full_eval_seed: int = 4_242
    #: Always run one full validation at the end of the run (S35).  Off by
    #: default so Stage 4.5/4.6 configurations keep their exact behaviour.
    full_eval_at_end: bool = False
    #: Per-iteration machine-readable training log basename (S23).
    training_log_name: str = "training_log"
    #: Rewrite the CSV every N iterations so a killed run still has its curves.
    training_log_every_iterations: int = 10
    #: Abort the run on non-finite losses, latents or diagnostics (S37).
    stop_on_non_finite: bool = True

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
        if self.batch_backend not in ("vectorized", "reference"):
            raise ValueError("batch_backend must be 'vectorized' or 'reference'")
        if (
            self.learner_updates_per_iteration
            and self.max_transitions is not None
            and self.batch_size > self.max_transitions
        ):
            raise ValueError(
                "batch_size exceeds max_transitions; the replay can never serve a batch"
            )
        if self.num_collectors < 1:
            raise ValueError("num_collectors must be >= 1")
        if self.collectors_per_worker < 1:
            raise ValueError("collectors_per_worker must be >= 1")
        if self.inference_mode not in ("server", "local"):
            raise ValueError("inference_mode must be 'server' or 'local'")
        if self.trajectory_queue_size < 1:
            raise ValueError("trajectory_queue_size must be >= 1")
        if self.sync_every_learner_updates < 1:
            raise ValueError("sync_every_learner_updates must be >= 1")
        if self.target_updates_per_env_step is not None and self.target_updates_per_env_step < 0:
            raise ValueError("target_updates_per_env_step must be >= 0")
        if self.full_eval_every_env_steps < 0:
            raise ValueError("full_eval_every_env_steps must be >= 0")
        if self.training_log_every_iterations < 1:
            raise ValueError("training_log_every_iterations must be >= 1")
        if self.full_eval_episodes < 1 or self.full_eval_horizon < 1:
            raise ValueError("full_eval_episodes and full_eval_horizon must be >= 1")

    @property
    def parallel_collection(self) -> bool:
        """``True`` when trajectories come from the collector pool."""
        return self.num_collectors > 1

    @property
    def num_workers(self) -> int:
        """Collector processes needed to host ``num_collectors`` collectors."""
        return max(1, math.ceil(self.num_collectors / max(1, self.collectors_per_worker)))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {spec.name: getattr(self, spec.name) for spec in fields(self)}
        payload["output_dir"] = str(self.output_dir)
        payload["processed_dir"] = str(self.processed_dir) if self.processed_dir else None
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
    full_validation: dict[str, Any] | None = None
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
            "full_validation": self.full_validation,
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
                batch_backend=self.config.batch_backend,
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
        #: Tier B evaluator: same pipeline, fixed longer horizon and its own
        #: fixed episode specs (Stage 4.7 S14-S15).
        self.full_evaluator = MuZeroEvaluator(
            dataset,
            env_config,
            ObservationEncoder(encoder_config),
            search_config=SearchConfig(
                num_simulations=self.config.num_simulations,
                discount=self.config.discount,
                add_root_noise=False,
                temperature=0.0,
                seed=self.config.full_eval_seed,
            ),
            selection_metric=self.config.selection_metric,
            eval_horizon=self.config.full_eval_horizon,
            eval_seed=self.config.full_eval_seed,
            context_length=encoder_config.context_length,
            device=self.device,
        )

        # -- counters ---------------------------------------------------------
        self.network_version = 0
        self.env_steps = 0
        #: Environment steps the *collectors* have produced (they may run ahead
        #: of what the learner has accounted for; see ``env_steps``).
        self.env_steps_produced = 0
        self.trajectories_collected = 0
        self.collection_seconds = 0.0
        self.learning_seconds = 0.0
        self.evaluation_seconds = 0.0
        self.best_score: float | None = None
        self.best_env_steps: int | None = None
        #: Tier B (full validation) best score, kept separate from the quick one
        #: so two different episode sets are never compared as one series (S15).
        self.best_full_score: float | None = None
        self.best_full_env_steps: int | None = None
        self.best_checkpoint_path: str | None = None
        self.history: list[TrainingProgress] = []
        self.validation_specs: list[Any] | None = None
        self.full_validation_specs: list[Any] | None = None
        self.resume_report: dict[str, Any] = {"resumed": False, "replay_restored": False}
        self._iteration = 0
        self._last_eval_env_steps = -1
        self._last_full_eval_env_steps = -1
        self._last_checkpoint_env_steps = -1
        self.quick_eval_seconds = 0.0
        self.full_eval_seconds = 0.0
        self.checkpoint_seconds = 0.0
        self.last_full_evaluation: MuZeroEvaluation | None = None
        self.last_evaluation: MuZeroEvaluation | None = None
        self._rng = np.random.default_rng(self.config.seed)
        self._calibrated_scales: dict[str, float] | None = None
        self.training_log: list[dict[str, Any]] = []
        # -- Stage 4.6: profiling, parallel collection ------------------------
        self.profiler = PhaseTimer(enabled=self.config.profile)
        self.collector.timer = self.profiler
        self._pool: MuZeroCollectorPool | None = None
        self._pool_events: list[dict[str, Any]] = []
        self._last_taken_transitions = 0
        self._last_sync_gradient_updates = 0
        self.pool_shutdown_report: dict[str, Any] | None = None
        self.collector_shutdown_report: dict[str, Any] | None = None

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

    # -- parallel collection lifecycle (S3-S8) --------------------------------

    @property
    def pool(self) -> MuZeroCollectorPool | None:
        """The collector pool, or ``None`` in single-process mode."""
        return self._pool

    def pool_collector_config(self) -> CollectorConfig:
        """Collector config handed to the worker processes."""
        return CollectorConfig(
            split=self.config.split,
            horizon=self.config.horizon,
            num_simulations=self.config.num_simulations,
            discount=self.config.discount,
            training=True,
            temperature=self.current_temperature(),
            seed=self.config.seed,
            network_version=self.network_version,
            capture_diagnostics=True,
            # Deterministic per-decision streams are what make a parallel run
            # reproducible and comparable with the single-process collector.
            per_decision_seed=True,
        )

    def _ensure_pool(self) -> MuZeroCollectorPool:
        """Build the collector pool on first use (never in single-process mode)."""
        if self._pool is not None:
            return self._pool
        if not self.config.parallel_collection:
            raise RuntimeError("parallel collection is disabled (num_collectors <= 1)")
        buffer = self.replay.buffer if isinstance(self.replay, LockedReplay) else self.replay
        existing_ids = [
            int(trajectory.metadata.trajectory_id) for trajectory in self.replay.trajectories
        ]
        stride = 1_000_000
        pool_config = CollectorPoolConfig(
            num_workers=self.config.num_workers,
            collectors_per_worker=self.config.collectors_per_worker,
            max_inference_batch_size=self.config.max_inference_batch_size,
            max_batch_wait_ms=self.config.max_batch_wait_ms,
            inference_mode=self.config.inference_mode,
            trajectory_queue_size=self.config.trajectory_queue_size,
            worker_stop_timeout_s=self.config.collector_stop_timeout_s,
            torch_threads_per_worker=self.config.torch_threads_per_worker,
            profile=self.config.profile,
            # Resume must not reuse ids restored from the replay store (S40).
            episode_offset=(
                (max(existing_ids) // stride + 1) * stride if existing_ids else 0
            ),
        )
        self._pool = MuZeroCollectorPool(
            config=pool_config,
            dataset_spec=WorkerDatasetSpec.from_dataset(
                self.dataset,
                processed_dir=self.config.processed_dir or DEFAULT_PROCESSED_DIR,
                backend=self.config.dataset_backend,
            ),
            env_config=self.env_config,
            encoder_config=self.encoder_config,
            collector_config=self.pool_collector_config(),
            model_factory=lambda: build_muzero_network(self.model.config),
            model_config=self.model.config,
            model=self.model,
            replay=buffer,
            model_version_string=model_version(self.model),
            split=self.config.split,
            inference_device=self.config.inference_device or str(self.device),
        )
        # The trainer and the trainer's learner must both sample through the
        # writer's lock, so replay insertion can never race with sampling.
        self.replay = self._pool.replay
        self.learner.replay = self.replay
        self._pool_events.append(
            {
                "event": "pool_started",
                "seconds": time.perf_counter(),
                "config": pool_config.to_dict(),
                "worker_pids": self._pool.worker_pids,
            }
        )
        return self._pool

    def sync_inference_weights(self, *, force: bool = False) -> dict[str, Any] | None:
        """Publish learner weights to inference on the configured schedule (S16)."""
        pool = self._pool
        if pool is None:
            return None
        every = max(1, self.config.sync_every_learner_updates)
        if not force and (self.gradient_updates - self._last_sync_gradient_updates) < every:
            return None
        report = pool.sync_weights(
            self.model,
            version=self.network_version,
            model_version_string=model_version(self.model),
        )
        self._last_sync_gradient_updates = self.gradient_updates
        self._pool_events.append(
            {
                "event": "weights_synced",
                "env_steps": self.env_steps,
                "gradient_updates": self.gradient_updates,
                "network_version": self.network_version,
                "report": {key: value for key, value in report.items() if key != "inference"},
            }
        )
        return report

    def pool_diagnostics(self) -> dict[str, Any] | None:
        """Live pool diagnostics (workers, queues, inference batches, replay)."""
        pool = self._pool
        if pool is None:
            return None
        return pool.diagnostics()

    def close(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Shut the collector pool down cleanly (brief S36).  Idempotent."""
        if self._pool is None:
            return {"pool": "not_started"}
        self.pool_shutdown_report = self._pool.close(timeout_s=timeout_s)
        self._pool_events.append({"event": "pool_closed", "report": self.pool_shutdown_report})
        self.collector_shutdown_report = self.pool_shutdown_report
        return self.pool_shutdown_report

    def __enter__(self) -> MuZeroTrainer:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- phases ---------------------------------------------------------------

    def collect_phase(self) -> tuple[list[CollectedTrajectory], dict[str, float]]:
        """Collect TRAIN trajectories with the current weights (S9-S13)."""
        if self.config.parallel_collection:
            return self.collect_phase_parallel()
        with self.profiler.phase("collection_phase"):
            return self._collect_phase_single()

    def _collect_phase_single(self) -> tuple[list[CollectedTrajectory], dict[str, float]]:
        """Stage 4.5 in-process collection (unchanged behaviour)."""
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
        # Apply the (possibly scheduled) temperature to the search itself, not
        # only to the recorded metadata.
        self.collector.search_config = replace(
            self.collector.search_config, temperature=float(temperature)
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
        self._last_taken_transitions = sum(len(item.trajectory) for item in collected)
        return collected, summary

    def collect_phase_parallel(self) -> tuple[list[CollectedTrajectory], dict[str, float]]:
        """Collect TRAIN trajectories from the collector pool (Stage 4.6).

        The pool produces continuously; this phase waits for
        ``trajectories_per_iteration`` freshly written trajectories and hands
        them to replay through the writer.  Worker-side production is reported
        separately (``env_steps_produced``) because collectors deliberately run
        ahead of the learner.
        """
        pool = self._ensure_pool()
        with self.profiler.phase("collection_phase"):
            pool.set_temperature(self.current_temperature())
            self.model.eval()
            collected = pool.collect_trajectories(
                self.config.trajectories_per_iteration,
                timeout_s=self.config.queue_wait_timeout_s,
            )
        records = [record for item in collected for record in item.records]
        for item in collected:
            trajectory = item.trajectory
            if trajectory.metadata.split != self.config.split:
                raise ValueError(
                    f"refusing to account a {trajectory.metadata.split!r} trajectory as TRAIN"
                )
        summary = search_summary(records) if records else {}
        self._last_taken_transitions = sum(len(item.trajectory) for item in collected)
        return collected, summary

    def learn_phase(
        self, updates: int | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Run the configured number of joint learner updates (S15)."""
        self.model.train()
        count = (
            self.config.learner_updates_per_iteration if updates is None else int(updates)
        )
        metrics: list[dict[str, Any]] = []
        staleness: list[float] = []
        sampled_actions = np.zeros(self.model.config.num_actions, dtype=np.int64)
        version_by_id = {
            int(trajectory.metadata.trajectory_id): int(trajectory.metadata.network_version)
            for trajectory in self.replay.trajectories
        }
        for _ in range(count):
            with self.profiler.phase("replay_sampling"):
                batch = self.replay.sample(
                    self.config.batch_size,
                    target_config=self.target_config,
                    rng=self._rng,
                )
            with self.profiler.phase("learner_forward_backward"):
                row = self.learner.train_step(batch)
            metrics.append(row)
            sampled_actions += np.bincount(
                batch.actions[:, 0].numpy(), minlength=self.model.config.num_actions
            )
            for trajectory_id in batch.trajectory_ids.tolist():
                staleness.append(self.network_version - version_by_id.get(int(trajectory_id), 0))
        self._sampled_action_counts = sampled_actions
        if metrics:
            self.network_version += 1
            self.sync_inference_weights()
        self.model.eval()
        stats = staleness_summary(np.asarray(staleness, dtype=np.float64))
        return metrics, stats

    # -- validation -----------------------------------------------------------

    def evaluate(self) -> MuZeroEvaluation:
        """Tier A: cheap deterministic VALIDATION evaluation (S23-S25, S14)."""
        self.model.eval()
        if self.validation_specs is None:
            self.validation_specs = self.evaluator.selection_episode_specs(
                "validation", self.config.eval_episodes, self.config.eval_seed
            )
        with self.profiler.phase("validation"):
            return self.evaluator.evaluate(
                self.model,
                "validation",
                self.config.eval_episodes,
                episode_specs=self.validation_specs,
            )

    def evaluate_full(self, *, n_episodes: int | None = None) -> MuZeroEvaluation:
        """Tier B: the serious, longer fixed validation (S14-S15).

        Uses its own fixed episode specs (never resampled) and its own horizon,
        so a full score is only ever compared with another full score.
        """
        episodes = self.config.full_eval_episodes if n_episodes is None else int(n_episodes)
        self.model.eval()
        specs = self.full_validation_specs
        if specs is None or len(specs) != episodes:
            specs = self.full_evaluator.selection_episode_specs(
                "validation", episodes, self.config.full_eval_seed
            )
            self.full_validation_specs = specs
        with self.profiler.phase("validation_full"):
            return self.full_evaluator.evaluate(
                self.model, "validation", episodes, episode_specs=specs
            )

    # -- training loop --------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Run the integrated loop until ``max_env_steps`` is reached."""
        try:
            while self.env_steps < self.config.max_env_steps:
                self._iteration += 1
                progress = self.run_iteration()
                if (
                    self.config.progress_every_iterations
                    and self._iteration % self.config.progress_every_iterations == 0
                ):
                    print(self.format_progress(progress), flush=True)
            final_evaluation = self._maybe_evaluate(force=True)
            # Only run a final full validation if the schedule did not already
            # cover this exact environment-step boundary (never evaluate twice).
            final_full = (
                self._maybe_full_evaluate(force=True)
                if self.config.full_eval_at_end
                and self._last_full_eval_env_steps != self.env_steps
                else None
            )
            self.save_checkpoint("latest")
            if final_full is not None and self.best_full_env_steps == self.env_steps:
                self.save_checkpoint("best")
            # Close before reporting so the report carries the real shutdown
            # diagnostics (worker joins, queue drain, dropped trajectories).
            if self._pool is not None and self.pool_shutdown_report is None:
                self.close()
            self.write_training_log()
            return self.report(final_evaluation=final_evaluation)
        finally:
            # Never leave collector processes alive, on success or failure (S36).
            if self._pool is not None and self.pool_shutdown_report is None:
                self.close()

    def run_iteration(self) -> TrainingProgress:
        """One collect -> learn -> (evaluate/checkpoint) iteration."""
        previous_env_steps = self.env_steps
        previous_transitions = self.replay.num_transitions
        version_before = self.network_version

        start = time.perf_counter()
        collected, search = self.collect_phase()
        collection_seconds = time.perf_counter() - start
        self.collection_seconds += collection_seconds
        if self.config.parallel_collection:
            # Learner-visible environment steps: exactly the trajectories this
            # iteration accounted for.  ``env_steps_produced`` tracks the rest.
            new_transitions = self._last_taken_transitions
            self.env_steps += new_transitions
        else:
            self.env_steps = int(self.collector.stats.env_steps)
            new_transitions = self.replay.num_transitions - previous_transitions
        self.refresh_produced_counters()
        self.trajectories_collected += len(collected)
        returns = [float(item.trajectory.rewards.sum()) for item in collected]

        learner_rows: list[dict[str, Any]] = []
        staleness: dict[str, float] = {}
        learning_seconds = 0.0
        planned_updates = self.updates_for_iteration(new_transitions)
        if self.warmup_complete() and planned_updates:
            start = time.perf_counter()
            learner_rows, staleness = self.learn_phase(planned_updates)
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
        full_evaluation = self._maybe_full_evaluate()
        if full_evaluation is not None:
            progress.full_validation = full_evaluation.headline()
        checkpoints = self._maybe_checkpoint(
            score=evaluation.score if evaluation else None,
            full_score=full_evaluation.score if full_evaluation else None,
        )
        progress.checkpoints = checkpoints
        if self.env_steps == previous_env_steps and self.env_steps < self.config.max_env_steps:
            raise RuntimeError("collection made no environment progress; refusing to spin forever")
        self._check_finite(progress)
        self.append_training_log(progress)
        self.history.append(progress)
        return progress

    def refresh_produced_counters(self) -> None:
        """Update worker-side production counters from the pool (no-op if single)."""
        pool = self._pool
        if pool is None:
            self.env_steps_produced = self.env_steps
            return
        aggregate = pool.aggregate_stats()
        self.env_steps_produced = int(aggregate.get("env_steps", 0))

    def updates_for_iteration(self, new_transitions: int) -> int:
        """Learner updates to run this iteration (brief S8 ratio guard).

        With ``target_updates_per_env_step`` set, the update count follows the
        *measured* number of new transitions, so adding collectors cannot change
        the effective learner/environment ratio.
        """
        if self.config.target_updates_per_env_step is not None:
            return round(self.config.target_updates_per_env_step * new_transitions)
        return int(self.config.learner_updates_per_iteration)

    def _maybe_evaluate(self, *, force: bool = False) -> MuZeroEvaluation | None:
        due = (
            self.env_steps - self._last_eval_env_steps >= self.config.eval_every_env_steps
            or self._last_eval_env_steps < 0
        )
        if not force and not due:
            return None
        start = time.perf_counter()
        evaluation = self.evaluate()
        self.last_evaluation = evaluation
        elapsed = time.perf_counter() - start
        self.evaluation_seconds += elapsed
        self.quick_eval_seconds += elapsed
        self._last_eval_env_steps = self.env_steps
        score = evaluation.score
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_env_steps = self.env_steps
        return evaluation

    def _maybe_full_evaluate(self, *, force: bool = False) -> MuZeroEvaluation | None:
        """Tier B evaluation on its own schedule (S14, S17, S35)."""
        every = self.config.full_eval_every_env_steps
        due = every > 0 and self.env_steps - self._last_full_eval_env_steps >= every
        if not force and not due:
            return None
        start = time.perf_counter()
        evaluation = self.evaluate_full()
        elapsed = time.perf_counter() - start
        self.evaluation_seconds += elapsed
        self.full_eval_seconds += elapsed
        self._last_full_eval_env_steps = self.env_steps
        self.last_full_evaluation = evaluation
        score = evaluation.score
        if self.best_full_score is None or score > self.best_full_score:
            self.best_full_score = score
            self.best_full_env_steps = self.env_steps
        return evaluation

    def _maybe_checkpoint(
        self, *, score: float | None, full_score: float | None = None
    ) -> dict[str, str]:
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
        # ``best.pt`` follows the full score when Tier B is enabled, otherwise
        # the quick score - never a mixture of the two series (S15, S35).
        if self.config.full_eval_every_env_steps > 0:
            improved = full_score is not None and self.best_full_env_steps == self.env_steps
        else:
            improved = score is not None and self.best_env_steps == self.env_steps
        if improved:
            written["best"] = str(self.save_checkpoint("best"))
        return written

    # -- stop conditions (S37) ------------------------------------------------

    def _check_finite(self, progress: TrainingProgress) -> None:
        """Abort loudly on numerical corruption instead of finishing the run."""
        if not self.config.stop_on_non_finite:
            return
        for name, value in progress.learner.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise NumericalCorruptionError(
                    f"non-finite learner metric {name}={value} at env step {self.env_steps}"
                )
        for name, value in progress.search.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise NumericalCorruptionError(
                    f"non-finite search diagnostic {name}={value} at env step {self.env_steps}"
                )
        if not np.isfinite(progress.mean_trajectory_return):
            raise NumericalCorruptionError(
                f"non-finite episode return at env step {self.env_steps}"
            )

    # -- training log (S23, S24, S33) -----------------------------------------

    def append_training_log(self, progress: TrainingProgress) -> None:
        """Persist one machine-readable row per iteration (never stdout only)."""
        row: dict[str, Any] = {
            "iteration": progress.iteration,
            "env_steps": progress.env_steps,
            "env_steps_produced": self.env_steps_produced,
            "gradient_updates": self.gradient_updates,
            "network_version": progress.network_version,
            "trajectories_collected": progress.trajectories_collected,
            "new_transitions": progress.new_transitions,
            "replay_transitions": progress.replay_transitions,
            "learner_updates": progress.learner_updates,
            "updates_per_env_step": progress.updates_per_env_step,
            "mean_trajectory_return": progress.mean_trajectory_return,
            "collection_seconds": progress.collection_seconds,
            "learning_seconds": progress.learning_seconds,
            "evaluation_seconds": progress.evaluation_seconds,
            "quick_eval_seconds_total": self.quick_eval_seconds,
            "full_eval_seconds_total": self.full_eval_seconds,
            "checkpoint_seconds_total": self.checkpoint_seconds,
            "best_score": self.best_score,
            "best_full_score": self.best_full_score,
        }
        row.update({f"train_{key}": value for key, value in (progress.learner or {}).items()})
        row.update({f"search_{key}": value for key, value in (progress.search or {}).items()})
        row.update({f"staleness_{key}": value for key, value in (progress.staleness or {}).items()})
        row.update(self.replay_composition_metrics())
        if progress.validation:
            row.update({f"quick_{key}": value for key, value in progress.validation.items()})
        if progress.full_validation:
            row.update({f"full_{key}": value for key, value in progress.full_validation.items()})
        self.training_log.append(row)
        # Flush periodically: a long run that is interrupted (or stopped by the
        # stop conditions) must still have its curves on disk.
        if len(self.training_log) % self.config.training_log_every_iterations == 0:
            self.write_training_log()

    def replay_composition_metrics(self) -> dict[str, Any]:
        """Replay size/composition and sampled-action mix for the log (S33)."""
        report = self.replay.sampling_diagnostics()
        metrics: dict[str, Any] = {
            "replay_trajectories": self.replay.num_trajectories,
            "replay_positions": self.replay.total_positions,
        }
        for key, value in report.get("actions", {}).items():
            if isinstance(value, (int, float)):
                metrics[f"replay_action_{key}"] = float(value)
        for key, value in report.get("events", {}).items():
            if isinstance(value, (int, float)):
                metrics[f"replay_event_{key}"] = float(value)
        sampled = getattr(self, "_sampled_action_counts", None)
        total = float(sum(sampled)) if sampled is not None else 0.0
        metrics["sampled_positions"] = total
        if total > 0 and sampled is not None:
            for action, count in enumerate(sampled):
                metrics[f"sampled_action_{action}_fraction"] = float(count) / total
        metrics["replay_batch_backend"] = str(self.replay.config.batch_backend)
        return metrics

    def write_training_log(self, path: str | Path | None = None) -> Path:
        """Write ``training_log.csv`` (one row per iteration) next to the report."""
        destination = Path(path) if path is not None else self.log_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not self.training_log:
            destination.write_text("", encoding="utf-8")
            return destination
        keys: list[str] = []
        for row in self.training_log:
            for key in row:
                if key not in keys:
                    keys.append(key)
        import csv

        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in self.training_log:
                writer.writerow(row)
        return destination

    @property
    def log_path(self) -> Path:
        return self.output_dir / f"{self.config.training_log_name}.csv"

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
        with self.profiler.phase("checkpointing"):
            checkpoint_start = time.perf_counter()
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
            self.checkpoint_seconds += time.perf_counter() - checkpoint_start
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
        if self.config.parallel_collection:
            lines.extend(
                [
                    "",
                    f"collection mode      : parallel "
                    f"({self.config.num_workers} workers x "
                    f"{self.config.collectors_per_worker} collectors)",
                    f"produced env steps   : {self.env_steps_produced}",
                ]
            )
        lines.append("=" * 56)
        return "\n".join(lines)

    def timing_breakdown(self) -> dict[str, Any]:
        """Wall-time accounting by phase (Stage 4.7 S18/S38).

        ``evaluation_seconds`` is kept as the sum of both tiers so Stage 4.5/4.6
        reports stay comparable; the tiers are also reported separately.
        """
        total = (
            self.collection_seconds
            + self.learning_seconds
            + self.evaluation_seconds
            + self.checkpoint_seconds
        )
        return {
            "collection_seconds": self.collection_seconds,
            "learning_seconds": self.learning_seconds,
            "evaluation_seconds": self.evaluation_seconds,
            "quick_evaluation_seconds": self.quick_eval_seconds,
            "full_evaluation_seconds": self.full_eval_seconds,
            "checkpoint_seconds": self.checkpoint_seconds,
            "total_seconds": total,
            "collection_fraction": self.collection_seconds / total if total else 0.0,
            "learning_fraction": self.learning_seconds / total if total else 0.0,
            "evaluation_fraction": self.evaluation_seconds / total if total else 0.0,
            "quick_evaluation_fraction": self.quick_eval_seconds / total if total else 0.0,
            "full_evaluation_fraction": self.full_eval_seconds / total if total else 0.0,
            "validation_fraction": (
                (self.quick_eval_seconds + self.full_eval_seconds) / total if total else 0.0
            ),
            "checkpoint_fraction": self.checkpoint_seconds / total if total else 0.0,
        }

    def throughput(self) -> dict[str, float]:
        if self._pool is not None:
            aggregate = self._pool.aggregate_stats()
            searches = int(aggregate.get("searches", 0))
            recurrent_calls = int(aggregate.get("recurrent_inference_calls", 0))
            produced_env_steps = int(aggregate.get("env_steps", 0))
        else:
            searches = int(self.collector.stats.searches)
            recurrent_calls = int(self.collector.stats.recurrent_inference_calls)
            produced_env_steps = int(self.env_steps)
        env_steps = max(self.env_steps, 1)
        return {
            "env_steps_per_sec": (
                self.env_steps / self.collection_seconds if self.collection_seconds else 0.0
            ),
            "env_steps_produced_per_sec": (
                produced_env_steps / self.collection_seconds
                if self.collection_seconds
                else 0.0
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
                recurrent_calls / self.collection_seconds
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
            "env_steps_produced": float(produced_env_steps),
        }

    def report(self, *, final_evaluation: MuZeroEvaluation | None = None) -> dict[str, Any]:
        """Full structured report of the integrated run."""
        if final_evaluation is not None:
            # Cache it so a later ``write_report()`` (which has no argument to
            # pass) still reports the final validation instead of null.
            self.last_evaluation = final_evaluation
        final_evaluation = final_evaluation or self.last_evaluation
        validation = final_evaluation.headline() if final_evaluation else None
        return {
            "config": self.config.to_dict(),
            "counters": {
                "env_steps": self.env_steps,
                "env_steps_produced": self.env_steps_produced,
                "trajectories_collected": self.trajectories_collected,
                "gradient_updates": self.gradient_updates,
                "network_version": self.network_version,
                "searches": int(
                    self._pool.aggregate_stats().get("searches", 0)
                    if self._pool is not None
                    else self.collector.stats.searches
                ),
            },
            "replay": self.replay.memory_report(),
            "replay_composition": self.replay.sampling_diagnostics(),
            "throughput": self.throughput(),
            "timing": self.timing_breakdown(),
            "best_validation_score": self.best_score,
            "best_env_steps": self.best_env_steps,
            "best_full_validation_score": self.best_full_score,
            "best_full_env_steps": self.best_full_env_steps,
            "selection_metric_name": self.config.selection_metric,
            "validation": validation,
            "full_validation": self.report_full_validation(),
            "resume": self.resume_report,
            "collection": self.collection_report(),
            "profiling": self.profiling_report(),
            "training_log_path": str(self.log_path),
            "iterations": [progress.to_dict() for progress in self.history],
        }

    def report_full_validation(self) -> dict[str, Any]:
        """Tier B headline (its own fixed episode set), or an explicit "not run"."""
        if self.last_full_evaluation is None:
            return {
                "tier": "B",
                "ran": False,
                "configured_episodes": self.config.full_eval_episodes,
                "configured_horizon": self.config.full_eval_horizon,
            }
        return {
            "tier": "B",
            "ran": True,
            "episodes": self.last_full_evaluation.metrics.get("episode_count"),
            "horizon": self.config.full_eval_horizon,
            "seed": self.config.full_eval_seed,
            "selection_score": self.last_full_evaluation.score,
            "best_selection_score": self.best_full_score,
            "best_env_steps": self.best_full_env_steps,
            "headline": self.last_full_evaluation.headline(),
            "search": self.last_full_evaluation.search,
        }

    def profiling_report(self) -> dict[str, Any]:
        """Trainer-side and worker-side phase timings, merged (brief S2)."""
        reports = [self.profiler.report()]
        pool = self._pool
        if pool is not None:
            worker_profile = pool.aggregate_stats().get("profiling") or {}
            if worker_profile:
                reports.append(worker_profile)
        return merge_phase_reports(*reports)

    def collection_report(self) -> dict[str, Any]:
        """How collection was run, and what the pool did (Stage 4.6)."""
        pool = self._pool
        payload: dict[str, Any] = {
            "mode": "parallel" if self.config.parallel_collection else "single_process",
            "num_collectors": self.config.num_collectors,
            "collectors_per_worker": self.config.collectors_per_worker,
            "num_workers": self.config.num_workers if self.config.parallel_collection else 0,
        }
        if pool is not None:
            payload["config"] = pool.config.to_dict()
            payload["aggregate"] = pool.aggregate_stats()
            payload["inference"] = (
                pool.server.diagnostics()
                if pool.server is not None
                else pool.aggregate_stats().get("inference")
            )
            payload["writer"] = pool.writer.diagnostics()
            payload["events"] = list(self._pool_events)
            payload["shutdown"] = self.pool_shutdown_report
        else:
            payload["collector_stats"] = self.collector.stats.to_dict()
        return payload

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
    """Average every numeric metric the learner reports over the update group.

    Stage 4.7 needs the *whole* curve set (per-unroll-step errors, entropies,
    HOLD probabilities, latent norms), not a hand-picked subset, so the log can
    answer S23/S30-S32 without another training run.
    """
    if not rows:
        return {}
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
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
