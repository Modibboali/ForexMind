"""Shared training infrastructure (Phase 3).

``BaseTrainer`` owns data access, the experience collector, replay buffer,
metrics, logger, checkpointing, the training loop, the validation evaluation
schedule, and best-checkpoint selection.  Algorithm-specific pieces
(SAC/PPO) subclass it and implement ``update``/``policy``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from forexmind.config import EnvironmentConfig, default_config
from forexmind.data.splits import SplitDataset
from forexmind.episodes.config import EpisodeConfig
from forexmind.episodes.sampler import EpisodeSampler, EpisodeSpec
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.checkpoint import CheckpointManager
from forexmind.training.collector import (
    EnvWorker,
    ProcessCollector,
    SyncCollector,
    Transition,
)
from forexmind.training.config import ExperimentConfig, TrainingEnvConfig
from forexmind.training.data import (
    DEFAULT_INSTRUMENT_ORDER,
    DEFAULT_PROCESSED_DIR,
)
from forexmind.training.metrics import (
    MetricStore,
    TrainerLogger,
    action_distribution,
    pathological_warnings,
)
from forexmind.training.progress import make_progress_bar

SELECTION_SCHEMA = "sampled_episode_v1"


def _to_float(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def build_env_config(env_cfg: TrainingEnvConfig) -> EnvironmentConfig:
    return default_config(
        initial_balance=env_cfg.initial_balance,
        leverage=env_cfg.leverage,
        spread_value=env_cfg.spread,
        commission_per_unit=env_cfg.commission,
        sizing_mode=env_cfg.sizing_mode,
        min_reward=env_cfg.min_reward,
        max_reward=env_cfg.max_reward,
        account_currency=env_cfg.account_currency,
        instrument_spreads=dict(env_cfg.instrument_spreads),
    )


class BaseTrainer(ABC):
    def __init__(
        self,
        config: ExperimentConfig,
        run_dir: str | Path,
        *,
        dataset: SplitDataset | None = None,
        processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Global seeding for reproducible runs (the replay buffer and the
        # environment workers use the same base seed).
        torch.manual_seed(config.compute.seed)
        np.random.seed(config.compute.seed)
        self.device = self._resolve_device(config.compute.learner_device)
        torch.set_num_threads(max(1, config.compute.torch_threads))
        interop = config.compute.torch_interop_threads
        if interop is not None:
            # Must be set before any parallel torch work is initialised.
            # torch defaults interop threads to the physical core count (e.g.
            # 112 on Kaggle) which can oversubscribe alongside worker
            # processes; making it configurable lets the launcher cap it.
            try:
                torch.set_num_interop_threads(max(1, int(interop)))
            except RuntimeError as exc:  # pragma: no cover - pool already started
                print(f"  !! could not set torch interop threads to {interop}: {exc}")
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, config.compute.torch_threads)))
        os.environ.setdefault("MKL_NUM_THREADS", str(max(1, config.compute.torch_threads)))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, config.compute.torch_threads)))

        if dataset is not None:
            self.dataset = dataset
        else:
            from forexmind.training.dataset_mmap import resolve_dataset

            self.dataset, self.dataset_backend = resolve_dataset(
                processed_dir=processed_dir,
                split_config=None,
                instruments=tuple(config.environment.instruments)
                if config.environment.instruments
                else DEFAULT_INSTRUMENT_ORDER,
                backend=config.compute.dataset_backend,
            )
        self.env_config = build_env_config(config.environment)
        self.encoder = ObservationEncoder(
            EncoderConfig(
                context_length=config.environment.context_length,
                initial_balance=self.env_config.margin.initial_balance,
            )
        )
        self.window_config = WindowConfig(context_length=config.environment.context_length)
        self.episode_config = EpisodeConfig(
            split=config.environment.split,
            horizon=config.environment.horizon,
            context_length=config.environment.context_length,
            seed=config.compute.seed,
        )
        self.obs_dim = self.encoder.config.spec.encoded_shape[0]
        self.action_dim = 10 if config.algorithm.name == "ppo" else 1

        self._resolve_worker_count()

        self.collector = self._build_collector()
        self._warn_memory_budget()
        self._producer_worker_ids: set[int] = set()
        self._cpu_sampler = self._make_cpu_sampler()
        self.metric_store = MetricStore()
        self.logger = TrainerLogger(verbose=True)
        self.checkpoints = CheckpointManager(self.run_dir)
        self._episode_reward_by_worker: dict[int, float] = {}
        self._episode_steps_by_worker: dict[int, int] = {}
        self._episode_returns: list[float] = []
        self._episode_lengths: list[int] = []
        self._recent_actions: list[float] = []
        self._recent_rewards: list[float] = []
        self._diag_history: list[dict[str, float]] = []
        self._env_steps = 0
        self._gradient_updates = 0
        self._episodes = 0
        self.best_score = -float("inf")
        self.best_checkpoint: str | None = None
        self.best_checkpoint_step: int | None = None
        from forexmind.training.evaluator import (
            DEFAULT_SELECTION_METRIC,
            LEGACY_SELECTION_METRICS,
            VALID_SELECTION_METRICS,
        )

        configured_metric = config.selection.metric
        self.legacy_selection_metric_name: str | None = None
        if configured_metric in VALID_SELECTION_METRICS:
            self.selection_metric_name = configured_metric
        elif configured_metric in LEGACY_SELECTION_METRICS:
            self.legacy_selection_metric_name = configured_metric
            self.selection_metric_name = DEFAULT_SELECTION_METRIC
            print(
                f"[selection] legacy metric {configured_metric!r} is disabled; "
                f"using {self.selection_metric_name!r}",
                flush=True,
            )
        else:
            raise ValueError(f"unsupported checkpoint selection metric {configured_metric!r}")
        self.legacy_selection_score: float | None = None
        self.latest_validation_summary: dict[str, Any] = {}
        self.validation_episode_specs: list[EpisodeSpec] = []
        self.legacy_validation_history: list[dict[str, Any]] = []
        self.validation_history: list[dict[str, Any]] = []
        self._start_wall = time.perf_counter()

    # -- abstract -------------------------------------------------------------

    @abstractmethod
    def update(self) -> dict[str, float]:
        """One learner update (SAC or PPO).  Returns loss diagnostics."""

    @abstractmethod
    def policy(self) -> nn.Module:
        """The exploration policy module."""

    @abstractmethod
    def state_dicts(self) -> dict[str, Any]:
        """Everything needed to save/restore this trainer."""

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """Restore this trainer from a checkpoint."""

    def value_net(self) -> nn.Module | None:
        """Optional critic/value network to also sync to the workers (PPO)."""
        return None

    @property
    def env_steps(self) -> int:
        """Number of environment transitions consumed by the trainer."""
        return self._env_steps

    @property
    def gradient_updates(self) -> int:
        """Number of learner optimizer updates completed by the trainer."""
        return self._gradient_updates

    # -- device ---------------------------------------------------------------

    def _resolve_device(self, name: str) -> torch.device:
        if name == "cuda" and not torch.cuda.is_available():
            print("  !! learner_device=cuda requested but CUDA unavailable; using cpu")
            return torch.device("cpu")
        return torch.device(name)

    # -- memory planning ------------------------------------------------------

    def _estimate_worker_rss_mb(self) -> float:
        """Estimated per-worker private RSS by dataset backend.

        With ``parquet`` every worker materialises the full dataset privately
        (dataset_summary's estimate + process overhead).  With ``mmap`` the
        data pages are shared by the OS, so each worker's *private* footprint
        is small (torch/pandas/env/rollout only).
        """
        backend = getattr(self, "dataset_backend", "parquet")
        if backend == "mmap":
            return 400.0  # private per-worker footprint (shared data excluded)
        from forexmind.training.data import dataset_summary

        data_mb = _to_float(dataset_summary(self.dataset).get("estimated_memory_mb"), 0.0)
        return data_mb + 400.0  # dataset copy + worker overhead

    def _resolve_worker_count(self) -> None:
        """Resolve ``compute.num_workers`` to a concrete int (handles "auto")."""
        from forexmind.training.memory_planning import (
            estimate_auto_worker_count,
            total_ram_mb,
        )

        cfg = self.config.compute
        if cfg.num_workers == "auto":
            n = estimate_auto_worker_count(
                worker_rss_mb=self._estimate_worker_rss_mb(),
                memory_limit_fraction=cfg.memory_limit_fraction or 0.70,
            )
            self._resolved_num_workers = n
            print(
                f"[memory] num_workers=auto -> {n} "
                f"(RAM {total_ram_mb() / 1000:.0f} GB, "
                f"fraction {cfg.memory_limit_fraction or 0.70:.2f}, "
                f"est worker {self._estimate_worker_rss_mb():.0f} MB)"
            )
        else:
            self._resolved_num_workers = int(cfg.num_workers)

    def _warn_memory_budget(self) -> None:
        """Warn when the estimated process-tree RAM exceeds the budget fraction."""
        from forexmind.training.memory_planning import (
            estimate_process_tree_mb,
            total_ram_mb,
        )

        cfg = self.config.compute
        if cfg.memory_limit_fraction is None:
            return
        ram_mb = total_ram_mb()
        if ram_mb <= 0:
            return
        worker_mb = self._estimate_worker_rss_mb()
        est_tree_mb = estimate_process_tree_mb(
            worker_rss_mb=worker_mb,
            workers=self._resolved_num_workers,
        )
        limit_mb = cfg.memory_limit_fraction * ram_mb
        print(
            f"[memory] est process-tree ~{est_tree_mb:.0f} MB "
            f"(workers={self._resolved_num_workers} x {worker_mb:.0f} MB + reserve) "
            f"vs budget {limit_mb:.0f} MB "
            f"({cfg.memory_limit_fraction:.0%} of {ram_mb / 1000:.0f} GB)"
        )
        if est_tree_mb > limit_mb:
            print(
                f"  !! WARNING: estimated process-tree memory {est_tree_mb:.0f} MB "
                f"exceeds the {cfg.memory_limit_fraction:.0%} budget "
                f"({limit_mb:.0f} MB). Reduce compute.num_workers or set "
                f"compute.dataset_backend to 'mmap' (shared dataset store)."
            )

    # -- collector ------------------------------------------------------------

    def _make_cpu_sampler(self) -> Any:
        """Process-tree CPU sampler for trainer + live workers (Kaggle)."""
        from forexmind.training.runtime_diagnostics import ProcessTreeCpuSampler

        return ProcessTreeCpuSampler(worker_pids=getattr(self.collector, "worker_pids", []))

    def _start_cpu_sampling(self) -> None:
        sampler = self._cpu_sampler
        if sampler is not None:
            sampler.start()

    def _report_cpu_interval(self) -> None:
        """Print average process-tree CPU since the previous log interval.

        Uses psutil window semantics: ``start()`` primes the counters and
        ``stop()`` returns the average percent since the prime, so each call
        reports the average over the whole logging interval without blocking.
        """
        sampler = self._cpu_sampler
        if sampler is None or not sampler.has_procs:
            return
        cpu = sampler.stop()
        print(
            "[cpu] interval_avg "
            f"trainer={float(cpu.get('trainer_cpu_percent', 0.0)):.1f}% "
            f"worker_agg={float(cpu.get('worker_cpu_percent', 0.0)):.1f}% "
            f"tree={float(cpu.get('process_tree_cpu_percent', 0.0)):.1f}% "
            f"live_workers={getattr(self.collector, 'alive_workers', 0)} "
            f"producers={len(self._producer_worker_ids)} "
            f"env_steps={self._env_steps:,}",
            flush=True,
        )
        sampler.start()

    def _build_collector(self) -> SyncCollector | ProcessCollector:
        cfg = self.config
        model = cfg.model
        if cfg.compute.collect_backend == "process":
            return ProcessCollector(
                processed_dir=str(DEFAULT_PROCESSED_DIR),
                split_config=self.dataset.split_config,
                instruments=self.dataset.instruments,
                env_config=self.env_config,
                encoder_config=self.encoder.config,
                window_config=self.window_config,
                episode_config=self.episode_config,
                algorithm=cfg.algorithm.name,
                model=model,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                global_seed=cfg.compute.seed,
                num_workers=self._resolved_num_workers,
                dataset_backend=cfg.compute.dataset_backend,
            )
        worker = EnvWorker(
            dataset=self.dataset,
            env_config=self.env_config,
            encoder_config=self.encoder.config,
            window_config=self.window_config,
            episode_config=self.episode_config,
            algorithm=cfg.algorithm.name,
            model_config=model,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            worker_id=0,
            global_seed=cfg.compute.seed,
            policy=None,
        )
        return SyncCollector(worker)

    def _sync_policy_to_workers(self) -> None:
        self.collector.set_policy(self.policy(), self.value_net())

    def _mean_recent_diag(self) -> dict[str, float]:
        if not self._diag_history:
            return {}
        keys = sorted(set().union(*(d.keys() for d in self._diag_history)))
        out: dict[str, float] = {}
        for k in keys:
            vals = [d[k] for d in self._diag_history if k in d]
            out[k] = float(np.mean(vals)) if vals else 0.0
        self._diag_history.clear()
        return out

    # -- training loop --------------------------------------------------------

    def train(self, *, resume: str | Path | None = None) -> dict[str, Any]:
        try:
            self._setup_signal_handler()
            if resume is not None:
                self._restore_from_checkpoint(resume)
                if self._env_steps >= self.config.training.total_env_steps:
                    target = self.config.training.total_env_steps
                    print(
                        "Resume target:\n"
                        f"  current_env_steps = {self._env_steps}\n"
                        f"  target_env_steps  = {target}\n"
                        "  remaining_steps   = 0\n"
                        "No training performed: checkpoint already meets/exceeds target.",
                        flush=True,
                    )
                    return self._finalize_no_training()
                remaining = self.config.training.total_env_steps - self._env_steps
                print(
                    "Resume target:\n"
                    f"  current_env_steps = {self._env_steps}\n"
                    f"  target_env_steps  = {self.config.training.total_env_steps}\n"
                    f"  remaining_steps   = {remaining}",
                    flush=True,
                )
            # Attach the policy to the collector (workers start acting with it).
            self._sync_policy_to_workers()
            self._log_startup()
            self._start_cpu_sampling()

            # Guarantee a checkpoint exists from the very first step so a run
            # that is interrupted before the first periodic checkpoint interval
            # still leaves something resumable.
            if resume is None and self.checkpoints.latest_path() is None:
                self._save_checkpoint("step_0")

            total = self.config.training.total_env_steps
            bar = make_progress_bar(
                total,
                desc=f"{self.config.algorithm.name.upper()} training",
                unit=" env steps",
                initial=self._env_steps,
            )
            with bar:
                while self._env_steps < total:
                    random_action = self._env_steps < self.config.training.warmup_steps
                    transitions = self.collector.collect(
                        self.config.training.collect_batch, random_action=random_action
                    )
                    self._ingest(transitions)
                    self._consume_transitions(transitions)
                    self._maybe_resync_policy()
                    self._maybe_log()
                    self._maybe_evaluate()
                    self._maybe_checkpoint()
                    bar.update(len(transitions))
                    bar.set_postfix(
                        grad=f"{self._gradient_updates:,}",
                        ret=f"{float(np.mean(self._recent_returns())):+.6f}",
                        **self._progress_postfix(),
                    )

            return self.finalize()
        except BaseException:
            # Rescue: persist whatever state exists so a killed/interrupted run
            # can be resumed.  Re-raise afterwards so the caller sees the error.
            with suppress(Exception):
                self._rescue_checkpoint()
            with suppress(Exception):
                self.collector.close()
            raise

    def _setup_signal_handler(self) -> None:
        """Convert SIGTERM/SIGINT into a catchable KeyboardInterrupt so the
        rescue checkpoint runs before the process is torn down (e.g. Kaggle
        session timeout)."""
        import signal

        def _handle(signum: int, _frame: object) -> None:
            raise KeyboardInterrupt(f"received signal {signum}")

        for sig in (signal.SIGTERM, signal.SIGINT):
            with suppress(ValueError, OSError):  # non-main thread / unsupported
                signal.signal(sig, _handle)

    def _rescue_checkpoint(self) -> None:
        if self._env_steps > 0 or not (self.run_dir / "checkpoints").exists():
            self._save_checkpoint(f"rescue_step_{self._env_steps}")
        self.metric_store.to_csv(self.run_dir / "learning_curve.csv")
        self.metric_store.to_jsonl(self.run_dir / "training_log.jsonl")
        partial = {
            "status": "interrupted",
            "algorithm": self.config.algorithm.name,
            "env_steps": self._env_steps,
            "gradient_updates": self._gradient_updates,
            "episodes": self._episodes,
            "best_checkpoint": self.best_checkpoint,
            "selection_metric_name": self.selection_metric_name,
            "best_selection_score": (
                self.best_score if np.isfinite(self.best_score) else None
            ),
            "best_checkpoint_step": self.best_checkpoint_step,
            "legacy_selection_score": self.legacy_selection_score,
            "warnings": self._collect_warnings(),
        }
        (self.run_dir / "training_summary.json").write_text(
            json.dumps(partial, indent=2, default=str), encoding="utf-8"
        )
        print(f"\n[rescue] saved partial state at env_steps={self._env_steps} in {self.run_dir}")

    def _ingest(self, transitions: list[Transition]) -> None:
        """Bookkeeping for collected transitions (env steps, episode returns)."""
        self._producer_worker_ids.update(t.worker_id for t in transitions)
        for t in transitions:
            worker_id = int(t.worker_id)
            if worker_id not in self._episode_reward_by_worker:
                self._episode_reward_by_worker[worker_id] = 0.0
                self._episode_steps_by_worker[worker_id] = 0
            # Collectors restart episodes on resume. Never attach a saved partial
            # episode's statistics to the first fresh worker trajectory.
            if t.trajectory_step == 0 and self._episode_steps_by_worker[worker_id] > 0:
                self._episode_reward_by_worker[worker_id] = 0.0
                self._episode_steps_by_worker[worker_id] = 0
            self._env_steps += 1
            self._episode_reward_by_worker[worker_id] += t.reward
            self._episode_steps_by_worker[worker_id] += 1
            self._recent_actions.append(t.action)
            self._recent_rewards.append(t.reward)
            if t.terminated or t.truncated:
                self._episode_returns.append(self._episode_reward_by_worker[worker_id])
                self._episode_lengths.append(self._episode_steps_by_worker[worker_id])
                self._episode_reward_by_worker[worker_id] = 0.0
                self._episode_steps_by_worker[worker_id] = 0
                self._episodes += 1
            if len(self._recent_actions) > 10000:
                self._recent_actions = self._recent_actions[-5000:]
                self._recent_rewards = self._recent_rewards[-5000:]

    @abstractmethod
    def _consume_transitions(self, transitions: list[Transition]) -> None:
        """Store transitions and run learner updates (algorithm-specific)."""

    def _maybe_resync_policy(self) -> None:
        """Push the learner policy to the workers (off-policy: occasionally)."""
        interval = self.config.training.policy_sync_interval
        if interval > 0 and self._env_steps % interval < self.config.training.collect_batch:
            self._sync_policy_to_workers()

    def replay_size(self) -> int:
        return 0

    def _record_diagnostics(self, diag: dict[str, float]) -> None:
        self._diag_history.append(diag)

    def _progress_postfix(self) -> dict[str, object]:
        """Algorithm-specific fields shown in the tqdm bar (default: none)."""
        return {}

    # -- schedules ------------------------------------------------------------

    def _maybe_log(self) -> None:
        interval = self.config.logging.log_every_env_steps
        if self._env_steps % interval < self.config.training.collect_batch:
            diag = self._mean_recent_diag()
            fields: dict[str, Any] = {
                "Environment steps": f"{self._env_steps:,}",
                "Gradient updates": f"{self._gradient_updates:,}",
                "Training episodes": f"{self._episodes:,}",
                "Replay size": f"{self.replay_size():,}",
                "Recent mean return": round(float(np.mean(self._recent_returns())), 6),
                "Recent median return": round(float(np.median(self._recent_returns())), 6),
                "Recent mean episode length": round(float(np.mean(self._recent_lengths())), 2),
            }
            for k in ("actor_loss", "critic_loss", "alpha", "entropy", "q1"):
                if k in diag:
                    fields[k] = round(diag[k], 6)
            for k in (
                "rollout_transitions",
                "unique_trajectories",
                "rollout_completed_episodes",
                "advantage_mean",
                "advantage_std",
                "advantage_min",
                "advantage_max",
            ):
                if k in diag:
                    fields[k] = round(diag[k], 6)
            self.metric_store.record(
                env_steps=self._env_steps,
                gradient_updates=self._gradient_updates,
                mean_episode_return=float(np.mean(self._recent_returns())),
                mean_episode_length=float(np.mean(self._recent_lengths())),
                **diag,
            )
            self.logger.progress_block(
                f"{self.config.algorithm.name.upper()} TRAINING PROGRESS", fields
            )
            self._report_cpu_interval()

    def _recent_returns(self, n: int = 200) -> list[float]:
        return self._episode_returns[-n:] or [0.0]

    def _recent_lengths(self, n: int = 200) -> list[int]:
        return self._episode_lengths[-n:] or [0]

    def _validation_specs_path(self) -> Path:
        return self.run_dir / "validation_episode_specs.json"

    @staticmethod
    def _spec_from_dict(value: dict[str, Any]) -> EpisodeSpec:
        return EpisodeSpec(
            instrument=str(value["instrument"]),
            split=str(value["split"]),
            start_index=int(value["start_index"]),
            end_index=int(value["end_index"]),
            horizon=int(value["horizon"]),
            context_length=int(value["context_length"]),
            seed=int(value["seed"]),
        )

    def _validate_fixed_specs(self, specs: list[EpisodeSpec]) -> None:
        expected = self.config.evaluation.validation_episodes
        if len(specs) != expected:
            raise ValueError(
                f"validation specification count {len(specs)} != configured {expected}"
            )
        for spec in specs:
            if (
                spec.split != "validation"
                or spec.horizon != self.config.evaluation.eval_horizon
                or spec.context_length != self.config.environment.context_length
            ):
                raise ValueError("saved validation specifications do not match configuration")
        by_instrument: dict[str, list[EpisodeSpec]] = {}
        for spec in specs:
            by_instrument.setdefault(spec.instrument, []).append(spec)
        for instrument, group in by_instrument.items():
            for i, left in enumerate(group):
                left_start = left.start_index - left.context_length + 1
                for right in group[i + 1 :]:
                    right_start = right.start_index - right.context_length + 1
                    if left_start <= right.end_index and right_start <= left.end_index:
                        raise ValueError(
                            f"overlapping fixed validation specifications for {instrument}"
                        )

    def _write_validation_specs(self) -> None:
        payload = {
            "evaluation_type": "sampled_independent_episodes",
            "selection_schema": SELECTION_SCHEMA,
            "selection_metric_name": self.selection_metric_name,
            "validation_seed": self.config.evaluation.eval_seed,
            "validation_episode_count": len(self.validation_episode_specs),
            "non_overlapping_observation_windows": True,
            "episode_specs": [spec.to_dict() for spec in self.validation_episode_specs],
        }
        self._validation_specs_path().write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def _fixed_validation_specs(self) -> list[EpisodeSpec]:
        if self.validation_episode_specs:
            return self.validation_episode_specs
        path = self._validation_specs_path()
        specs: list[EpisodeSpec] | None = None
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("selection_schema") == SELECTION_SCHEMA
                and payload.get("selection_metric_name") == self.selection_metric_name
                and payload.get("validation_seed") == self.config.evaluation.eval_seed
            ):
                specs = [self._spec_from_dict(value) for value in payload["episode_specs"]]
        if specs is None:
            config = EpisodeConfig(
                split="validation",
                horizon=self.config.evaluation.eval_horizon,
                context_length=self.config.environment.context_length,
                seed=self.config.evaluation.eval_seed,
            )
            specs = EpisodeSampler(self.dataset, config).sample_non_overlapping(
                self.config.evaluation.validation_episodes,
                seed=self.config.evaluation.eval_seed,
                split="validation",
            )
        self._validate_fixed_specs(specs)
        self.validation_episode_specs = specs
        self._write_validation_specs()
        return specs

    def _validation_specs_fingerprint(self) -> str:
        encoded = json.dumps(
            [spec.to_dict() for spec in self._fixed_validation_specs()], sort_keys=True
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _matched_validation_baselines(
        self, evaluator: Any, policy_evaluation: Any, specs: list[EpisodeSpec]
    ) -> dict[str, Any]:
        if self.config.algorithm.name != "ppo":
            return {"available": False, "reason": "matched categorical actions apply to PPO"}
        path = self.run_dir / "validation_baseline_episodes.json"
        fingerprint = self._validation_specs_fingerprint()
        reports = None
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("episode_specs_sha256") == fingerprint:
                reports = cached.get("reports")
        if reports is None:
            reports = evaluator.evaluate_matched_baselines(specs)
            path.write_text(
                json.dumps(
                    {"episode_specs_sha256": fingerprint, "reports": reports}, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
        report = evaluator.matched_baseline_report(
            policy_evaluation, specs, baseline_reports=reports
        )
        return report

    def _maybe_evaluate(self) -> None:
        interval = self.config.logging.evaluate_every_env_steps
        if self._env_steps % interval < self.config.training.collect_batch:
            self._evaluate_validation()

    def _evaluate_validation(self) -> dict[str, Any]:
        from forexmind.training.evaluator import PolicyEvaluator

        evaluator = PolicyEvaluator(
            self.dataset,
            self.env_config,
            self.encoder,
            self.window_config,
            selection_metric=self.selection_metric_name,
            lambda_drawdown=self.config.selection.lambda_drawdown,
            eval_horizon=self.config.evaluation.eval_horizon,
            eval_seed=self.config.evaluation.eval_seed,
            context_length=self.config.environment.context_length,
        )
        specs = self._fixed_validation_specs()
        eval_result = evaluator.evaluate(
            self.policy(),
            self.config.algorithm.name,
            "validation",
            self.config.evaluation.validation_episodes,
            episode_specs=specs,
        )
        score = eval_result.score
        metrics = eval_result.metrics
        rows = cast(list[dict[str, Any]], metrics["episodes"])
        non_jpy = [row["total_return"] for row in rows if row["instrument"] != "USDJPY"]
        positive_profit = sum(max(row["total_return"], 0.0) for row in rows)
        jpy_positive_profit = sum(
            max(row["total_return"], 0.0)
            for row in rows
            if row["instrument"] == "USDJPY"
        )
        baseline_report = self._matched_validation_baselines(evaluator, eval_result, specs)
        entry = {
            "env_steps": self._env_steps,
            "gradient_updates": self._gradient_updates,
            "validation_selection_score": score,
            "selection_metric_name": self.selection_metric_name,
            "mean_episode_log_return": metrics["mean_episode_log_return"],
            "mean_episode_return": metrics["mean_episode_return"],
            "median_episode_return": metrics["median_episode_return"],
            "episode_return_std": metrics["episode_return_std"],
            "p10_episode_return": metrics["p10_episode_return"],
            "p25_episode_return": metrics["p25_episode_return"],
            "p75_episode_return": metrics["p75_episode_return"],
            "p90_episode_return": metrics["p90_episode_return"],
            "profitable_episode_fraction": metrics["profitable_episode_fraction"],
            "worst_episode_return": metrics["worst_episode_return"],
            "best_episode_return": metrics["best_episode_return"],
            "mean_turnover_per_episode": metrics["mean_turnover_per_episode"],
            "mean_executions_per_episode": metrics["mean_executions_per_episode"],
            "mean_episode_return_excluding_usdjpy": (
                float(np.mean(non_jpy)) if non_jpy else None
            ),
            "usdjpy_contribution_to_total_episode_profit": (
                jpy_positive_profit / positive_profit if positive_profit else None
            ),
            "overlap_diagnostics": metrics["overlap_diagnostics"],
            "per_instrument": metrics["per_instrument"],
            "actions": metrics["actions"],
            "matched_baselines": baseline_report,
            "cross_episode_mean_return_series": metrics[
                "cross_episode_mean_return_series"
            ],
        }
        self.validation_history.append(entry)
        self.latest_validation_summary = entry
        val_fields = {
            f"val_{k}": v
            for k, v in entry.items()
            if k
            in {
                "validation_selection_score",
                "mean_episode_log_return",
                "mean_episode_return",
                "median_episode_return",
                "episode_return_std",
                "p10_episode_return",
                "p90_episode_return",
                "profitable_episode_fraction",
                "mean_turnover_per_episode",
                "mean_executions_per_episode",
                "mean_episode_return_excluding_usdjpy",
                "usdjpy_contribution_to_total_episode_profit",
            }
        }
        self.metric_store.record(
            env_steps=self._env_steps,
            gradient_updates=self._gradient_updates,
            **val_fields,
        )

        if score > self.best_score:
            self.best_score = score
            self.best_checkpoint = "best"
            self.best_checkpoint_step = self._env_steps
            self._save_checkpoint("best")
            self.logger.progress_block(
                f"{self.config.algorithm.name.upper()} EVALUATION",
                {
                    "Environment steps": f"{self._env_steps:,}",
                    "Gradient updates": f"{self._gradient_updates:,}",
                    "Training episodes": f"{self._episodes:,}",
                    "Validation episodes": self.config.evaluation.validation_episodes,
                    "Selection metric": self.selection_metric_name,
                    "Mean episode log return": round(
                        _to_float(entry.get("mean_episode_log_return")), 8
                    ),
                    "Mean episode return": round(
                        _to_float(entry.get("mean_episode_return")), 8
                    ),
                    "Median episode return": round(
                        _to_float(entry.get("median_episode_return")), 8
                    ),
                    "Profitable episodes": round(
                        _to_float(entry.get("profitable_episode_fraction")), 4
                    ),
                    "Mean turnover / episode": round(
                        _to_float(entry.get("mean_turnover_per_episode")), 4
                    ),
                    "Score": round(score, 4),
                    "Best selection score": round(self.best_score, 8),
                    "Best checkpoint step": self.best_checkpoint_step,
                    "Current checkpoint": self.best_checkpoint,
                },
            )
        return entry

    def _maybe_checkpoint(self) -> None:
        interval = self.config.logging.checkpoint_every_env_steps
        if self._env_steps % interval < self.config.training.collect_batch:
            self._save_checkpoint(f"step_{self._env_steps}")

    def _save_checkpoint(self, tag: str) -> None:
        from forexmind.training.checkpoint import build_checkpoint_state

        specs = self._fixed_validation_specs()
        current_score = self.latest_validation_summary.get("validation_selection_score")
        best_score = self.best_score if np.isfinite(self.best_score) else None
        state = build_checkpoint_state(
            algorithm=self.config.algorithm.name,
            policy_state={
                k: v.detach().cpu().numpy() for k, v in self.policy().state_dict().items()
            },
            critic_states=self.state_dicts().get("critics"),
            target_states=self.state_dicts().get("targets"),
            optimizers=self.state_dicts().get("optimizers", {}),
            log_alpha=self.state_dicts().get("log_alpha"),
            env_steps=self._env_steps,
            gradient_updates=self._gradient_updates,
            episodes=self._episodes,
            config=self.config,
            dataset_version=self.config.dataset_version,
            # Never populate the ambiguous legacy field in a new-schema checkpoint.
            best_validation_score=None,
            selection_schema=SELECTION_SCHEMA,
            selection_metric_name=self.selection_metric_name,
            validation_selection_score=current_score,
            best_selection_score=best_score,
            best_checkpoint_step=self.best_checkpoint_step,
            validation_seed=self.config.evaluation.eval_seed,
            validation_episode_count=self.config.evaluation.validation_episodes,
            validation_episode_specs=[spec.to_dict() for spec in specs],
            validation_summary=self.latest_validation_summary,
            legacy_selection_score=self.legacy_selection_score,
            best_checkpoint=self.best_checkpoint,
            validation_history=self.validation_history,
            trainer_state={
                "episode_reward_by_worker": dict(self._episode_reward_by_worker),
                "episode_steps_by_worker": dict(self._episode_steps_by_worker),
                "episode_returns": list(self._episode_returns),
                "episode_lengths": list(self._episode_lengths),
            },
            rng_state={
                "replay_meta": self.state_dicts().get("replay_meta", {}),
                "numpy_global": np.random.get_state(),
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "trainer_local": self._local_rng_state(),
            },
        )
        state.update(
            mean_episode_return=self.latest_validation_summary.get("mean_episode_return"),
            median_episode_return=self.latest_validation_summary.get("median_episode_return"),
            profitable_episode_fraction=self.latest_validation_summary.get(
                "profitable_episode_fraction"
            ),
            validation_seed=self.config.evaluation.eval_seed,
            validation_episode_count=len(specs),
        )
        self.checkpoints.save(tag, state)

    def _restore_from_checkpoint(self, path: str | Path) -> None:
        state = self.checkpoints.load(path)
        self.load_state(state)
        self._env_steps = int(state.get("env_steps", 0))
        self._gradient_updates = int(state.get("gradient_updates", 0))
        self._episodes = int(state.get("episodes", 0))
        if state.get("selection_schema") == SELECTION_SCHEMA:
            metric = state.get("selection_metric_name")
            from forexmind.training.evaluator import VALID_SELECTION_METRICS

            if metric not in VALID_SELECTION_METRICS:
                raise ValueError(
                    f"checkpoint has invalid selection metric {metric!r} for {SELECTION_SCHEMA}"
                )
            self.selection_metric_name = str(metric)
            best_score = state.get("best_selection_score")
            self.best_score = float(best_score) if best_score is not None else -float("inf")
            self.best_checkpoint = state.get("best_checkpoint")
            best_step = state.get("best_checkpoint_step")
            self.best_checkpoint_step = int(best_step) if best_step is not None else None
            self.latest_validation_summary = dict(state.get("validation_summary", {}) or {})
            self.validation_history = list(state.get("validation_history", []) or [])
            saved_specs = list(state.get("validation_episode_specs", []) or [])
            if saved_specs:
                self.validation_episode_specs = [
                    self._spec_from_dict(dict(value)) for value in saved_specs
                ]
                self._validate_fixed_specs(self.validation_episode_specs)
                self._write_validation_specs()
        else:
            old_score = state.get("legacy_selection_score")
            if old_score is None:
                old_score = state.get("best_validation_score")
            self.legacy_selection_score = (
                float(old_score) if isinstance(old_score, (int, float)) else None
            )
            self.legacy_validation_history = list(state.get("validation_history", []) or [])
            self.best_score = -float("inf")
            self.best_checkpoint = None
            self.best_checkpoint_step = None
            self.latest_validation_summary = {}
            self.validation_history = []
            print(
                "[resume] checkpoint used the legacy synthetic evaluator; "
                f"legacy_selection_score={self.legacy_selection_score!r}. "
                "New episode-level selection starts unavailable and will be "
                "established at the next validation.",
                flush=True,
            )
        trainer_state = state.get("trainer_state", {}) or {}
        rewards = trainer_state.get("episode_reward_by_worker", {}) or {}
        steps = trainer_state.get("episode_steps_by_worker", {}) or {}
        self._episode_reward_by_worker = {int(k): float(v) for k, v in dict(rewards).items()}
        self._episode_steps_by_worker = {int(k): int(v) for k, v in dict(steps).items()}
        self._episode_returns = [float(v) for v in trainer_state.get("episode_returns", []) or []]
        self._episode_lengths = [int(v) for v in trainer_state.get("episode_lengths", []) or []]
        self._restore_rng_state(state.get("rng", {}) or {})
        print(f"Resumed from {path} at env_steps={self._env_steps}")

    def _local_rng_state(self) -> dict[str, Any]:
        return {}

    def _load_local_rng_state(self, state: dict[str, Any]) -> None:
        return None

    def _restore_rng_state(self, rng_state: dict[str, Any]) -> None:
        if not rng_state:
            print("[resume] RNG state unavailable in checkpoint; using freshly seeded RNGs")
            return
        try:
            if "numpy_global" in rng_state:
                np.random.set_state(rng_state["numpy_global"])
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state.get("torch_cuda"):
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
            self._load_local_rng_state(dict(rng_state.get("trainer_local", {}) or {}))
        except Exception as exc:
            print(f"[resume] could not restore full RNG state ({exc}); continuing with seeded RNGs")

    def _finalize_no_training(self) -> dict[str, Any]:
        workers_alive = getattr(self.collector, "alive_workers", 0)
        self.collector.close()
        if self.best_checkpoint is None:
            self._evaluate_validation()
        summary = {
            "status": "already_complete",
            "algorithm": self.config.algorithm.name,
            "env_steps": self._env_steps,
            "gradient_updates": self._gradient_updates,
            "episodes": self._episodes,
            "best_checkpoint": self.best_checkpoint,
            "selection_metric_name": self.selection_metric_name,
            "best_selection_score": (
                self.best_score if np.isfinite(self.best_score) else None
            ),
            "best_checkpoint_step": self.best_checkpoint_step,
            "legacy_selection_score": self.legacy_selection_score,
            "wall_seconds": 0.0,
            "steps_per_second": 0.0,
            "workers_configured": self._resolved_num_workers,
            "workers_alive_at_finalize": workers_alive,
            "workers_producing_transitions": len(self._producer_worker_ids),
            "dataset_backend": getattr(self, "dataset_backend", "unknown"),
            "warnings": self._collect_warnings(),
            "message": "No training performed: checkpoint already meets/exceeds target.",
        }
        self.metric_store.to_csv(self.run_dir / "learning_curve.csv")
        self.metric_store.to_jsonl(self.run_dir / "training_log.jsonl")
        (self.run_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        return summary

    def _log_startup(self) -> None:
        from forexmind.training.data import dataset_summary
        from forexmind.training.runtime_diagnostics import (
            cpu_affinity,
            logical_cpu_count,
            thread_env_report,
            torch_thread_report,
        )

        ds = dataset_summary(self.dataset)
        worker_pids = getattr(self.collector, "worker_pids", [])
        alive_workers = getattr(self.collector, "alive_workers", 0)
        torch_report = torch_thread_report()
        thread_env = thread_env_report()
        affinity = cpu_affinity()
        self.logger.progress_block(
            f"{self.config.algorithm.name.upper()} TRAINING START",
            {
                "Run dir": str(self.run_dir),
                "Device": str(self.device),
                "Workers": self.config.compute.num_workers,
                "Backend": self.config.compute.collect_backend,
                "obs_dim": self.obs_dim,
                "Dataset loaded": ds["dataset_loaded"],
                "Instruments": ds["num_instruments"],
                "M5 rows": f"{ds['total_m5_rows']:,}",
                "Estimated memory MB": ds["estimated_memory_mb"],
                "Seed": self.config.compute.seed,
                "Logical CPUs": logical_cpu_count(),
                "CPU affinity count": len(affinity) if affinity is not None else "unknown",
                "Torch threads": torch_report,
                "OMP_NUM_THREADS": thread_env["OMP_NUM_THREADS"],
                "MKL_NUM_THREADS": thread_env["MKL_NUM_THREADS"],
                "OPENBLAS_NUM_THREADS": thread_env["OPENBLAS_NUM_THREADS"],
                "Worker PIDs": worker_pids,
                "Alive workers": alive_workers,
                "Learning config": (
                    f"lr={self.config.training.learning_rate} "
                    "actor_lr="
                    f"{self.config.training.actor_lr or self.config.training.learning_rate} "
                    "critic_lr="
                    f"{self.config.training.critic_lr or self.config.training.learning_rate} "
                    f"max_grad_norm={self.config.training.max_grad_norm} "
                    f"finite_check={self.config.training.finite_check}"
                ),
                "Dataset backend": getattr(self, "dataset_backend", "unknown"),
            },
        )
        from forexmind.training.runtime_diagnostics import print_memory_report

        print_memory_report(
            worker_pids=worker_pids,
            workers_configured=self._resolved_num_workers,
            label="STARTUP MEMORY",
        )

    # -- finalization ---------------------------------------------------------

    def finalize(self) -> dict[str, Any]:
        from forexmind.training.runtime_diagnostics import (
            memory_report,
            print_memory_report,
            print_process_tree_report,
        )

        # Capture the final process-tree CPU AND memory sample BEFORE closing
        # workers (once workers are joined their CPU/RSS is gone).
        cpu_report: dict[str, object] = {}
        sampler = self._cpu_sampler
        if sampler is not None and sampler.has_procs:
            cpu_report = dict(sampler.stop())
            print_process_tree_report(
                worker_pids=getattr(self.collector, "worker_pids", []),
                workers_configured=self._resolved_num_workers
                if self.config.compute.collect_backend == "process"
                else 0,
                sample_seconds=0.0,
                cpu_sample=cpu_report,
                producer_ids=self._producer_worker_ids,
                label="FINAL PROCESS TREE",
            )
        mem_report = dict(
            memory_report(
                trainer_pid=os.getpid(),
                worker_pids=getattr(self.collector, "worker_pids", []),
            )
        )
        print_memory_report(
            worker_pids=getattr(self.collector, "worker_pids", []),
            workers_configured=self._resolved_num_workers
            if self.config.compute.collect_backend == "process"
            else 0,
            label="FINAL MEMORY",
        )
        workers_alive = getattr(self.collector, "alive_workers", 0)
        self.collector.close()
        if self.best_checkpoint is None:
            # A best checkpoint must always be earned by the valid selector.
            self._evaluate_validation()
        # Always leave a final checkpoint so the completed run is resumable /
        # inspectable even if the periodic intervals never aligned with the end.
        self._save_checkpoint("final")
        wall = time.perf_counter() - self._start_wall
        self.metric_store.to_csv(self.run_dir / "learning_curve.csv")
        self.metric_store.to_jsonl(self.run_dir / "training_log.jsonl")
        summary = {
            "status": "completed",
            "algorithm": self.config.algorithm.name,
            "env_steps": self._env_steps,
            "gradient_updates": self._gradient_updates,
            "episodes": self._episodes,
            "best_checkpoint": self.best_checkpoint,
            "selection_metric_name": self.selection_metric_name,
            "best_selection_score": (
                self.best_score if np.isfinite(self.best_score) else None
            ),
            "best_checkpoint_step": self.best_checkpoint_step,
            "legacy_selection_score": self.legacy_selection_score,
            "wall_seconds": round(wall, 2),
            "steps_per_second": round(self._env_steps / wall, 1) if wall > 0 else 0.0,
            "workers_configured": self._resolved_num_workers,
            "workers_alive_at_finalize": workers_alive,
            "workers_producing_transitions": len(self._producer_worker_ids),
            "dataset_backend": getattr(self, "dataset_backend", "unknown"),
            "trainer_rss_mb": mem_report.get("trainer_rss_mb", 0.0),
            "worker_rss_aggregate_mb": mem_report.get("worker_rss_aggregate_mb", 0.0),
            "worker_uss_aggregate_mb": mem_report.get("worker_uss_aggregate_mb"),
            "process_tree_rss_mb": mem_report.get("process_tree_rss_mb", 0.0),
            "trainer_cpu_percent": cpu_report.get("trainer_cpu_percent", 0.0),
            "worker_cpu_percent": cpu_report.get("worker_cpu_percent", 0.0),
            "process_tree_cpu_percent": cpu_report.get("process_tree_cpu_percent", 0.0),
            "effective_cores_utilized": cpu_report.get("effective_cores_utilized", 0.0),
            "warnings": self._collect_warnings(),
        }
        (self.run_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        return summary

    def _collect_warnings(self) -> list[str]:
        return pathological_warnings(
            {
                "action_distribution": action_distribution(
                    np.asarray(self._recent_actions, dtype=float)
                ),
                "mean_reward": (
                    float(np.mean(self._recent_rewards)) if self._recent_rewards else 0.0
                ),
            }
        )
