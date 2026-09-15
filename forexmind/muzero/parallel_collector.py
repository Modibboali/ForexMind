"""Parallel MuZero collection with batched inference (Stage 4.6, S3-S8, S19-S21).

Producer architecture (brief S6)::

    collector worker 0 (env, RNG, MCTS tree, causal episodes) -+
    collector worker 1 (env, RNG, MCTS tree, causal episodes) -+--> trajectory queue
    collector worker N-1                                       -+        (bounded)
                                                                          |
                                                              replay writer thread
                                                                          |
                                                             TrajectoryReplayBuffer

and, when ``inference_mode == "server"``, every worker's search sends its leaf
requests to the central :class:`~forexmind.muzero.inference_service.BatchedInferenceServer`
running in the trainer process, so no worker holds a model copy (brief S11).

Design rules honoured here:

* **Isolation** (S3): a worker owns its environments, RNG streams and search
  trees; nothing mutable is shared between workers.
* **Deterministic seeding** (S4): episode/search/decision seeds are derived from
  ``(global_seed, worker_rank, episode_index, decision_index)``, so worker 0 and
  worker 1 never produce the same stream and a fixed global seed reproduces the
  same worker-level streams.
* **No dataset duplication** (S5): workers open the shared memory-mapped store
  when it exists, so the OS shares the physical pages.
* **Bounded queues and backpressure** (S7): the trajectory queue is bounded; a
  worker that finds it full blocks (and reports its blocked time) instead of
  buffering without limit.  Trajectories are only dropped if the pool is
  explicitly configured to drop them.
* **One writer** (S19): collectors never mutate replay; a single writer thread
  owns insertion, so capacity enforcement stays consistent.
* **Failures are loud** (S37, S38): a dead worker or a broken inference service
  raises instead of silently continuing with missing collectors.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forexmind.data.splits import DEFAULT_INSTRUMENT_ORDER, SplitConfig, SplitDataset
from forexmind.muzero.collector import (
    CollectedTrajectory,
    CollectionStats,
    CollectorConfig,
    EpisodeRun,
    MuZeroCollector,
    derive_decision_seed,
)
from forexmind.muzero.diagnostics import RootSearchRecord
from forexmind.muzero.inference_service import (
    BatchedInferenceServer,
    InferenceServiceError,
    InferenceStats,
    LocalInferenceBackend,
    RemoteInferenceBackend,
)
from forexmind.muzero.profiling import PhaseTimer, merge_phase_reports
from forexmind.muzero.replay import TrajectoryReplayBuffer
from forexmind.muzero.trajectory import MuZeroTrajectory, TrajectoryMetadata

__all__ = [
    "CollectorPoolConfig",
    "CollectorWorkerError",
    "LockedReplay",
    "MuZeroCollectorPool",
    "ReplayWriter",
    "WorkerDatasetSpec",
    "trajectory_from_payload",
    "trajectory_to_payload",
]


class CollectorWorkerError(RuntimeError):
    """Raised when a collector worker dies unexpectedly (brief S37)."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class CollectorPoolError(RuntimeError):
    """Raised for pool lifecycle errors (timeouts, double close, ...)."""


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class WorkerDatasetSpec:
    """How a worker should build its dataset (never the dataset itself).

    Workers rebuild the dataset from a *description* so nothing large is pickled
    per process; ``backend="auto"`` prefers the shared memory-mapped store so all
    workers map the same physical pages (brief S5).
    """

    processed_dir: str
    split_config: dict[str, str]
    instruments: tuple[str, ...] = DEFAULT_INSTRUMENT_ORDER
    backend: str = "auto"

    @classmethod
    def from_dataset(
        cls,
        dataset: SplitDataset,
        *,
        processed_dir: str | Path,
        backend: str = "auto",
    ) -> WorkerDatasetSpec:
        return cls(
            processed_dir=str(processed_dir),
            split_config=dataset.split_config.to_dict(),
            instruments=tuple(dataset.instruments),
            backend=backend,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed_dir": self.processed_dir,
            "split_config": dict(self.split_config),
            "instruments": list(self.instruments),
            "backend": self.backend,
        }


@dataclass(frozen=True, slots=True)
class CollectorPoolConfig:
    """Everything that controls parallel collection (nothing implicit)."""

    num_workers: int = 4
    #: Episodes driven in lock-step inside one worker (batched MCTS roots).
    collectors_per_worker: int = 2
    max_inference_batch_size: int = 32
    max_batch_wait_ms: float = 2.0
    inference_mode: str = "server"  # "server" | "local"
    #: Bounded trajectory queue: backpressure instead of unbounded RAM (S7).
    trajectory_queue_size: int = 8
    #: Drop trajectories when the queue is full instead of blocking (S7).
    drop_when_full: bool = False
    worker_stop_timeout_s: float = 60.0
    inference_request_timeout_s: float = 120.0
    torch_threads_per_worker: int = 1
    per_decision_seed: bool = True
    #: Wall-time phase profiling inside the workers (brief S2).
    profile: bool = False
    #: Episode index space is partitioned per worker: worker ``w`` uses indices
    #: ``w * episode_stride + k`` so ids are unique and deterministic (S4).
    episode_stride: int = 1_000_000
    #: Shift all episode indices (used on resume so new trajectory ids cannot
    #: collide with trajectories restored from the replay store).
    episode_offset: int = 0

    def __post_init__(self) -> None:
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")
        if self.collectors_per_worker < 1:
            raise ValueError(
                f"collectors_per_worker must be >= 1, got {self.collectors_per_worker}"
            )
        if self.max_inference_batch_size < 1:
            raise ValueError("max_inference_batch_size must be >= 1")
        if self.max_batch_wait_ms < 0.0:
            raise ValueError("max_batch_wait_ms must be >= 0")
        if self.inference_mode not in ("server", "local"):
            raise ValueError("inference_mode must be 'server' or 'local'")
        if self.trajectory_queue_size < 1:
            raise ValueError("trajectory_queue_size must be >= 1")
        if self.torch_threads_per_worker < 1:
            raise ValueError("torch_threads_per_worker must be >= 1")
        if self.episode_stride <= self.collectors_per_worker:
            raise ValueError("episode_stride must exceed collectors_per_worker")

    @property
    def num_collectors(self) -> int:
        return self.num_workers * self.collectors_per_worker

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "collectors_per_worker": self.collectors_per_worker,
            "num_collectors": self.num_collectors,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_batch_wait_ms": self.max_batch_wait_ms,
            "inference_mode": self.inference_mode,
            "trajectory_queue_size": self.trajectory_queue_size,
            "drop_when_full": self.drop_when_full,
            "torch_threads_per_worker": self.torch_threads_per_worker,
            "profile": self.profile,
            "per_decision_seed": self.per_decision_seed,
            "episode_stride": self.episode_stride,
            "worker_stop_timeout_s": self.worker_stop_timeout_s,
            "inference_request_timeout_s": self.inference_request_timeout_s,
        }


# --------------------------------------------------------------------------- #
# wire format
# --------------------------------------------------------------------------- #


def _record_to_payload(record: RootSearchRecord) -> dict[str, Any]:
    return {
        "prior": record.prior,
        "visits": record.visits,
        "policy": record.policy,
        "q_values": record.q_values,
        "predicted_rewards": record.predicted_rewards,
        "mask": record.mask,
        "action": int(record.action),
        "network_value": float(record.network_value),
        "search_value": float(record.search_value),
        "tree_depth": int(record.tree_depth),
        "real_reward": None if record.real_reward is None else float(record.real_reward),
    }


def _record_from_payload(payload: dict[str, Any]) -> RootSearchRecord:
    return RootSearchRecord(
        prior=np.asarray(payload["prior"], dtype=np.float64),
        visits=np.asarray(payload["visits"], dtype=np.float64),
        policy=np.asarray(payload["policy"], dtype=np.float64),
        q_values=np.asarray(payload["q_values"], dtype=np.float64),
        predicted_rewards=np.asarray(payload["predicted_rewards"], dtype=np.float64),
        mask=np.asarray(payload["mask"], dtype=bool),
        action=int(payload["action"]),
        network_value=float(payload["network_value"]),
        search_value=float(payload["search_value"]),
        tree_depth=int(payload["tree_depth"]),
        real_reward=payload["real_reward"],
    )


def trajectory_to_payload(
    trajectory: MuZeroTrajectory,
    *,
    records: list[RootSearchRecord] | None = None,
    worker_id: int = -1,
    network_version: int | None = None,
) -> dict[str, Any]:
    """Compact, picklable wire payload (NumPy arrays, never Python objects)."""
    return {
        "observations": trajectory.observations,
        "actions": trajectory.actions,
        "rewards": trajectory.rewards,
        "root_policies": trajectory.root_policies,
        "root_values": trajectory.root_values,
        "action_masks": trajectory.action_masks,
        "terminated": trajectory.terminated,
        "truncated": trajectory.truncated,
        "planning_exposure": trajectory.planning_exposure,
        "planning_is_flat": trajectory.planning_is_flat,
        "boundary_value": float(trajectory.boundary_value),
        "metadata": trajectory.metadata.to_dict(),
        "extra": dict(trajectory.extra),
        "records": [_record_to_payload(record) for record in (records or [])],
        "worker_id": int(worker_id),
        "network_version": (
            int(trajectory.metadata.network_version)
            if network_version is None
            else int(network_version)
        ),
        "produced_at": time.time(),
    }


def trajectory_from_payload(
    payload: dict[str, Any],
) -> tuple[MuZeroTrajectory, list[RootSearchRecord]]:
    """Rebuild a trajectory (and its optional diagnostics) from the wire."""
    trajectory = MuZeroTrajectory(
        observations=np.asarray(payload["observations"], dtype=np.float32),
        actions=np.asarray(payload["actions"], dtype=np.int64),
        rewards=np.asarray(payload["rewards"], dtype=np.float32),
        root_policies=np.asarray(payload["root_policies"], dtype=np.float32),
        root_values=np.asarray(payload["root_values"], dtype=np.float32),
        action_masks=np.asarray(payload["action_masks"], dtype=bool),
        terminated=np.asarray(payload["terminated"], dtype=bool),
        truncated=np.asarray(payload["truncated"], dtype=bool),
        # float64 on both sides of the wire: rounding this can flip a six-action
        # mask near the exposure tolerance (see collector.EpisodeRun.finish).
        planning_exposure=np.asarray(payload["planning_exposure"], dtype=np.float64),
        planning_is_flat=np.asarray(payload["planning_is_flat"], dtype=bool),
        boundary_value=float(payload["boundary_value"]),
        metadata=TrajectoryMetadata(**payload["metadata"]),
        extra=dict(payload.get("extra", {})),
    )
    # Keep the production timestamp: benchmarks and diagnostics measure
    # throughput on the *production* window, not on when the trainer happened to
    # collect the trajectory from the queue (brief S28).
    trajectory.extra["produced_at"] = float(payload.get("produced_at", 0.0))
    trajectory.extra["producer_worker_id"] = int(payload.get("worker_id", -1))
    records = [_record_from_payload(item) for item in payload.get("records", [])]
    return trajectory, records


def payload_nbytes(payload: dict[str, Any]) -> int:
    """Total array bytes in a payload (serialization cost estimate, S20)."""
    total = 0
    for value in payload.values():
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
    for item in payload.get("records", []):
        for value in item.values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
    return total


# --------------------------------------------------------------------------- #
# replay writer / locked replay
# --------------------------------------------------------------------------- #


class LockedReplay:
    """Thread-safe view of a :class:`TrajectoryReplayBuffer`.

    The replay writer thread appends while the trainer samples, so both sides
    take the same lock.  Sampling is therefore never observed half-inserted
    (brief S19) and the writer cannot corrupt the buffer's offsets.
    """

    def __init__(self, buffer: TrajectoryReplayBuffer) -> None:
        self._buffer = buffer
        self.lock = threading.RLock()

    @property
    def buffer(self) -> TrajectoryReplayBuffer:
        return self._buffer

    @property
    def config(self) -> Any:
        return self._buffer.config

    def __len__(self) -> int:
        with self.lock:
            return len(self._buffer)

    @property
    def num_trajectories(self) -> int:
        with self.lock:
            return self._buffer.num_trajectories

    @property
    def num_transitions(self) -> int:
        with self.lock:
            return self._buffer.num_transitions

    @property
    def total_positions(self) -> int:
        with self.lock:
            return self._buffer.total_positions

    @property
    def trajectories(self) -> tuple[MuZeroTrajectory, ...]:
        with self.lock:
            return self._buffer.trajectories

    def add(self, trajectory: MuZeroTrajectory, **kwargs: Any) -> None:
        with self.lock:
            self._buffer.add(trajectory, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            return self._buffer.sample(*args, **kwargs)

    def clear(self) -> None:
        with self.lock:
            self._buffer.clear()

    def memory_report(self) -> dict[str, Any]:
        with self.lock:
            return self._buffer.memory_report()

    def sampling_diagnostics(self) -> dict[str, Any]:
        with self.lock:
            return self._buffer.sampling_diagnostics()

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return self._buffer.to_dict()


class ReplayWriter:
    """Single thread that drains the trajectory queue into replay (brief S19).

    Collectors never touch replay; they only hand over payloads.  The writer
    records the insertion cost, the queue occupancy it saw and the end-to-end
    handoff latency (production timestamp to insertion), so queue/backpressure
    behaviour is measurable (brief S7, S20).
    """

    def __init__(
        self,
        *,
        trajectory_queue: Any,
        stats_queue: Any,
        replay: LockedReplay,
        require_split: str = "train",
    ) -> None:
        self.trajectory_queue = trajectory_queue
        self.stats_queue = stats_queue
        self.replay = replay
        self.require_split = require_split
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pending: list[CollectedTrajectory] = []
        self._pending_signal = threading.Condition(self._lock)
        self._worker_stats: dict[int, dict[str, Any]] = {}
        self._last_trajectory: dict[int, int] = {}
        #: Authoritative per-worker production counts, taken from each payload
        #: as it is inserted (worker-reported snapshots can lag by a round).
        self.written_by_worker: dict[int, int] = {}
        self.trajectories_written = 0
        self.transitions_written = 0
        self.dropped = 0
        self.insertion_seconds = 0.0
        self.handoff_seconds = 0.0
        self.handoff_samples = 0
        self.payload_bytes = 0
        self.failure: BaseException | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise CollectorPoolError("replay writer already started")
        self._thread = threading.Thread(target=self._run, name="muzero-replay-writer", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 30.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
            if thread.is_alive():  # pragma: no cover - defensive
                raise CollectorPoolError("replay writer did not stop")
        self._thread = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- writer loop ----------------------------------------------------------

    def _run(self) -> None:  # pragma: no cover - threaded, exercised via the pool
        while not self._stop.is_set():
            processed = self._drain_stats()
            try:
                payload = self.trajectory_queue.get(timeout=0.02)
            except queue.Empty:
                continue
            except (EOFError, OSError):  # pragma: no cover - queue torn down
                break
            if payload is None:
                continue
            self._write(payload)
            del processed
        # Final flush so a clean shutdown never loses a produced trajectory.
        self._drain_stats()
        while True:
            try:
                payload = self.trajectory_queue.get_nowait()
            except (queue.Empty, EOFError, OSError):
                break
            if payload is None:
                continue
            self._write(payload)

    def _drain_stats(self) -> int:
        drained = 0
        while True:
            try:
                message = self.stats_queue.get_nowait()
            except (queue.Empty, EOFError, OSError):
                break
            if not isinstance(message, dict):
                continue
            worker_id = int(message.get("worker_id", -1))
            # Merge, never replace: a later "stopped" message must not erase the
            # counters the worker already reported (brief S37 diagnostics).
            self._worker_stats[worker_id] = {**self._worker_stats.get(worker_id, {}), **message}
            if message.get("last_trajectory_id") is not None:
                self._last_trajectory[worker_id] = int(message["last_trajectory_id"])
            drained += 1
        return drained

    def drain(self) -> int:
        """Synchronously consume any pending worker stats messages."""
        return self._drain_stats()

    def _write(self, payload: dict[str, Any]) -> None:
        start = time.perf_counter()
        trajectory, records = trajectory_from_payload(payload)
        self.replay.add(trajectory, require_split=self.require_split)
        elapsed = time.perf_counter() - start
        producer = int(payload.get("worker_id", -1))
        self.written_by_worker[producer] = self.written_by_worker.get(producer, 0) + 1
        produced_at = float(payload.get("produced_at", 0.0))
        if produced_at > 0.0:
            self.handoff_seconds += max(0.0, time.time() - produced_at)
            self.handoff_samples += 1
        self.insertion_seconds += elapsed
        self.payload_bytes += payload_nbytes(payload)
        self.trajectories_written += 1
        self.transitions_written += len(trajectory)
        with self._pending_signal:
            self._pending.append(CollectedTrajectory(trajectory=trajectory, records=records))
            self._pending_signal.notify_all()

    # -- consumption ----------------------------------------------------------

    def take(self, count: int, *, timeout_s: float = 600.0) -> list[CollectedTrajectory]:
        """Block until ``count`` freshly written trajectories are available."""
        deadline = time.perf_counter() + timeout_s
        with self._pending_signal:
            while len(self._pending) < count:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    raise CollectorPoolError(
                        f"timed out waiting for {count} trajectories; "
                        f"{len(self._pending)} available after {timeout_s:.1f}s"
                    )
                self._pending_signal.wait(timeout=min(remaining, 0.5))
            taken = self._pending[:count]
            del self._pending[:count]
            return taken

    @property
    def pending_count(self) -> int:
        with self._pending_signal:
            return len(self._pending)

    @property
    def worker_stats(self) -> dict[int, dict[str, Any]]:
        return dict(self._worker_stats)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "alive": self.is_alive,
            "trajectories_written": self.trajectories_written,
            "trajectories_written_by_worker": dict(sorted(self.written_by_worker.items())),
            "transitions_written": self.transitions_written,
            "dropped": self.dropped,
            "pending": self.pending_count,
            "insertion_seconds": self.insertion_seconds,
            "mean_insertion_ms": (
                1e3 * self.insertion_seconds / self.trajectories_written
                if self.trajectories_written
                else 0.0
            ),
            "mean_handoff_ms": (
                1e3 * self.handoff_seconds / self.handoff_samples if self.handoff_samples else 0.0
            ),
            "max_handoff_ms": 1e3 * self.handoff_seconds / max(self.handoff_samples, 1),
            "payload_mb": self.payload_bytes / 1e6,
            "workers_reporting": sorted(self._worker_stats),
        }


# --------------------------------------------------------------------------- #
# worker process
# --------------------------------------------------------------------------- #


class _RemoteModelHandle:
    """Stand-in model for workers whose inference lives in the service.

    Search only needs ``config.num_actions`` from it, and the collector takes
    its model fingerprint from the pool, so a remote worker never builds or
    holds a network (brief S11).
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.network_version = 0

    def named_parameters(self) -> Any:  # pragma: no cover - only used by model_version()
        return iter(())


def _build_worker_dataset(spec: dict[str, Any]) -> tuple[SplitDataset, str]:
    from forexmind.training.dataset_mmap import resolve_dataset

    dataset, backend = resolve_dataset(
        processed_dir=spec["processed_dir"],
        split_config=SplitConfig.from_dict(spec["split_config"]),
        instruments=tuple(spec["instruments"]),
        backend=spec.get("backend", "auto"),
    )
    return dataset, backend


def _worker_main(  # pragma: no cover - runs in a subprocess
    cfg: dict[str, Any],
    request_queue: Any,
    response_queue: Any,
    trajectory_queue: Any,
    stats_queue: Any,
    control_queue: Any,
) -> None:
    """Collector worker process: own envs, own RNG, own trees (brief S3-S4)."""
    worker_id = int(cfg["worker_id"])
    mode = str(cfg["inference_mode"])
    try:
        import torch

        print(
            f"[collector-worker] start worker_id={worker_id} pid={os.getpid()} "
            f"parent_pid={os.getppid()} mode={mode}",
            flush=True,
        )
        torch.set_num_threads(int(cfg["torch_threads_per_worker"]))
        # Interop threads can only be set once per process; a worker that cannot
        # change them is fine (it means torch was already initialised).
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)

        dataset, dataset_backend = _build_worker_dataset(cfg["dataset"])
        encoder_config = cfg["encoder_config"]
        env_config = cfg["env_config"]
        collector_config = CollectorConfig(**cfg["collector_config"])
        model_version_string = str(cfg["model_version"])
        episodes_per_worker = cfg.get("episodes_per_worker")
        first_episode = int(cfg["episode_offset"]) + int(cfg["worker_id"]) * int(
            cfg["episode_stride"]
        )

        stats = CollectionStats()
        if mode == "server":
            from forexmind.muzero.config import MuZeroConfig

            model: Any = _RemoteModelHandle(MuZeroConfig(**cfg["model_config"]))
            backend: Any = RemoteInferenceBackend(
                worker_id=worker_id,
                request_queue=request_queue,
                response_queue=response_queue,
                timeout_s=float(cfg["inference_request_timeout_s"]),
                stats=InferenceStats(),
            )
        else:
            from forexmind.muzero.config import MuZeroConfig
            from forexmind.muzero.inference import build_muzero_network

            model = build_muzero_network(MuZeroConfig(**cfg["model_config"]))
            model.eval()
            backend = LocalInferenceBackend(model, stats=InferenceStats())

        collector = MuZeroCollector(
            dataset,
            env_config,
            encoder_config,
            model,
            collector_config,
            instruments=tuple(cfg["instruments"]),
            model_version_string=model_version_string,
        )
        from forexmind.muzero.search import MuZeroMCTS

        driver_backend = backend
        timer = PhaseTimer(enabled=bool(cfg.get("profile")))
        collector.timer = timer
        driver = MuZeroMCTS(
            model,
            collector.search_config,
            rng=np.random.default_rng(1),
            backend=driver_backend,
            timer=timer,
        )

        blocked_seconds = 0.0
        dropped = 0
        current_temperature = float(collector_config.effective_temperature)
        payload_build_seconds = 0.0
        payload_bytes = 0
        episodes_done = 0
        last_trajectory_id: int | None = None
        last_stats_sent = 0.0
        stats_seq = 0
        stop_requested = False
        next_episode = first_episode
        runs: list[EpisodeRun] = []

        def space_for_runs() -> int:
            return int(cfg["collectors_per_worker"]) - len(runs)

        def merged_counters() -> CollectionStats:
            """Completed-run counters plus the in-flight runs' partial counters.

            Trajectories are absorbed into ``stats`` when they are *shipped*, so
            without this the visible counters would jump only at episode ends
            and a profiler would misread a healthy collector as idle.
            """
            merged = CollectionStats(
                trajectories=stats.trajectories,
                env_steps=stats.env_steps,
                searches=stats.searches,
                recurrent_inference_calls=stats.recurrent_inference_calls,
                terminated=stats.terminated,
                truncated=stats.truncated,
                reward_sum=stats.reward_sum,
                action_counts=stats.action_counts.copy(),
            )
            for run in runs:
                merged.absorb(run.stats)
            return merged

        def send_stats(force: bool = False) -> None:
            nonlocal last_stats_sent, stats_seq
            now = time.perf_counter()
            if not force and now - last_stats_sent < 1.0:
                return
            last_stats_sent = now
            stats_seq += 1
            stats_queue.put(
                {
                    "worker_id": worker_id,
                    "stats_seq": stats_seq,
                    "pid": os.getpid(),
                    "dataset_backend": dataset_backend,
                    "episodes": episodes_done,
                    "dropped": dropped,
                    "last_trajectory_id": last_trajectory_id,
                    "blocked_seconds": blocked_seconds,
                    "payload_build_seconds": payload_build_seconds,
                    "payload_bytes": payload_bytes,
                    **merged_counters().to_dict(),
                    "inference": backend.diagnostics(),
                    "profiling": timer.report(),
                }
            )

        while not stop_requested:
            # -- control messages (non-blocking) ------------------------------
            while True:
                try:
                    message = control_queue.get_nowait()
                except (queue.Empty, EOFError, OSError):
                    break
                if message is None:
                    continue
                kind = message.get("kind")
                if kind == "stop":
                    stop_requested = True
                elif kind == "set_weights":
                    if mode == "local":
                        # The parent ships NumPy arrays (cheap to pickle through
                        # the queue); torch requires tensors.
                        model.load_state_dict(
                            {
                                key: torch.as_tensor(value)
                                for key, value in message["state"].items()
                            }
                        )
                        model.network_version = int(message["version"])
                    model_version_string = str(message["model_version"])
                    collector.model_version = model_version_string
                elif kind == "stats":
                    send_stats(force=True)
                elif kind == "set_temperature":
                    from dataclasses import replace as _replace

                    current_temperature = float(message["temperature"])
                    driver.config = _replace(
                        driver.config, temperature=current_temperature
                    )
                else:  # pragma: no cover - defensive
                    raise ValueError(f"unknown control message {kind!r}")
            if stop_requested:
                break

            # -- keep the pool full -------------------------------------------
            while space_for_runs() > 0 and (
                episodes_per_worker is None or episodes_done + len(runs) < episodes_per_worker
            ):
                runs.append(
                    collector.start_run(
                        next_episode,
                        search=driver,
                        backend=driver_backend,
                        worker_rank=int(cfg["worker_rank_base"]) + len(runs),
                        # Each concurrent episode gets its own environment: two
                        # runs must never share mutable environment state (S3).
                        fresh_env=True,
                    )
                )
                next_episode += 1
            if not runs:
                send_stats()
                time.sleep(0.01)
                continue

            # -- one decision for every run, one batched inference call -------
            inputs = [run.decision_inputs() for run in runs]
            rngs = [
                np.random.default_rng(derive_decision_seed(run.search_seed, run.decision_index))
                for run in runs
            ]
            results = driver.search_batch(
                [item[0] for item in inputs],
                [item[1] for item in inputs],
                planning_states=[item[2] for item in inputs],
                rngs=rngs,
                add_root_noise=collector_config.training,
            )
            for run, (_obs, mask, _state), result in zip(runs, inputs, results, strict=True):
                stats.searches += 1
                stats.recurrent_inference_calls += int(
                    result.diagnostics.recurrent_inference_calls
                )
                run.record_decision(
                    result,
                    mask,
                    network_version=backend.network_version,
                    temperature=current_temperature,
                )
                run.advance()

            # -- finish and ship ----------------------------------------------
            finished = [run for run in runs if run.done]
            if finished:
                boundary: list[tuple[EpisodeRun, tuple[np.ndarray, np.ndarray, Any]]] = []
                for run in finished:
                    request = run.boundary_request()
                    if request is not None:
                        boundary.append((run, request))
                if boundary:
                    results = driver.search_batch(
                        [item[1][0] for item in boundary],
                        [item[1][1] for item in boundary],
                        planning_states=[item[1][2] for item in boundary],
                        rngs=[
                            np.random.default_rng(
                                derive_decision_seed(
                                    run.search_seed, run.decision_index + 1_000_003
                                )
                            )
                            for run, _ in boundary
                        ],
                        add_root_noise=False,
                    )
                    for (run, _request), result in zip(boundary, results, strict=True):
                        run.apply_boundary_result(result)
                for run in finished:
                    collected = run.finish()
                    payload_build = time.perf_counter()
                    payload = trajectory_to_payload(
                        collected.trajectory,
                        records=collected.records,
                        worker_id=worker_id,
                        network_version=run.network_version,
                    )
                    payload_build_seconds += time.perf_counter() - payload_build
                    payload_bytes += payload_nbytes(payload)
                    if cfg.get("drop_when_full"):
                        try:
                            trajectory_queue.put(payload, timeout=0.0)
                            queued = True
                        except queue.Full:
                            queued = False
                            dropped += 1
                    else:
                        queued = True
                        while True:
                            try:
                                trajectory_queue.put(payload, timeout=0.25)
                                break
                            except queue.Full:
                                # Bounded queue: block (and account for it) rather
                                # than growing RAM without limit (brief S7).
                                blocked_seconds += 0.25
                    if queued:
                        stats.absorb(run.stats)
                        episodes_done += 1
                        last_trajectory_id = int(collected.trajectory.metadata.trajectory_id)
                    runs.remove(run)
            # Publish immediately after shipping: a short benchmark can finish
            # inside the periodic 1 s window, and its counters must not be lost.
            send_stats(force=bool(finished))
            if episodes_per_worker is not None and episodes_done >= int(episodes_per_worker):
                break
        send_stats(force=True)
    except BaseException as exc:  # pragma: no cover - failure path
        import traceback

        traceback.print_exc()
        with contextlib.suppress(Exception):
            stats_queue.put(
                {
                    "worker_id": worker_id,
                    "pid": os.getpid(),
                    "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        raise
    finally:  # pragma: no cover - subprocess teardown
        print(f"[collector-worker] stop worker_id={worker_id} pid={os.getpid()}", flush=True)
        with contextlib.suppress(Exception):
            stats_queue.put({"worker_id": worker_id, "pid": os.getpid(), "stopped": True})


# --------------------------------------------------------------------------- #
# pool
# --------------------------------------------------------------------------- #


class MuZeroCollectorPool:
    """Collector processes, bounded trajectory queue and replay writer (S3-S8)."""

    def __init__(
        self,
        *,
        config: CollectorPoolConfig,
        dataset_spec: WorkerDatasetSpec,
        env_config: Any,
        encoder_config: Any,
        collector_config: CollectorConfig,
        model_factory: Any,
        model_config: Any,
        model: Any | None = None,
        replay: TrajectoryReplayBuffer | None = None,
        model_version_string: str = "",
        split: str = "train",
        episodes_per_worker: int | None = None,
        inference_device: str | None = None,
    ) -> None:
        self.config = config
        self.dataset_spec = dataset_spec
        self.collector_config = collector_config
        self.model_config = model_config
        self.split = split
        self.episodes_per_worker = episodes_per_worker
        # NB: never ``replay or ...`` - an empty buffer is falsy via __len__.
        self.replay = (
            replay
            if isinstance(replay, LockedReplay)
            else LockedReplay(
                replay if replay is not None else TrajectoryReplayBuffer()
            )
        )
        self.start_time = time.perf_counter()
        self._closed = False
        self._worker_error: BaseException | None = None

        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self.request_queue = ctx.Queue()
        self.response_queues = {wid: ctx.Queue() for wid in range(config.num_workers)}
        self.trajectory_queue = ctx.Queue(maxsize=config.trajectory_queue_size)
        self.stats_queue = ctx.Queue()
        self.control_queues = {wid: ctx.Queue() for wid in range(config.num_workers)}

        self.writer = ReplayWriter(
            trajectory_queue=self.trajectory_queue,
            stats_queue=self.stats_queue,
            replay=self.replay,
            require_split=split,
        )
        self.writer.start()

        self.inference_stats = InferenceStats()
        self.server: BatchedInferenceServer | None = None
        if config.inference_mode == "server":
            self.server = BatchedInferenceServer(
                model_factory,
                request_queue=self.request_queue,
                response_queues=self.response_queues,
                max_batch_size=config.max_inference_batch_size,
                max_batch_wait_ms=config.max_batch_wait_ms,
                stats=self.inference_stats,
                device=inference_device,
            )
            source = model if model is not None else model_factory()
            self.server.sync_weights(
                source, version=0, model_version_string=model_version_string
            )
            self.server.start()

        worker_config = {
            "dataset": dataset_spec.to_dict(),
            "instruments": list(dataset_spec.instruments),
            "env_config": env_config,
            "encoder_config": encoder_config,
            "collector_config": self._collector_config_dict(collector_config),
            "model_config": dict(getattr(model_config, "to_dict", lambda: {})())
            if hasattr(model_config, "to_dict")
            else dict(model_config),
            "inference_mode": config.inference_mode,
            "torch_threads_per_worker": config.torch_threads_per_worker,
            "profile": bool(config.profile),
            "inference_request_timeout_s": config.inference_request_timeout_s,
            "collectors_per_worker": config.collectors_per_worker,
            "episode_stride": config.episode_stride,
            "episode_offset": config.episode_offset,
            "episodes_per_worker": episodes_per_worker,
            "drop_when_full": config.drop_when_full,
            "model_version": model_version_string,
        }
        self._processes: list[Any] = []
        self._worker_configs: list[dict[str, Any]] = []
        for worker_id in range(config.num_workers):
            cfg = dict(
                worker_config,
                worker_id=worker_id,
                worker_rank_base=worker_id * config.collectors_per_worker,
            )
            self._worker_configs.append(cfg)
            process = ctx.Process(
                target=_worker_main,
                args=(
                    cfg,
                    self.request_queue,
                    self.response_queues[worker_id],
                    self.trajectory_queue,
                    self.stats_queue,
                    self.control_queues[worker_id],
                ),
                name=f"muzero-collector-{worker_id}",
            )
            process.start()
            self._processes.append(process)

    @staticmethod
    def _collector_config_dict(config: CollectorConfig) -> dict[str, Any]:
        payload = {
            field_name: getattr(config, field_name)
            for field_name in (
                "split",
                "horizon",
                "num_simulations",
                "discount",
                "training",
                "temperature",
                "dirichlet_alpha",
                "root_exploration_fraction",
                "seed",
                "boundary_search",
                "network_version",
                "capture_diagnostics",
                "per_decision_seed",
            )
        }
        return payload

    # -- worker health --------------------------------------------------------

    @property
    def worker_pids(self) -> list[int]:
        return [int(p.pid) for p in self._processes if p.pid is not None]

    @property
    def alive_workers(self) -> int:
        return sum(1 for p in self._processes if p.is_alive())

    def worker_status(self) -> list[dict[str, Any]]:
        stats = self.writer.worker_stats
        return [
            {
                "worker_id": index,
                "pid": int(p.pid) if p.pid is not None else None,
                "alive": bool(p.is_alive()),
                "exitcode": p.exitcode,
                "episodes": stats.get(index, {}).get("episodes"),
                "last_trajectory_id": stats.get(index, {}).get("last_trajectory_id"),
                "dataset_backend": stats.get(index, {}).get("dataset_backend"),
                "blocked_seconds": stats.get(index, {}).get("blocked_seconds"),
                "error": stats.get(index, {}).get("error"),
            }
            for index, p in enumerate(self._processes)
        ]

    def check_health(self) -> None:
        """Fail loudly on a dead worker or a broken inference service (S37-S38)."""
        if self._worker_error is not None:
            raise self._worker_error
        server = self.server
        if server is not None and server.fatal_error is not None:
            raise InferenceServiceError(
                f"inference server failed: {type(server.fatal_error).__name__}: "
                f"{server.fatal_error}"
            )
        for index, process in enumerate(self._processes):
            if process.is_alive():
                continue
            if process.exitcode == 0 and (
                self._closed or self._worker_finished_quota(index)
            ):
                continue
            stats = self.writer.worker_stats.get(index, {})
            details = {
                "worker_id": index,
                "pid": process.pid,
                "exitcode": process.exitcode,
                "last_trajectory_id": stats.get("last_trajectory_id"),
                "episodes": stats.get("episodes"),
                "error": stats.get("error"),
            }
            error = CollectorWorkerError(
                f"collector worker {index} (pid {process.pid}) exited with code "
                f"{process.exitcode} after {stats.get('episodes', 'unknown')} episodes; "
                f"last trajectory {stats.get('last_trajectory_id')}"
                + (f"; reported error: {stats.get('error')}" if stats.get("error") else ""),
                details=details,
            )
            self._worker_error = error
            raise error

    def _worker_finished_quota(self, index: int) -> bool:
        """``True`` when a worker exited cleanly after its configured episodes."""
        quota = self.episodes_per_worker
        if quota is None:
            return False
        stats = self.writer.worker_stats.get(index, {})
        return int(stats.get("episodes", 0)) >= int(quota)

    # -- collection -----------------------------------------------------------

    def collect_trajectories(
        self, count: int, *, timeout_s: float = 600.0
    ) -> list[CollectedTrajectory]:
        """Wait for ``count`` newly collected trajectories (row via the writer)."""
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if count == 0:
            return []
        self.check_health()
        taken = self.writer.take(count, timeout_s=timeout_s)
        self.check_health()
        return taken

    def sync_weights(
        self, model: Any, *, version: int, model_version_string: str = ""
    ) -> dict[str, Any]:
        """Publish new weights to inference (atomic) and to local workers (S16-S17)."""
        report: dict[str, Any] = {"version": int(version), "workers_signalled": 0}
        server = self.server
        if server is not None:
            report["inference"] = server.sync_weights(
                model, version=version, model_version_string=model_version_string
            )
        else:
            state = {
                key: value.detach().cpu().numpy()
                for key, value in model.state_dict().items()
            }
            for control in self.control_queues.values():
                control.put(
                    {
                        "kind": "set_weights",
                        "state": state,
                        "version": int(version),
                        "model_version": str(model_version_string),
                    }
                )
                report["workers_signalled"] += 1
        self.model_version_string = str(model_version_string)
        self.inference_version = int(version)
        return report

    # -- diagnostics ----------------------------------------------------------

    def queue_size(self) -> int | None:
        try:
            return int(self.trajectory_queue.qsize())
        except (NotImplementedError, OSError):  # pragma: no cover - platform dependent
            return None

    def aggregate_stats(self) -> dict[str, Any]:
        """Sum the worker-reported counters (one snapshot per worker)."""
        stats = self.writer.worker_stats
        totals: dict[str, Any] = {
            "trajectories": 0,
            "env_steps": 0,
            "searches": 0,
            "recurrent_inference_calls": 0,
            "terminated": 0,
            "truncated": 0,
            "reward_sum": 0.0,
            "action_counts": [0] * 6,
            "blocked_seconds": 0.0,
            "dropped": 0,
            "payload_build_seconds": 0.0,
            "payload_bytes": 0,
            "workers_reporting": 0,
            "dataset_backends": {},
            # Authoritative production counts (worker snapshots can lag a round).
            "trajectories_written": self.writer.trajectories_written,
            "transitions_written": self.writer.transitions_written,
            "trajectories_written_by_worker": dict(
                sorted(self.writer.written_by_worker.items())
            ),
            "inference_round_trips": 0,
            "inference_round_trip_ms": 0.0,
            "inference": {},
            "profiling": {},
        }
        phase_reports: list[dict[str, Any]] = []
        local_calls = 0
        local_items = 0
        local_max_batch = 0
        local_latency = 0.0
        local_errors = 0
        for worker_id, snapshot in sorted(stats.items()):
            if snapshot.get("stopped") and "env_steps" not in snapshot:
                continue
            totals["workers_reporting"] += 1
            for key in (
                "trajectories",
                "env_steps",
                "searches",
                "recurrent_inference_calls",
                "terminated",
                "truncated",
            ):
                totals[key] += int(snapshot.get(key, 0))
            totals["reward_sum"] += float(snapshot.get("reward_sum", 0.0))
            counts = list(snapshot.get("action_counts", [0] * 6))
            if len(counts) == 6:
                totals["action_counts"] = [
                    int(a) + int(b) for a, b in zip(totals["action_counts"], counts, strict=True)
                ]
            totals["blocked_seconds"] += float(snapshot.get("blocked_seconds", 0.0))
            totals["dropped"] += int(snapshot.get("dropped", 0))
            totals["payload_build_seconds"] += float(snapshot.get("payload_build_seconds", 0.0))
            totals["payload_bytes"] += int(snapshot.get("payload_bytes", 0))
            backend = snapshot.get("dataset_backend")
            if backend:
                totals["dataset_backends"][backend] = (
                    int(totals["dataset_backends"].get(backend, 0)) + 1
                )
            inference = snapshot.get("inference") or {}
            totals["inference_round_trips"] += int(inference.get("round_trips", 0))
            totals["inference_round_trip_ms"] += float(
                inference.get("mean_round_trip_ms", 0.0)
            )
            if snapshot.get("profiling"):
                phase_reports.append(snapshot["profiling"])
            if inference.get("mode") == "local":
                local_calls += int(inference.get("calls", 0))
                local_items += int(inference.get("items", 0))
                local_max_batch = max(
                    local_max_batch, int(inference.get("max_batch_size_observed", 0))
                )
                local_latency += float(inference.get("mean_inference_latency_ms", 0.0))
                local_errors += int(inference.get("errors", 0))
            del worker_id
        if local_calls:
            # Local mode: every worker runs its own model copy, so call/item
            # counts sum and the mean batch size is the item/call ratio.
            totals["inference"] = {
                "mode": "local",
                "calls": local_calls,
                "items": local_items,
                "mean_batch_size": local_items / local_calls,
                "max_batch_size_observed": local_max_batch,
                "mean_inference_latency_ms": local_latency
                / max(totals["workers_reporting"], 1),
                "errors": local_errors,
            }
        if phase_reports:
            totals["profiling"] = merge_phase_reports(*phase_reports)
        elapsed = max(time.perf_counter() - self.start_time, 1e-9)
        totals["elapsed_seconds"] = elapsed
        totals["env_steps_per_sec"] = totals["env_steps"] / elapsed
        totals["searches_per_sec"] = totals["searches"] / elapsed
        totals["simulations_per_sec"] = (
            totals["searches"] * int(self.collector_config.num_simulations) / elapsed
        )
        return totals

    def refresh_worker_stats(self, *, timeout_s: float = 1.0) -> int:
        """Ask workers for fresh counters and wait for each to answer.

        The writer thread drains the stats queue continuously, so "a message
        arrived" is not observable here; workers therefore stamp every snapshot
        with a monotonic ``stats_seq`` and this waits for that counter to move.
        """
        self.writer.drain()
        baseline = {
            worker_id: int(snapshot.get("stats_seq", 0))
            for worker_id, snapshot in self.writer.worker_stats.items()
        }
        self.request_stats()
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            time.sleep(0.02)
            self.writer.drain()
            stats = self.writer.worker_stats
            if all(
                int(stats.get(worker_id, {}).get("stats_seq", 0)) > baseline.get(worker_id, 0)
                for worker_id in range(len(self._processes))
            ):
                break
        return len(self.writer.worker_stats)

    def diagnostics(self, *, fresh: bool = False) -> dict[str, Any]:
        if fresh:
            self.refresh_worker_stats()
        else:
            self.writer.drain()
        payload: dict[str, Any] = {
            "config": self.config.to_dict(),
            "workers": self.worker_status(),
            "alive_workers": self.alive_workers,
            "worker_pids": self.worker_pids,
            "trajectory_queue": {
                "size": self.queue_size(),
                "max_size": self.config.trajectory_queue_size,
            },
            "writer": self.writer.diagnostics(),
            "replay": self.replay.memory_report(),
            "aggregate": self.aggregate_stats(),
            "closed": self._closed,
        }
        if self.server is not None:
            payload["inference_server"] = self.server.diagnostics()
        return payload

    # -- lifecycle ------------------------------------------------------------

    def request_stats(self) -> None:
        """Ask every worker to publish a fresh stats snapshot."""
        for control in self.control_queues.values():
            control.put({"kind": "stats"})

    def set_temperature(self, temperature: float) -> None:
        """Publish the root-visit temperature the workers should search with.

        The trainer owns the schedule (it knows the global environment-step
        count); workers apply it to the search and to the metadata they record.
        """
        for control in self.control_queues.values():
            control.put({"kind": "set_temperature", "temperature": float(temperature)})

    def flush(self, *, timeout_s: float = 5.0) -> int:
        """Wait until the queue is drained into replay; return trajectories left."""
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            self.writer._drain_stats()
            size = self.queue_size()
            if size in (0, None):
                return 0
            time.sleep(0.01)
        return int(self.queue_size() or 0)

    def close(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Clean shutdown: stop collection, flush, stop inference, stop writer (S36)."""
        if self._closed:
            return self._shutdown_report
        timeout = self.config.worker_stop_timeout_s if timeout_s is None else timeout_s
        report: dict[str, Any] = {
            "requested_at": time.time(),
            "workers_signalled": 0,
            "workers_joined": 0,
            "workers_terminated": [],
            "queue_drained": True,
        }
        for control in self.control_queues.values():
            control.put({"kind": "stop"})
            report["workers_signalled"] += 1
        deadline = time.perf_counter() + timeout
        for index, process in enumerate(self._processes):
            remaining = max(0.0, deadline - time.perf_counter())
            process.join(timeout=remaining)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
                report["workers_terminated"].append(index)
            else:
                report["workers_joined"] += 1
        leftover = self.flush(timeout_s=10.0)
        report["queue_drained"] = leftover == 0
        report["queue_leftover"] = leftover
        if self.server is not None:
            self.server.stop()
            report["inference_server_stopped"] = not self.server.is_alive
        self.writer.stop()
        report["writer_stopped"] = not self.writer.is_alive
        report["alive_workers"] = self.alive_workers
        report["trajectories_written"] = self.writer.trajectories_written
        report["transitions_written"] = self.writer.transitions_written
        report["dropped"] = self.writer.dropped
        report["aggregate"] = self.aggregate_stats()
        if self.server is not None:
            report["inference_server"] = self.server.diagnostics()
        for queue_object in [
            self.request_queue,
            self.trajectory_queue,
            self.stats_queue,
            *self.response_queues.values(),
            *self.control_queues.values(),
        ]:
            # Best effort: a queue that refuses to close must not abort shutdown.
            with contextlib.suppress(Exception):
                queue_object.close()
        self._closed = True
        self._shutdown_report = report
        return report

    #: populated by :meth:`close`
    _shutdown_report: dict[str, Any] = {}

    def __enter__(self) -> MuZeroCollectorPool:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
