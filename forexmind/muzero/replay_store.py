"""Replay persistence for the integrated MuZero loop (Stage 4.5).

Design (brief S30): the replay is stored as **one compressed NumPy shard per
trajectory** plus a JSON index, never inside the model checkpoint.

    <directory>/<name>_index.json          # format, version, config, metadata
    <directory>/<name>_shard_00000.npz     # one trajectory's arrays
    ...

Resume semantics, stated exactly:

* shards are written and read in **insertion order**, so FIFO order is
  preserved and therefore so is which trajectories would be evicted next;
* a trajectory that was already evicted from the in-memory buffer is not
  resurrected, because only live trajectories are written;
* loading reconstructs a :class:`~forexmind.muzero.replay.TrajectoryReplayBuffer`
  with the saved capacity configuration and re-validates every trajectory, so a
  corrupted shard fails loudly instead of silently entering training;
* if a caller loads with a *smaller* capacity than the file was written with,
  load reports the number of trajectories dropped by FIFO eviction instead of
  pretending the replay was restored exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forexmind.muzero.replay import ReplayConfig, TrajectoryReplayBuffer
from forexmind.muzero.trajectory import MuZeroTrajectory, TrajectoryMetadata

__all__ = [
    "REPLAY_FORMAT",
    "REPLAY_VERSION",
    "ReplayStoreReport",
    "load_replay",
    "replay_store_report",
    "save_replay",
]

REPLAY_FORMAT = "forexmind.muzero.replay"
REPLAY_VERSION = 1

_ARRAYS = (
    "observations",
    "actions",
    "rewards",
    "root_policies",
    "root_values",
    "action_masks",
    "terminated",
    "truncated",
    "planning_exposure",
    "planning_is_flat",
)


@dataclass(frozen=True, slots=True)
class ReplayStoreReport:
    """What a save or load actually did (capacity drops are never silent)."""

    directory: str
    name: str
    trajectories_written: int = 0
    trajectories_loaded: int = 0
    trajectories_dropped_on_load: int = 0
    transitions_loaded: int = 0
    bytes_on_disk: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "name": self.name,
            "trajectories_written": self.trajectories_written,
            "trajectories_loaded": self.trajectories_loaded,
            "trajectories_dropped_on_load": self.trajectories_dropped_on_load,
            "transitions_loaded": self.transitions_loaded,
            "bytes_on_disk": self.bytes_on_disk,
        }


def _index_path(directory: Path, name: str) -> Path:
    return directory / f"{name}_index.json"


def _shard_path(directory: Path, name: str, position: int) -> Path:
    return directory / f"{name}_shard_{position:05d}.npz"


def save_replay(
    replay: TrajectoryReplayBuffer,
    directory: str | Path,
    *,
    name: str = "replay",
    extra: dict[str, Any] | None = None,
) -> ReplayStoreReport:
    """Write every live trajectory to its own shard plus a JSON index."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    trajectories = replay.trajectories
    bytes_written = 0
    for position, trajectory in enumerate(trajectories):
        trajectory.validate()
        path = _shard_path(target, name, position)
        payload: dict[str, Any] = {
            field_name: np.asarray(getattr(trajectory, field_name)) for field_name in _ARRAYS
        }
        payload["boundary_value"] = np.asarray([trajectory.boundary_value], dtype=np.float64)
        # A unicode (not object) array so shards load with allow_pickle=False.
        payload["metadata_json"] = np.asarray(
            [json.dumps(trajectory.metadata.to_dict(), sort_keys=True)], dtype=np.str_
        )
        np.savez_compressed(path, **payload)
        bytes_written += path.stat().st_size

    index = {
        "format": REPLAY_FORMAT,
        "version": REPLAY_VERSION,
        "name": name,
        "config": replay.config.to_dict(),
        "num_trajectories": len(trajectories),
        "num_transitions": replay.num_transitions,
        "memory_report": replay.memory_report(),
        "trajectory_ids": [int(t.metadata.trajectory_id) for t in trajectories],
        "network_versions": [int(t.metadata.network_version) for t in trajectories],
        "extra": dict(extra or {}),
    }
    index_path = _index_path(target, name)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    bytes_written += index_path.stat().st_size
    return ReplayStoreReport(
        directory=str(target),
        name=name,
        trajectories_written=len(trajectories),
        bytes_on_disk=bytes_written,
    )


def load_replay(
    directory: str | Path,
    *,
    name: str = "replay",
    config: ReplayConfig | None = None,
    require_split: str | None = "train",
    strict: bool = True,
) -> tuple[TrajectoryReplayBuffer, ReplayStoreReport]:
    """Rebuild a replay buffer from shards, preserving insertion order."""
    source = Path(directory)
    index_path = _index_path(source, name)
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} does not exist; not a MuZero replay store")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("format") != REPLAY_FORMAT:
        raise ValueError(f"{index_path} is not a ForexMind MuZero replay store")
    if strict and int(index.get("version", 0)) != REPLAY_VERSION:
        raise ValueError(
            f"replay store version {index.get('version')} != supported {REPLAY_VERSION}"
        )

    stored = index.get("config", {})
    resolved = config or ReplayConfig(
        max_trajectories=int(stored.get("max_trajectories", 256)),
        max_transitions=stored.get("max_transitions"),
        sampling=str(stored.get("sampling", "uniform")),
        decision_rich_weight=float(stored.get("decision_rich_weight", 0.0)),
        seed=int(stored.get("seed", 0)),
    )
    buffer = TrajectoryReplayBuffer(resolved)
    expected = int(index.get("num_trajectories", 0))
    transitions = 0
    bytes_on_disk = index_path.stat().st_size
    for position in range(expected):
        path = _shard_path(source, name, position)
        if not path.exists():
            raise FileNotFoundError(f"replay shard {path} is missing (index expects {expected})")
        with np.load(path, allow_pickle=False) as shard:
            metadata = TrajectoryMetadata(**json.loads(str(shard["metadata_json"][0])))
            trajectory = MuZeroTrajectory(
                **{field_name: shard[field_name] for field_name in _ARRAYS},
                boundary_value=float(shard["boundary_value"][0]),
                metadata=metadata,
            )
        trajectory.validate()
        buffer.add(trajectory, require_split=require_split)
        transitions += len(trajectory)
        bytes_on_disk += path.stat().st_size

    dropped = max(0, expected - len(buffer))
    return buffer, ReplayStoreReport(
        directory=str(source),
        name=name,
        trajectories_loaded=len(buffer),
        trajectories_dropped_on_load=dropped,
        transitions_loaded=int(buffer.num_transitions if dropped == 0 else transitions),
        bytes_on_disk=bytes_on_disk,
    )


def replay_store_report(directory: str | Path, *, name: str = "replay") -> dict[str, Any]:
    """Read-only summary of a stored replay without loading the shards."""
    source = Path(directory)
    index_path = _index_path(source, name)
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} does not exist; not a MuZero replay store")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(source.glob(f"{name}_shard_*.npz"))
    return {
        "format": index.get("format"),
        "version": index.get("version"),
        "num_trajectories": index.get("num_trajectories"),
        "num_transitions": index.get("num_transitions"),
        "shards_present": len(shards),
        "bytes_on_disk": int(sum(path.stat().st_size for path in shards))
        + index_path.stat().st_size,
        "network_versions": index.get("network_versions", []),
        "config": index.get("config", {}),
    }
