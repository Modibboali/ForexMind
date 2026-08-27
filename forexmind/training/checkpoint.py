"""Checkpoint management (Phase 3).

Each checkpoint stores actor/critic/target parameters, optimizers,
entropy-temperature state, training counters, config, dataset version, and RNG
metadata.  The replay buffer itself is NOT included (kept out to keep
checkpoints small — documented in README); only its metadata is stored, so a
resumed run starts with an empty buffer and refills it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.run_dir / "checkpoints" / f"{name}.pt"

    def save(self, name: str, state: dict[str, Any]) -> Path:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        # Mark "latest" pointer.
        (self.run_dir / "checkpoints" / "latest.txt").write_text(
            f"{name}.pt\n", encoding="utf-8"
        )
        return path

    def load(self, path: str | Path) -> dict[str, Any]:
        return torch.load(path, map_location="cpu", weights_only=False)

    def latest_path(self) -> Path | None:
        marker = self.run_dir / "checkpoints" / "latest.txt"
        if marker.is_file():
            p = self.run_dir / "checkpoints" / marker.read_text(encoding="utf-8").strip()
            if p.is_file():
                return p
        # Fallback: newest *.pt
        pts = sorted(self.run_dir.glob("checkpoints/*.pt"), key=lambda p: p.stat().st_mtime)
        return pts[-1] if pts else None

    def best_path(self) -> Path | None:
        p = self.run_dir / "checkpoints" / "best.pt"
        return p if p.is_file() else None


def build_checkpoint_state(
    *,
    algorithm: str,
    policy_state: dict[str, Any],
    critic_states: dict[str, dict[str, Any]] | None,
    target_states: dict[str, dict[str, Any]] | None,
    optimizers: dict[str, Any],
    log_alpha: Any,
    env_steps: int,
    gradient_updates: int,
    episodes: int,
    config: Any,
    dataset_version: str,
    rng_state: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a serializable checkpoint dict."""
    return {
        "algorithm": algorithm,
        "policy": policy_state,
        "critics": critic_states or {},
        "targets": target_states or {},
        "optimizers": optimizers,
        "log_alpha": log_alpha,
        "env_steps": env_steps,
        "gradient_updates": gradient_updates,
        "episodes": episodes,
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "dataset_version": dataset_version,
        "rng": rng_state,
    }
