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
        (self.run_dir / "checkpoints" / "latest.txt").write_text(f"{name}.pt\n", encoding="utf-8")
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


def discover_checkpoints(run_root: str | Path) -> list[Path]:
    """Find every ``*.pt`` checkpoint under ``run_root`` (recursively)."""
    root = Path(run_root)
    if not root.is_dir():
        return []
    pts: list[Path] = []
    for p in sorted(root.rglob("*.pt")):
        if p.is_file():
            pts.append(p)
    pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pts


def resolve_checkpoint(
    checkpoint_arg: str | Path,
    run_root: str | Path | None = None,
) -> Path:
    """Resolve a ``--checkpoint`` argument to an existing ``.pt`` file.

    Accepts (in order of preference):
    * an exact path to a ``.pt`` file,
    * a directory → its ``best.pt``, ``latest.pt``, or newest ``*.pt``,
    * a run name (e.g. ``sac_cpu_seed42``) searched under ``run_root``,
    * a bare name like ``best.pt`` / ``latest.pt`` searched under ``run_root``.

    Raises ``FileNotFoundError`` with a list of the checkpoints actually found
    under ``run_root`` so a mistyped path is easy to fix.
    """
    given = Path(checkpoint_arg)

    if given.is_file():
        return given
    if given.is_dir():
        # Prefer the CheckpointManager layout: <dir>/checkpoints/{best,latest}.pt
        for cand in (
            given / "checkpoints" / "best.pt",
            given / "checkpoints" / "latest.pt",
            given / "best.pt",
            given / "latest.pt",
        ):
            if cand.is_file():
                return cand
        newest = discover_checkpoints(given)
        if newest:
            return newest[0]
        raise FileNotFoundError(f"no .pt checkpoint found inside directory {given}")

    root = Path(run_root) if run_root is not None else None
    all_found = discover_checkpoints(root) if root is not None else []

    # Bare checkpoint name (e.g. "best.pt" / "latest.pt") anywhere under root.
    if given.name in ("best.pt", "latest.pt") or given.suffix == ".pt":
        for c in all_found:
            if c.name == given.name:
                return c
        raise FileNotFoundError(_not_found_msg(given, root, all_found))

    # Run name → <root>/<given>/checkpoints/{best,latest}.pt
    if root is not None:
        run_dir = root / given
        if run_dir.is_dir():
            for name in ("best.pt", "latest.pt"):
                cand = run_dir / "checkpoints" / name
                if cand.is_file():
                    return cand
            in_run = discover_checkpoints(run_dir)
            if in_run:
                raise FileNotFoundError(
                    f"no best/latest checkpoint inside {run_dir}, "
                    f"but found: {', '.join(str(p) for p in in_run[:10])}"
                )

    raise FileNotFoundError(_not_found_msg(given, root, all_found))


def _not_found_msg(given: Path, root: Path | None, all_found: list[Path]) -> str:
    if all_found:
        listing = "\n  ".join(str(p) for p in all_found[:20])
        return (
            f"checkpoint {str(given)!r} not found under {root or 'cwd'}. "
            f"Existing checkpoints:\n  {listing}"
        )
    return (
        f"checkpoint {str(given)!r} not found and no .pt files under "
        f"{root or 'cwd'}. Train first "
        "(python -m forexmind.training.train_sac --config configs/sac_cpu.yaml), "
        "then point --checkpoint at the produced best.pt."
    )


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
