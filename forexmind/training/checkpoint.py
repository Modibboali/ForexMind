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
        resolved = resolve_resume_checkpoint(path)
        return torch.load(resolved, map_location="cpu", weights_only=False)

    def latest_path(self) -> Path | None:
        marker = self.run_dir / "checkpoints" / "latest.txt"
        pointed = _checkpoint_from_marker(marker)
        if pointed is not None:
            return pointed
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


def _checkpoint_from_marker(marker: Path) -> Path | None:
    """Return the checkpoint named by a ``latest.txt`` marker, if valid."""
    if not marker.is_file():
        return None
    name = marker.read_text(encoding="utf-8").strip()
    if not name:
        return None
    pointed = Path(name)
    if not pointed.is_absolute():
        pointed = marker.parent / pointed
    return pointed if pointed.is_file() else None


def resolve_resume_checkpoint(checkpoint_arg: str | Path) -> Path:
    """Resolve an exact checkpoint, run directory, or latest-pointer alias.

    ``latest.pt`` is accepted as a compatibility alias for the actual
    ``latest.txt`` pointer written by :class:`CheckpointManager`.
    """
    given = Path(checkpoint_arg)
    if given.is_file():
        if given.name == "latest.txt":
            pointed = _checkpoint_from_marker(given)
            if pointed is None:
                raise FileNotFoundError(f"checkpoint marker {given} is empty or stale")
            return pointed
        return given

    if given.name == "latest.pt":
        pointed = _checkpoint_from_marker(given.with_name("latest.txt"))
        if pointed is not None:
            return pointed

    if given.is_dir():
        for marker in (given / "checkpoints" / "latest.txt", given / "latest.txt"):
            pointed = _checkpoint_from_marker(marker)
            if pointed is not None:
                return pointed
        newest = discover_checkpoints(given)
        if newest:
            return newest[0]
        raise FileNotFoundError(f"no resumable checkpoint found inside directory {given}")

    raise FileNotFoundError(
        f"resume checkpoint {given!s} not found; provide a .pt file, latest.txt, "
        "the documented latest.pt alias, or a run directory"
    )


def resolve_checkpoint(
    checkpoint_arg: str | Path,
    run_root: str | Path | None = None,
) -> Path:
    """Resolve a ``--checkpoint`` argument to an existing ``.pt`` file.

    Accepts (in order of preference):
    * an exact path to a ``.pt`` file,
    * a directory -> its ``best.pt``, ``latest.txt`` target, or newest ``*.pt``,
    * a run name (e.g. ``sac_cpu_seed42``) searched under ``run_root``,
    * a bare name like ``best.pt`` / ``latest.pt`` searched under ``run_root``.

    Raises ``FileNotFoundError`` with a list of the checkpoints actually found
    under ``run_root`` so a mistyped path is easy to fix.
    """
    given = Path(checkpoint_arg)

    if given.is_file():
        if given.name == "latest.txt":
            pointed = _checkpoint_from_marker(given)
            if pointed is None:
                raise FileNotFoundError(f"checkpoint marker {given} is empty or stale")
            return pointed
        return given
    if given.is_dir():
        # Prefer the CheckpointManager layout: <dir>/checkpoints/{best,latest}.pt
        for cand in (
            given / "checkpoints" / "best.pt",
            given / "best.pt",
        ):
            if cand.is_file():
                return cand
        for marker in (given / "checkpoints" / "latest.txt", given / "latest.txt"):
            pointed = _checkpoint_from_marker(marker)
            if pointed is not None:
                return pointed
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
        if given.name == "latest.pt":
            for marker in sorted(root.rglob("latest.txt")) if root is not None else ():
                pointed = _checkpoint_from_marker(marker)
                if pointed is not None:
                    return pointed
        raise FileNotFoundError(_not_found_msg(given, root, all_found))

    # Run name -> <root>/<given>/checkpoints/best.pt or latest.txt target.
    if root is not None:
        run_dir = root / given
        if run_dir.is_dir():
            cand = run_dir / "checkpoints" / "best.pt"
            if cand.is_file():
                return cand
            pointed = _checkpoint_from_marker(run_dir / "checkpoints" / "latest.txt")
            if pointed is not None:
                return pointed
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
    best_validation_score: float | None = None,
    best_checkpoint: str | None = None,
    validation_history: list[dict[str, Any]] | None = None,
    trainer_state: dict[str, Any] | None = None,
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
        "best_validation_score": best_validation_score,
        "best_checkpoint": best_checkpoint,
        "validation_history": validation_history or [],
        "trainer_state": trainer_state or {},
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "dataset_version": dataset_version,
        "rng": rng_state,
    }
