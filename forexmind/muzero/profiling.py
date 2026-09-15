"""Wall-time phase instrumentation for MuZero (Stage 4.6, brief S2).

The brief asks for a *measured* bottleneck profile before any optimisation, with
the phases reported separately rather than collapsed into one number::

    environment stepping        initial inference
    recurrent inference         MCTS tree logic
    trajectory serialization    replay insertion
    replay sampling             learner forward/backward
    validation                  checkpointing

Timing is opt-in (``PhaseTimer(enabled=False)`` is a no-op) so the measured
production path pays nothing for the instrumentation.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PhaseTimer", "merge_phase_reports", "phase"]


def phase(timer: PhaseTimer | None, name: str) -> Any:
    """``timer.phase(name)`` when profiling is enabled, a no-op otherwise."""
    if timer is None or not timer.enabled:
        return nullcontext()
    return timer.phase(name)


@dataclass(slots=True)
class PhaseTimer:
    """Accumulates wall time per named phase."""

    enabled: bool = False
    totals: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        if not self.enabled:
            return
        self.totals[name] = self.totals.get(name, 0.0) + float(seconds)
        self.calls[name] = self.calls.get(name, 0) + 1

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    @property
    def total_seconds(self) -> float:
        return float(sum(self.totals.values()))

    def reset(self) -> None:
        self.totals.clear()
        self.calls.clear()

    def report(self) -> dict[str, Any]:
        total = self.total_seconds
        phases = {
            name: {
                "seconds": seconds,
                "calls": int(self.calls.get(name, 0)),
                "fraction": (seconds / total if total > 0 else 0.0),
                "ms_per_call": (
                    1e3 * seconds / self.calls[name] if self.calls.get(name) else 0.0
                ),
            }
            for name, seconds in sorted(self.totals.items(), key=lambda item: -item[1])
        }
        return {
            "enabled": self.enabled,
            "total_seconds": total,
            "phases": phases,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.report()


def merge_phase_reports(*reports: dict[str, Any]) -> dict[str, Any]:
    """Merge several :meth:`PhaseTimer.report` payloads into one profile."""
    totals: dict[str, float] = {}
    calls: dict[str, int] = {}
    enabled = False
    for report in reports:
        if not report:
            continue
        enabled = enabled or bool(report.get("enabled"))
        for name, payload in (report.get("phases") or {}).items():
            totals[name] = totals.get(name, 0.0) + float(payload.get("seconds", 0.0))
            calls[name] = calls.get(name, 0) + int(payload.get("calls", 0))
    total = float(sum(totals.values()))
    return {
        "enabled": enabled,
        "total_seconds": total,
        "phases": {
            name: {
                "seconds": seconds,
                "calls": calls.get(name, 0),
                "fraction": (seconds / total if total > 0 else 0.0),
                "ms_per_call": (1e3 * seconds / calls[name] if calls.get(name) else 0.0),
            }
            for name, seconds in sorted(totals.items(), key=lambda item: -item[1])
        },
    }
