"""Cumulative decision/execution diagnostics, isolated by worker and episode.

Durations are in M5 decision steps. A position lasts from entry until flat or
sign reversal; same-direction resizing does not restart its clock. Episode
ends and resume boundaries censor open positions. Summary includes censored
durations (and reports their count), including currently open positions.
Turnover is summed absolute account notional / equity before each decision.
Transition percentages use all policy decisions as their denominator.
"""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np

from forexmind.environment.actions import ACTION_NAMES

TRANSITIONS = (
    "FLAT_LONG",
    "FLAT_SHORT",
    "LONG_HOLD",
    "LONG_FLAT",
    "LONG_SHORT",
    "SHORT_HOLD",
    "SHORT_FLAT",
    "SHORT_LONG",
)


def _direction(units: float) -> str:
    return "LONG" if units > 0 else "SHORT" if units < 0 else "FLAT"


def _duration_stats(hist: Counter) -> tuple[float, float, int]:
    count = sum(hist.values())
    if not count:
        return 0.0, 0.0, 0
    ranks = ((count - 1) // 2, count // 2)
    med: list[int] = []
    seen = 0
    for length, n in sorted(hist.items()):
        med.extend(length for rank in ranks if seen <= rank < seen + n)
        seen += n
    return sum(k * v for k, v in hist.items()) / count, float(np.mean(med)), max(hist)


class ActionDiagnostics:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.transitions: Counter[str] = Counter()
        self.holds: Counter[int] = Counter()
        self.durations: Counter[int] = Counter()
        self.active: dict[int, dict] = {}
        self.totals: Counter[str] = Counter()

    def _finish(self, worker: int, censored: bool = True) -> None:
        state = self.active.pop(worker, None)
        if state is None:
            return
        if state["hold"]:
            self.holds[state["hold"]] += 1
        if state["duration"]:
            self.durations[state["duration"]] += 1
            self.totals["censored_positions"] += int(censored)

    def record(
        self,
        action: int,
        info: dict,
        *,
        worker: int = 0,
        episode: int = 0,
        step: int = 0,
        done: bool = False,
    ) -> None:
        if worker in self.active and (step == 0 or self.active[worker]["episode"] != episode):
            self._finish(worker)
        state = self.active.setdefault(
            worker, {"episode": episode, "hold": 0, "duration": 0, "direction": "FLAT"}
        )
        self.counts[ACTION_NAMES[action]] += 1
        before = _direction(info["units_before"])
        after = _direction(info["units_after_policy"])
        label = f"{before}_HOLD" if action == 0 else f"{before}_{after}"
        if label in TRANSITIONS:
            self.transitions[label] += 1
        for key in (
            "position_changes",
            "actual_executions",
            "sign_reversals",
            "turnover",
            "forced_executions",
            "forced_turnover",
        ):
            self.totals[key] += info.get(key, 0)
        self.totals["hold_executions"] += int(action == 0) * info["actual_executions"]
        if action == 0:
            state["hold"] += 1
        elif state["hold"]:
            self.holds[state["hold"]] += 1
            state["hold"] = 0
        if state["duration"] and state["direction"] != after:
            self.durations[state["duration"]] += 1
            state["duration"] = 0
        state["direction"] = after
        if after != "FLAT":
            state["duration"] += 1
        if done:
            self._finish(worker, censored=not bool(info.get("forced_executions")))

    def summary(self) -> dict[str, float]:
        n = sum(self.counts.values())
        result = {"policy_decisions": float(n)}
        for name in ACTION_NAMES:
            result[f"count_{name}"] = float(self.counts[name])
            result[f"pct_{name.lower()}"] = 100.0 * self.counts[name] / max(1, n)
        result["pct_long"] = sum(result[f"pct_{a.lower()}"] for a in ACTION_NAMES[6:])
        result["pct_short"] = sum(result[f"pct_{a.lower()}"] for a in ACTION_NAMES[2:6])
        for name in TRANSITIONS:
            result[f"transitions_{name}"] = float(self.transitions[name])
            result[f"pct_transition_{name}"] = 100.0 * self.transitions[name] / max(1, n)
        for key in (
            "position_changes",
            "actual_executions",
            "sign_reversals",
            "turnover",
            "forced_executions",
            "forced_turnover",
            "hold_executions",
        ):
            result[key] = float(self.totals[key])
        holds, durations = self.holds.copy(), self.durations.copy()
        for state in self.active.values():
            if state["hold"]:
                holds[state["hold"]] += 1
            if state["duration"]:
                durations[state["duration"]] += 1
        mean, median, maximum = _duration_stats(holds)
        result.update(
            mean_consecutive_hold=mean,
            median_consecutive_hold=median,
            max_hold_streak=float(maximum),
        )
        mean, median, _ = _duration_stats(durations)
        result.update(
            mean_position_holding_duration=mean,
            median_position_holding_duration=median,
            censored_positions=float(
                self.totals["censored_positions"]
                + sum(s["duration"] > 0 for s in self.active.values())
            ),
        )
        return result

    def state_dict(self) -> dict:
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state: dict) -> None:
        if state:
            self.__dict__.update(copy.deepcopy(state))
            # Existing collectors restart episodes on resume; do not join streaks.
            for worker in list(self.active):
                self._finish(worker)
