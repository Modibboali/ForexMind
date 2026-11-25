"""Equal-weighted instrument aggregation and per-period reporting (Phase 2).

Never report only an aggregate: every evaluation exposes each instrument
individually and then an aggregate.  Instruments are aggregated with **equal
weight** (not by row count), so EURUSD cannot dominate.

Alignment
---------
All episodes in a benchmark share the same horizon, so per-step log returns
are aligned by step index.  For each instrument, the per-step mean across its
episodes forms that instrument's return series; the aggregate is the
per-step mean across instruments.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from forexmind.episodes.trajectory import Trajectory
from forexmind.evaluation.metrics import compute_series_metrics


def mean_log_return_series(
    trajectories: list[Trajectory],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(timestamps, mean_log_returns, n_episodes)`` for one instrument."""
    if not trajectories:
        raise ValueError("mean_log_return_series requires >= 1 trajectory")
    timestamps = trajectories[0].timestamps
    stacked = np.stack([t.log_returns for t in trajectories])
    mean = np.mean(stacked, axis=0)
    return timestamps, mean, len(trajectories)


def aggregate_across_instruments(
    grouped: Mapping[str, list[Trajectory]],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Equal-weight aggregate return series across instruments.

    Returns ``(timestamps, aggregate_log_returns, per_instrument_series)``.
    """
    per_instrument: dict[str, np.ndarray] = {}
    timestamps: np.ndarray | None = None
    for instr, trajs in grouped.items():
        ts, mean, _ = mean_log_return_series(trajs)
        timestamps = ts
        per_instrument[instr] = mean
    if not per_instrument:
        raise ValueError("aggregate_across_instruments requires >= 1 instrument")
    assert timestamps is not None
    stacked = np.stack(list(per_instrument.values()))
    agg = np.mean(stacked, axis=0)
    return timestamps, agg, per_instrument


def per_period_log_returns(
    timestamps: np.ndarray, log_returns: np.ndarray, freq: str = "year"
) -> dict[str, np.ndarray]:
    """Group log returns by period label (default: calendar year)."""
    groups: dict[str, list[float]] = defaultdict(list)
    if freq == "year":
        for ts, r in zip(timestamps, log_returns, strict=True):
            year = str(np.datetime64(ts, "Y"))
            groups[year].append(float(r))
    else:
        raise ValueError(f"unsupported period frequency {freq!r}; use 'year'")
    return {k: np.asarray(v, dtype=np.float64) for k, v in sorted(groups.items(), key=str)}


def per_period_report(
    timestamps: np.ndarray,
    log_returns: np.ndarray,
    periods_per_year: float,
    freq: str = "year",
) -> dict[str, dict[str, object]]:
    """Per-period metrics (drawdown/returns are within-period, reset each period)."""
    out: dict[str, dict[str, object]] = {}
    for label, returns in per_period_log_returns(timestamps, log_returns, freq).items():
        if len(returns) == 0:
            continue
        metrics = compute_series_metrics(returns, periods_per_year)
        out[label] = metrics
    return out
