"""Independent sampled episodes are a distribution, never a portfolio path.

All episode summaries have equal episode weight. The legacy diagnostic instead
weights instruments equally, then episodes equally within each instrument.
Annualized episode statistics describe short individual paths, not a strategy
track record. No random-episode concatenation or calendar alignment is inferred.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, cast

import numpy as np

from forexmind.environment.actions import ACTION_NAMES, resolve_action
from forexmind.episodes.trajectory import Trajectory
from forexmind.evaluation.aggregation import aggregate_across_instruments
from forexmind.evaluation.metrics import compute_series_metrics
from forexmind.training.action_diagnostics import ActionDiagnostics


def distribution(values: list[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {
            "count": 0,
            **dict.fromkeys(("mean", "median", "std", "min", "max", "p10", "p25", "p75", "p90")),
        }
    if not np.isfinite(a).all():
        raise ValueError("Non-finite episode statistic")
    return {
        "count": len(a),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        **{f"p{p}": float(np.percentile(a, p)) for p in (10, 25, 75, 90)},
    }


def action_summary(trajectories: list[Trajectory]) -> dict[str, float]:
    diagnostic = ActionDiagnostics()
    for episode, traj in enumerate(trajectories):
        indices = cast(list[int], traj.info.get("action_indices", []))
        infos = cast(list[dict[str, Any]], traj.info.get("action_diagnostics", []))
        for step, (action, info) in enumerate(zip(indices, infos, strict=True)):
            diagnostic.record(
                action, info, episode=episode, step=step, done=step == len(indices) - 1
            )
    return diagnostic.summary()


def episode_result(traj: Trajectory, periods_per_year: float) -> dict[str, Any]:
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("Annualization must be finite and positive")
    if traj.n_steps == 0 or len(traj.equity) != traj.n_steps + 1:
        raise ValueError("Episode needs one initial equity and one equity per decision")
    if not np.isfinite(traj.equity).all() or np.any(traj.equity <= 0):
        raise ValueError("Episode has non-positive or non-finite equity")
    if not np.allclose(
        np.log(traj.equity[1:] / traj.equity[:-1]), traj.log_returns, rtol=1e-9, atol=1e-12
    ):
        raise ValueError("Episode equity and log returns disagree")
    with np.errstate(over="ignore"):
        metrics: dict[str, Any] = compute_series_metrics(traj.log_returns, periods_per_year)
    diag = action_summary([traj])
    indices = cast(list[int], traj.info.get("action_indices", []))
    infos = cast(list[dict[str, Any]], traj.info.get("action_diagnostics", []))
    categorical = len(indices) == traj.n_steps
    if (indices or traj.info.get("action_semantics") == "categorical_v1") and not categorical:
        raise ValueError("Categorical action count does not match episode steps")
    changes = sum(float(i["units_after_policy"]) != float(i["units_before"]) for i in infos)
    holds_with_changes = sum(
        a == 0 and float(i["units_after_policy"]) != float(i["units_before"])
        for a, i in zip(indices, infos, strict=True)
    )
    reversals = sum(float(i["units_before"]) * float(i["units_after_policy"]) < 0 for i in infos)
    invariants = (
        {
            "decisions_match_steps": len(indices) == len(infos) == traj.n_steps,
            "hold_executions_zero": diag["hold_executions"] == 0 and holds_with_changes == 0,
            "executions_le_decisions": diag["actual_executions"] <= traj.n_steps,
            "position_changes_match_raw_units": diag["position_changes"] == changes,
            "executions_match_position_changes": diag["actual_executions"] == changes,
            "executions_match_trade_log": diag["actual_executions"] == len(traj.trade_log),
            "sign_reversals_match_raw_units": diag["sign_reversals"] == reversals,
            "reversals_le_executions": reversals <= diag["actual_executions"],
        }
        if categorical
        else {}
    )
    first_action = next((i for i, a in enumerate(indices) if a != 0), None)
    first_position = next((i for i, units in enumerate(traj.position_units) if units != 0), None)
    history = cast(list[dict[str, float]], traj.info.get("account_history", []))
    terminal = history[-1] if history else {}
    entry = history[first_position] if history and first_position is not None else {}
    return {
        **traj.spec.to_dict(),
        "policy": traj.agent_name,
        "n_steps": traj.n_steps,
        "initial_timestamp": traj.info.get("initial_timestamp"),
        "final_timestamp": str(traj.timestamps[-1]),
        "total_return": float(traj.equity[-1] / traj.equity[0] - 1),
        "cumulative_log_return": float(traj.log_returns.sum()),
        "mean_step_reward": float(traj.rewards.mean()),
        "step_volatility": metrics["per_period_volatility"],
        "max_drawdown": metrics["max_drawdown_pct"],
        "episode_sharpe": metrics["sharpe"],
        "episode_sortino": metrics["sortino"],
        "annualized_return": (
            metrics["annualized_return"] if np.isfinite(metrics["annualized_return"]) else None
        ),
        "annualized_return_unavailable_reason": (
            None
            if np.isfinite(metrics["annualized_return"])
            else "Short-path annualization overflow"
        ),
        "annualized_volatility": metrics["annualized_volatility"],
        "periods_per_year": periods_per_year,
        "turnover": (
            diag["turnover"]
            if categorical
            else cast(dict[str, Any], traj.metrics["trading"])["turnover"]
        ),
        "actual_executions": diag["actual_executions"] if categorical else len(traj.trade_log),
        "position_changes": changes if categorical else len(traj.trade_log),
        "sign_reversals": reversals if categorical else None,
        "first_non_hold_action": ACTION_NAMES[indices[first_action]]
        if first_action is not None
        else None,
        "first_non_hold_decision_index": first_action,
        "time_to_first_position_steps": first_position,
        "initial_exposure": entry.get("exposure_fraction", 0.0 if first_position is None else None),
        "entry_direction": (
            "FLAT"
            if first_position is None
            else "LONG"
            if traj.position_units[first_position] > 0
            else "SHORT"
        ),
        "final_exposure": terminal.get("exposure_fraction"),
        "final_position_units": float(traj.position_units[-1]),
        "terminal_unrealized_pnl": terminal.get("unrealized_pnl"),
        "terminal_balance": terminal.get("balance"),
        "terminal_equity": float(traj.equity[-1]),
        "mean_position_holding_duration": (
            diag["mean_position_holding_duration"] if categorical else None
        ),
        "median_position_holding_duration": (
            diag["median_position_holding_duration"] if categorical else None
        ),
        "censored_positions": diag["censored_positions"] if categorical else None,
        "action_diagnostics": diag,
        "invariants": invariants,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "total_return",
        "cumulative_log_return",
        "mean_step_reward",
        "step_volatility",
        "max_drawdown",
        "turnover",
        "actual_executions",
        "position_changes",
        "sign_reversals",
        "time_to_first_position_steps",
        "initial_exposure",
        "final_exposure",
        "terminal_unrealized_pnl",
        "terminal_equity",
        "mean_position_holding_duration",
        "episode_sharpe",
        "episode_sortino",
    )
    stats = {key: distribution([r[key] for r in rows if r[key] is not None]) for key in keys}
    turnover = stats["turnover"]
    return {
        "episode_count": len(rows),
        "episode_statistics": stats,
        "mean_episode_return": stats["total_return"]["mean"],
        "mean_episode_log_return": stats["cumulative_log_return"]["mean"],
        "median_episode_return": stats["total_return"]["median"],
        "episode_return_std": stats["total_return"]["std"],
        "p10_episode_return": stats["total_return"]["p10"],
        "p25_episode_return": stats["total_return"]["p25"],
        "p75_episode_return": stats["total_return"]["p75"],
        "p90_episode_return": stats["total_return"]["p90"],
        "worst_episode_return": stats["total_return"]["min"],
        "best_episode_return": stats["total_return"]["max"],
        "profitable_episode_fraction": sum(r["total_return"] > 0 for r in rows) / len(rows),
        "total_turnover_all_episodes": sum(r["turnover"] for r in rows),
        **{f"{k}_turnover_per_episode": turnover[k] for k in ("mean", "median", "min", "max")},
        "mean_executions_per_episode": stats["actual_executions"]["mean"],
        "median_executions_per_episode": stats["actual_executions"]["median"],
        "mean_time_to_first_entry": stats["time_to_first_position_steps"]["mean"],
        "median_time_to_first_entry": stats["time_to_first_position_steps"]["median"],
        "episodes_entering_long": sum(r["entry_direction"] == "LONG" for r in rows),
        "episodes_entering_short": sum(r["entry_direction"] == "SHORT" for r in rows),
        "episodes_remaining_flat": sum(r["entry_direction"] == "FLAT" for r in rows),
        "final_long_episodes": sum(r["final_position_units"] > 0 for r in rows),
        "final_short_episodes": sum(r["final_position_units"] < 0 for r in rows),
        "final_flat_episodes": sum(r["final_position_units"] == 0 for r in rows),
    }


def sampled_report(trajectories: list[Trajectory], periods_per_year: float) -> dict[str, Any]:
    if not trajectories or not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("Sampled evaluation requires episodes and positive annualization")
    rows = [episode_result(t, periods_per_year) for t in trajectories]
    grouped: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        grouped[t.spec.instrument].append(t)
    diagnostic: dict[str, Any] = {"available": False, "reason": "Unequal episode lengths"}
    if len({t.n_steps for t in trajectories}) == 1:
        _, logs, _ = aggregate_across_instruments(grouped)
        with np.errstate(over="ignore"):
            legacy = compute_series_metrics(logs, periods_per_year)
        undefined = [k for k, v in legacy.items() if isinstance(v, float) and not np.isfinite(v)]
        legacy.update(dict.fromkeys(undefined))
        curve = np.r_[1.0, np.exp(np.cumsum(logs))]
        returns = curve[1:] / curve[:-1] - 1.0
        diagnostic = {
            "available": True,
            "is_portfolio": False,
            "diagnostic_only": True,
            "undefined_diagnostic_metrics": undefined,
            "weighting": (
                "equal instruments, then equal episodes within instrument; relative step index"
            ),
            "log_returns": logs.tolist(),
            "simple_returns": returns.tolist(),
            "return_observation_count": len(returns),
            "mean_return": float(returns.mean()),
            "return_std_ddof1": float(returns.std(ddof=1)),
            "sqrt_periods_per_year": float(np.sqrt(periods_per_year)),
            **{f"diagnostic_{k}": v for k, v in legacy.items()},
        }
    actions = action_summary(trajectories)
    actions.pop("turnover")  # The unqualified name was the ambiguous legacy sum.
    categorical = all(
        len(cast(list, t.info.get("action_indices", []))) == t.n_steps for t in trajectories
    )
    if not categorical:
        actions = {"policy_decisions": float(sum(t.n_steps for t in trajectories))}
    overlaps: list[dict[str, Any]] = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if a["instrument"] != b["instrument"]:
                continue
            # Absolute instrument M5 indices are comparable without inferred timestamps.
            common_steps = min(a["end_index"], b["end_index"]) - max(
                a["start_index"], b["start_index"]
            )
            if common_steps > 0:
                overlaps.append(
                    {
                        "instrument": a["instrument"],
                        "seed_a": a["seed"],
                        "seed_b": b["seed"],
                        "overlapping_decision_intervals": common_steps,
                    }
                )
    possible_pairs = sum(
        len(group) * (len(group) - 1) // 2 for group in grouped.values()
    )
    overlap_summary: dict[str, Any] = {
        "overlapping_episode_pairs": len(overlaps),
        "possible_same_instrument_pairs": possible_pairs,
        "overlap_fraction": len(overlaps) / possible_pairs if possible_pairs else 0.0,
        "largest_overlap_duration_steps": max(
            (item["overlapping_decision_intervals"] for item in overlaps), default=0
        ),
        "per_instrument": {},
    }
    for instrument, group in grouped.items():
        instrument_overlaps = [item for item in overlaps if item["instrument"] == instrument]
        instrument_pairs = len(group) * (len(group) - 1) // 2
        overlap_summary["per_instrument"][instrument] = {
            "episode_count": len(group),
            "overlapping_episode_pairs": len(instrument_overlaps),
            "possible_episode_pairs": instrument_pairs,
            "overlap_fraction": (
                len(instrument_overlaps) / instrument_pairs if instrument_pairs else 0.0
            ),
            "largest_overlap_duration_steps": max(
                (
                    item["overlapping_decision_intervals"]
                    for item in instrument_overlaps
                ),
                default=0,
            ),
        }
    return {
        "evaluation_type": "sampled_independent_episodes",
        "evaluation_mode": "independent_sampled_episodes",
        "is_portfolio_path": False,
        "portfolio_sharpe": None,
        "portfolio_sortino": None,
        "portfolio_calmar": None,
        "independence_definition": (
            "Independent account resets; sampled market windows may overlap "
            "and be statistically correlated"
        ),
        "overlapping_same_instrument_episode_pairs": overlaps,
        "overlap_diagnostics": overlap_summary,
        "categorical_action_diagnostics_available": categorical,
        "portfolio_metrics_unavailable_reason": (
            "Fresh accounts, random starts and instruments; no chronological capital path"
        ),
        "n_periods": None,
        "total_return": None,
        "sharpe": None,
        "sortino": None,
        "total_environment_decisions": sum(t.n_steps for t in trajectories),
        "raw_environment_reward_count": sum(len(t.rewards) for t in trajectories),
        "pooled_step_simple_return_std_diagnostic": (
            float(np.std(np.concatenate([t.simple_returns for t in trajectories]), ddof=1))
            if sum(t.n_steps for t in trajectories) > 1
            else 0.0
        ),
        "periods_per_year": periods_per_year,
        "annualization_assumption": (
            "mean observed M5 bars per instrument / split calendar years (365.25 days); "
            "observed closures/gaps included"
        ),
        "sortino_convention": (
            "zero target; RMS of negative simple returns only "
            "(legacy conditional downside convention)"
        ),
        "turnover_definition": (
            "sum over decisions of abs(executed units) * execution mid * quote-to-account "
            "factor / pre-decision equity; one-way policy turnover; forced turnover separate"
        )
        if categorical
        else "sum absolute executed account notional / initial account balance per episode",
        "duration_definition": (
            "M5 decision steps; entry delay is zero-based decisions before entry; "
            "same-direction resizing retains holding clock; open positions right-censored"
        ),
        "terminal_equity_includes_unrealized_pnl": True,
        "open_terminal_positions_have_no_hypothetical_closing_costs": True,
        "terminal_liquidation_requested": any(
            t.info.get("terminal_liquidation_requested", False) for t in trajectories
        ),
        **summarize_rows(rows),
        "actions": actions,
        "invariants": {
            k: all(r["invariants"].get(k, False) for r in rows) for k in rows[0]["invariants"]
        },
        "episodes": rows,
        "per_instrument": {
            instr: summarize_rows([r for r in rows if r["instrument"] == instr])
            for instr in grouped
        },
        "cross_episode_mean_return_series": diagnostic,
    }


def chronological_portfolio_metrics(
    trajectories: list[Trajectory], periods_per_year: float
) -> dict[str, Any]:
    """Accept one continuous account trajectory only; never infer joins on resets.

    A future multi-instrument portfolio simulator must supply its actual account
    path as one trajectory. Sorted dates alone cannot establish capital continuity.
    """
    if len(trajectories) != 1:
        raise ValueError(
            "Portfolio metrics require a single continuous account trajectory; "
            "cannot join independent episodes"
        )
    traj = trajectories[0]
    episode_result(traj, periods_per_year)
    if np.isnat(traj.timestamps).any() or np.any(
        np.diff(traj.timestamps) <= np.timedelta64(0, "ns")
    ):
        raise ValueError("Portfolio timestamps must be chronological and non-overlapping")
    return compute_series_metrics(traj.log_returns, periods_per_year)


def paired_comparison(ppo: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            row[k]
            for k in (
                "instrument",
                "split",
                "start_index",
                "end_index",
                "horizon",
                "context_length",
                "seed",
            )
        )

    left = {key(r): r for r in ppo["episodes"]}
    right = {key(r): r for r in baseline["episodes"]}
    if (
        left.keys() != right.keys()
        or len(left) != len(ppo["episodes"])
        or len(right) != len(baseline["episodes"])
    ):
        raise ValueError("Paired comparison requires identical unique episode specifications")
    pairs = [
        {
            "instrument": r["instrument"],
            "seed": r["seed"],
            "start_index": r["start_index"],
            "ppo_return": r["total_return"],
            "baseline_return": right[k]["total_return"],
            "advantage": r["total_return"] - right[k]["total_return"],
        }
        for k, r in left.items()
    ]
    differences = [r["advantage"] for r in pairs]
    return {
        "paired_advantage": distribution(differences),
        "fraction_ppo_beats_baseline": sum(v > 0 for v in differences) / len(differences),
        "pairs": pairs,
    }


class EnterAndHoldAgent:
    """Enter once at the first decision, then preserve exact units. FLAT stays flat."""

    def __init__(self, action_index: int) -> None:
        if action_index not in (0, *range(2, 10)):
            raise ValueError("Use HOLD (0) for the always-flat baseline or an exposure action")
        self.entry_action = action_index
        self.name = "FLAT" if action_index == 0 else ACTION_NAMES[action_index]
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self._entered = False
        self.last_action_index = 0

    def act(self, observation: Any) -> Any:
        self.last_action_index = 0 if self._entered else self.entry_action
        self._entered = True
        return resolve_action(self.last_action_index)
