"""Regression tests for independent episode reporting and matched true-HOLD baselines."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from forexmind.config import default_config
from forexmind.episodes.config import EpisodeConfig
from forexmind.episodes.sampler import EpisodeSampler, EpisodeSpec
from forexmind.episodes.trajectory import Trajectory
from forexmind.evaluation.runner import EvaluationRunner
from forexmind.evaluation.sampled import (
    EnterAndHoldAgent,
    chronological_portfolio_metrics,
    episode_result,
    paired_comparison,
    sampled_report,
)
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig

from tests.test_evaluation import _runner_dataset


def episode(logs: list[float], *, seed: int = 1, turnover: float = 1.0) -> Trajectory:
    returns = np.asarray(logs)
    n = len(returns)
    infos = [
        dict(
            units_before=0.0 if i == 0 else 1.0,
            units_after_policy=1.0,
            actual_executions=int(i == 0),
            position_changes=int(i == 0),
            sign_reversals=0,
            turnover=turnover if i == 0 else 0.0,
        )
        for i in range(n)
    ]
    return Trajectory(
        "synthetic",
        EpisodeSpec("EURUSD", "validation", 10, 10 + n, n, 8, seed),
        np.datetime64("2020-01-06T00:00") + np.arange(n) * np.timedelta64(5, "m"),
        np.asarray([9, *([0] * (n - 1))]),
        returns.copy(),
        np.r_[1.0, np.exp(np.cumsum(returns))],
        returns.copy(),
        np.ones(n),
        trade_log=[dict(units_delta=1.0)],
        info=dict(
            action_indices=[9, *([0] * (n - 1))],
            action_diagnostics=infos,
            action_semantics="categorical_v1",
        ),
    )


def test_timestep_averaging_inflation_is_only_a_named_diagnostic() -> None:
    a = episode([0.0101, -0.0099, 0.0101, -0.0099])
    b = episode([-0.0098, 0.0102, -0.0097, 0.0103], seed=2)
    report = sampled_report([a, b], 75000)
    assert report["n_periods"] is None
    assert report["sharpe"] is None and report["sortino"] is None
    assert report["total_return"] is None
    assert report["total_environment_decisions"] == 8
    diagnostic = report["cross_episode_mean_return_series"]
    assert diagnostic["return_observation_count"] == 4
    assert diagnostic["is_portfolio"] is False
    assert diagnostic["diagnostic_sharpe"] > 10 * max(
        r["episode_sharpe"] for r in report["episodes"]
    )
    assert report["mean_episode_return"] == pytest.approx(
        np.mean([a.equity[-1] - 1, b.equity[-1] - 1])
    )


def test_episode_metrics_are_independent_and_rewards_are_not_averaged_first() -> None:
    a = episode([0.01, -0.02, 0.03])
    before = sampled_report([a, episode([0.0, 0.0, 0.0], seed=2)], 75000)
    after = sampled_report([a, episode([-0.2, 0.1, -0.3], seed=2)], 75000)
    assert before["episodes"][0] == after["episodes"][0]
    assert before["episodes"][0]["mean_step_reward"] == pytest.approx(0.02 / 3)


def test_total_and_mean_turnover_and_raw_accounting() -> None:
    report = sampled_report(
        [episode([0.01, -0.01], turnover=1), episode([0.02, -0.01], seed=2, turnover=3)], 75000
    )
    assert report["total_turnover_all_episodes"] == 4
    assert report["mean_turnover_per_episode"] == 2
    assert report["median_turnover_per_episode"] == 2
    assert report["min_turnover_per_episode"] == 1
    assert report["max_turnover_per_episode"] == 3
    assert "turnover" not in report and "turnover" not in report["actions"]
    assert all(report["invariants"].values())
    assert report["actions"]["censored_positions"] == 2


def test_known_m5_annualization_uses_simple_returns_and_one_factor() -> None:
    returns = np.array([0.001, -0.002, 0.0005, -0.0001])
    t = episode(np.log1p(returns).tolist())
    ppy = 12 * 24 * 252
    row = episode_result(t, ppy)
    assert row["periods_per_year"] == ppy
    assert row["episode_sharpe"] == pytest.approx(
        returns.mean() / returns.std(ddof=1) * math.sqrt(ppy)
    )
    assert row["annualized_volatility"] == pytest.approx(returns.std(ddof=1) * math.sqrt(ppy))
    assert row["annualized_return"] == pytest.approx(
        np.prod(1 + returns) ** (ppy / len(returns)) - 1
    )
    downside = math.sqrt(np.mean(returns[returns < 0] ** 2))
    assert row["episode_sortino"] == pytest.approx(returns.mean() / downside * math.sqrt(ppy))


def test_independent_or_overlapping_paths_cannot_become_a_portfolio() -> None:
    a = episode([0.01, -0.01])
    b = episode([0.02, 0.01], seed=2)
    with pytest.raises(ValueError, match="cannot join"):
        chronological_portfolio_metrics([a, b], 75000)
    assert chronological_portfolio_metrics([a], 75000)["n_periods"] == 2
    a.timestamps[:] = a.timestamps[0]
    with pytest.raises(ValueError, match="chronological"):
        chronological_portfolio_metrics([a], 75000)


def test_unequal_episode_lengths_still_have_independent_summaries() -> None:
    report = sampled_report([episode([0.01, -0.01]), episode([0.02], seed=2)], 75000)
    assert report["total_environment_decisions"] == 3
    assert report["cross_episode_mean_return_series"]["available"] is False
    assert report["sharpe"] is None


def test_paired_baselines_require_identical_specs_and_match_by_spec() -> None:
    a, b = episode([0.01, -0.01]), episode([0.02, 0.01], seed=2)
    ppo = sampled_report([a, b], 75000)
    baseline = sampled_report([b, a], 75000)
    assert paired_comparison(ppo, baseline)["paired_advantage"]["mean"] == 0
    baseline["episodes"][0]["start_index"] += 1
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(ppo, baseline)


@pytest.mark.parametrize("action", [0, 6, 7, 8, 9, 5, 4, 3, 2])
def test_enter_once_hold_units_and_terminal_mtm(action: int) -> None:
    dataset = _runner_dataset()
    config = default_config(
        initial_balance="10000", leverage=50, spread_value=0.0002, sizing_mode="equity_fraction"
    )
    encoder = ObservationEncoder(EncoderConfig(context_length=8, initial_balance="10000"))
    runner = EvaluationRunner(
        dataset, config, encoder, WindowConfig(context_length=8), capture_account_state=True
    )
    specs = EpisodeSampler(
        dataset, EpisodeConfig(split="test", horizon=10, context_length=8)
    ).sample(1, seed=42)
    t = runner.run_episode(EnterAndHoldAgent(action), specs[0])
    row = episode_result(t, runner.periods_per_year("test"))
    assert t.info["action_indices"] == [action, *([0] * 9)]
    assert np.all(t.position_units == t.position_units[0])
    assert row["actual_executions"] == int(action != 0)
    assert row["terminal_equity"] == pytest.approx(
        row["terminal_balance"] + row["terminal_unrealized_pnl"]
    )
    assert t.info["terminal_liquidation_requested"] is False
    assert all(row["invariants"].values())


def test_corrupt_raw_execution_counts_are_flagged() -> None:
    t = episode([0.01, -0.01])
    t.info["action_diagnostics"][1]["actual_executions"] = 1
    report = sampled_report([t], 75000)
    assert not report["invariants"]["hold_executions_zero"]
    assert not report["invariants"]["executions_match_trade_log"]


def test_missing_categorical_records_fail_instead_of_becoming_continuous() -> None:
    t = episode([0.01, -0.01])
    t.info["action_indices"] = []
    t.info["action_diagnostics"] = []
    with pytest.raises(ValueError, match="action count"):
        sampled_report([t], 75000)


def test_overlapping_market_samples_are_disclosed() -> None:
    report = sampled_report([episode([0.01, -0.01]), episode([0.02, 0.01], seed=2)], 100)
    assert report["overlapping_same_instrument_episode_pairs"] == [
        dict(instrument="EURUSD", seed_a=1, seed_b=2, overlapping_decision_intervals=2)
    ]
    assert report["sharpe"] is None


def test_short_path_annualization_overflow_is_explicitly_unavailable() -> None:
    row = episode_result(episode([0.01, 0.02]), 75000)
    assert row["annualized_return"] is None
    assert row["annualized_return_unavailable_reason"] == "Short-path annualization overflow"
    report = sampled_report([episode([0.01, 0.02])], 75000)
    assert (
        "annualized_return"
        in report["cross_episode_mean_return_series"]["undefined_diagnostic_metrics"]
    )
    json.dumps(report, allow_nan=False)


def test_training_selection_and_standalone_reporting_share_corrected_metrics() -> None:
    from forexmind.training.evaluator import PolicyEvaluator

    from tests.test_training_eval import _ds, _env_encoder, _policy

    env, encoder, window = _env_encoder()
    evaluator = PolicyEvaluator(_ds(), env, encoder, window, eval_horizon=16)

    report = evaluator.evaluate_sampled(_policy(), "sac", "validation", 2, seed=42)
    assert report["sharpe"] is None
    assert report["selection_metric_name"] == "mean_episode_log_return"
    assert report["_selection_score"] == report["mean_episode_log_return"]
    assert report["total_environment_decisions"] == 32
    assert report["actions"]["policy_decisions"] == 32
    assert not report["categorical_action_diagnostics_available"]
