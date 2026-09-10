"""Stage 3.5 checkpoint-selection and chronological-validation regressions."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from forexmind.evaluation.chronological import (
    ChronologicalEvaluator,
    chronological_instrument_report,
    chronological_spec,
)
from forexmind.training.config import ModelConfig
from forexmind.training.evaluator import selection_score
from forexmind.training.networks import CategoricalPolicy
from tools.evaluate_checkpoint_curve import rank_checkpoint_rows


def test_checkpoint_ranking_uses_episode_outcomes_not_diagnostic_sharpe() -> None:
    rows = [
        {
            "checkpoint": "A.pt",
            "mean_episode_log_return": 0.02,
            "diagnostic_sharpe": 1.0,
        },
        {
            "checkpoint": "B.pt",
            "mean_episode_log_return": 0.01,
            "diagnostic_sharpe": 100.0,
        },
    ]
    ranked = rank_checkpoint_rows(rows)
    assert ranked[0]["checkpoint"] == "A.pt"
    assert selection_score(rows[0]) == pytest.approx(0.02)
    with pytest.raises(ValueError, match="finite"):
        rank_checkpoint_rows([{"mean_episode_log_return": float("nan")}])


def test_cached_static_baselines_recompute_each_checkpoints_paired_advantage() -> None:
    from forexmind.evaluation.sampled import sampled_report
    from forexmind.training.evaluator import PolicyEvaluation, PolicyEvaluator

    from tests.test_sampled_evaluation import episode
    from tests.test_training_eval import _ds, _env_encoder

    env, encoder, window = _env_encoder()
    evaluator = PolicyEvaluator(_ds(), env, encoder, window, eval_horizon=2)
    baseline = sampled_report([episode([0.0, 0.0])], 75_000)
    winning = sampled_report([episode([0.01, 0.01])], 75_000)
    losing = sampled_report([episode([-0.01, -0.01])], 75_000)
    cached = {"FLAT": baseline}

    first = evaluator.matched_baseline_report(
        PolicyEvaluation("validation", metrics=winning),
        [episode([0.0, 0.0]).spec],
        baseline_reports=cached,
    )
    second = evaluator.matched_baseline_report(
        PolicyEvaluation("validation", metrics=losing),
        [episode([0.0, 0.0]).spec],
        baseline_reports=cached,
    )
    assert first["paired"]["FLAT"]["paired_advantage"]["mean"] > 0
    assert second["paired"]["FLAT"]["paired_advantage"]["mean"] < 0


def test_fixed_validation_specs_are_deterministic_balanced_and_disjoint(tmp_path) -> None:
    from forexmind.training.evaluator import PolicyEvaluator

    from tests.synthetic import make_instrument, make_split_dataset, timeline_m5
    from tests.test_training_eval import _env_encoder

    dates = [
        "2020-01-06",
        "2020-06-01",
        "2020-12-07",
        "2021-03-01",
        "2021-06-01",
        "2021-09-01",
        "2022-03-01",
        "2022-09-01",
    ]
    bars = timeline_m5(dates, per_day=80)
    dataset = make_split_dataset(
        {
            "EURUSD": make_instrument("EURUSD", bars),
            "GBPUSD": make_instrument("GBPUSD", bars),
        }
    )
    env, encoder, window = _env_encoder()
    evaluator = PolicyEvaluator(dataset, env, encoder, window, eval_horizon=16)
    first = evaluator.selection_episode_specs("validation", 6, 42)
    second = evaluator.selection_episode_specs("validation", 6, 42)
    assert [spec.to_dict() for spec in first] == [spec.to_dict() for spec in second]
    counts = {
        instrument: sum(s.instrument == instrument for s in first)
        for instrument in dataset.instruments
    }
    assert max(counts.values()) - min(counts.values()) <= 1
    for i, left in enumerate(first):
        for right in first[i + 1 :]:
            if left.instrument != right.instrument:
                continue
            left_begin = left.start_index - left.context_length + 1
            right_begin = right.start_index - right.context_length + 1
            assert left.end_index < right_begin or right.end_index < left_begin


def test_legacy_checkpoint_score_is_not_reinterpreted(tmp_path) -> None:
    from tests.test_ppo_parallel_correctness import _trainer

    source = _trainer(tmp_path / "source")
    source._save_checkpoint("new")
    path = tmp_path / "source" / "checkpoints" / "new.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "selection_schema",
        "selection_metric_name",
        "validation_selection_score",
        "best_selection_score",
        "best_checkpoint_step",
        "validation_episode_specs",
        "validation_summary",
        "legacy_selection_score",
    ):
        state.pop(key, None)
    state["best_validation_score"] = 12.5
    legacy_path = tmp_path / "legacy.pt"
    torch.save(state, legacy_path)

    restored = _trainer(tmp_path / "restored")
    restored._restore_from_checkpoint(legacy_path)
    assert restored.legacy_selection_score == pytest.approx(12.5)
    assert restored.best_score == -math.inf
    assert restored.best_checkpoint is None
    assert restored.best_checkpoint_step is None
    assert restored.validation_history == []


def test_new_checkpoint_records_and_restores_selection_metadata(tmp_path) -> None:
    from tests.test_ppo_parallel_correctness import _trainer

    source = _trainer(tmp_path / "source")
    source.best_score = 0.012
    source.best_checkpoint = "best"
    source.best_checkpoint_step = 123
    source._env_steps = 150
    source._gradient_updates = 7
    source.latest_validation_summary = {
        "validation_selection_score": 0.01,
        "mean_episode_return": 0.011,
        "median_episode_return": 0.009,
        "profitable_episode_fraction": 0.6,
    }
    source._save_checkpoint("metadata")
    path = tmp_path / "source" / "checkpoints" / "metadata.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["selection_schema"] == "sampled_episode_v1"
    assert state["selection_metric_name"] == "mean_episode_log_return"
    assert state["validation_selection_score"] == pytest.approx(0.01)
    assert state["best_selection_score"] == pytest.approx(0.012)
    assert state["best_checkpoint_step"] == 123
    assert state["best_validation_score"] is None
    assert state["validation_seed"] == 42
    assert state["validation_episode_count"] == 2
    assert len(state["validation_episode_specs"]) == 2

    restored = _trainer(tmp_path / "restored")
    restored._restore_from_checkpoint(path)
    assert restored.best_score == pytest.approx(0.012)
    assert restored.best_checkpoint_step == 123
    assert [s.to_dict() for s in restored.validation_episode_specs] == state[
        "validation_episode_specs"
    ]


def test_chronological_evaluator_uses_one_full_strictly_ordered_account_path() -> None:
    from tests.test_training_eval import _ds, _env_encoder

    dataset = _ds()
    env, encoder, window = _env_encoder()
    policy = CategoricalPolicy(
        encoder.config.spec.encoded_shape[0], ModelConfig(hidden_dim=16, num_layers=2)
    )
    evaluator = ChronologicalEvaluator(dataset, env, encoder, window)
    result = evaluator.evaluate(policy, "ppo", instruments=["EURUSD"], seed=42)
    report = result["per_instrument"]["EURUSD"]
    spec = chronological_spec(dataset, "EURUSD", "validation", 8, 42)

    assert result["evaluation_type"] == "chronological_per_instrument"
    assert result["is_portfolio_path"] is False
    assert result["combined_portfolio_metrics"] is None
    assert report["evaluation_type"] == "chronological"
    assert report["is_portfolio_path"] is True
    assert report["account_lifecycle_count"] == 1
    assert report["n_periods"] == spec.horizon
    assert report["timestamps_strictly_increasing"] is True
    assert report["duplicate_timestamp_count"] == 0
    assert report["overlapping_period_count"] == 0
    assert report["periods_per_year"] > 0
    assert np.isfinite(report["sharpe"])
    assert report["portfolio_sharpe"] == report["sharpe"]


def test_chronological_report_rejects_duplicate_timestamps() -> None:
    from forexmind.training.policies import PolicyAgent

    from tests.test_training_eval import _ds, _env_encoder

    dataset = _ds()
    env, encoder, window = _env_encoder()
    policy = CategoricalPolicy(
        encoder.config.spec.encoded_shape[0], ModelConfig(hidden_dim=16, num_layers=2)
    )
    evaluator = ChronologicalEvaluator(dataset, env, encoder, window)
    spec = chronological_spec(dataset, "EURUSD", "validation", 8, 42)
    trajectory = evaluator.runner.run_episode(PolicyAgent(policy, "ppo"), spec)
    trajectory.timestamps[1] = trajectory.timestamps[0]
    with pytest.raises(ValueError, match="strictly increasing"):
        chronological_instrument_report(trajectory)
