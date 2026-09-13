"""Stage 4.5 integrated-loop tests: lifecycle, replay store, diagnostics, resume."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from forexmind.config import (
    EnvironmentConfig,
    ExecutionConfig,
    MarginConfig,
    PositionSizingConfig,
)
from forexmind.muzero.diagnostics import (
    RootSearchRecord,
    assert_records_are_legal,
    search_summary,
    staleness_summary,
)
from forexmind.muzero.replay_store import load_replay, replay_store_report, save_replay
from forexmind.muzero.trainer import MuZeroTrainer, MuZeroTrainingConfig
from forexmind.observation.encoder import EncoderConfig

from tests.synthetic import make_instrument, make_split_dataset, timeline_m5

DATES = [
    "2020-01-06",
    "2020-03-02",
    "2020-06-01",
    "2020-09-07",
    "2020-12-07",
    "2021-03-01",
    "2021-06-07",
    "2021-09-06",
    "2021-12-06",
]


def _dataset():
    return make_split_dataset({"EURUSD": make_instrument("EURUSD", timeline_m5(DATES, per_day=40))})


def _env_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        execution=ExecutionConfig(spread_mode="fixed", spread_value=0.0),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")),
        sizing=PositionSizingConfig(mode="equity_fraction"),
    )


def _trainer(tmp_path, **overrides) -> MuZeroTrainer:
    base: dict = dict(
        horizon=4,
        num_simulations=3,
        trajectories_per_iteration=1,
        min_replay_transitions_before_training=4,
        learner_updates_per_iteration=1,
        batch_size=2,
        unroll_steps=2,
        td_steps=2,
        latent_dim=8,
        hidden_dim=8,
        num_layers=1,
        max_trajectories=16,
        max_env_steps=8,
        eval_every_env_steps=8,
        eval_episodes=2,
        eval_horizon=4,
        checkpoint_every_env_steps=8,
        output_dir=tmp_path / "run",
        seed=0,
    )
    base.update(overrides)
    return MuZeroTrainer(
        _dataset(),
        _env_config(),
        EncoderConfig(context_length=8),
        MuZeroTrainingConfig(**base),
    )


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


def test_warmup_gates_learning_until_replay_is_ready(tmp_path) -> None:
    trainer = _trainer(
        tmp_path,
        horizon=2,
        num_simulations=2,
        min_replay_transitions_before_training=100,
        learner_updates_per_iteration=3,
        max_env_steps=4,
        eval_every_env_steps=10_000,
        checkpoint_every_env_steps=10_000,
    )
    progress = trainer.run_iteration()
    assert progress.learner_updates == 0
    assert trainer.gradient_updates == 0
    assert trainer.network_version == 0
    assert progress.replay_transitions > 0
    assert trainer.warmup_complete() is False


def test_learning_starts_after_warmup_and_respects_the_configured_ratio(tmp_path) -> None:
    trainer = _trainer(
        tmp_path,
        horizon=2,
        num_simulations=2,
        min_replay_transitions_before_training=4,
        learner_updates_per_iteration=3,
        max_env_steps=4,
        eval_every_env_steps=10_000,
        checkpoint_every_env_steps=10_000,
    )
    first = trainer.run_iteration()
    assert first.learner_updates == 0  # warm-up: 2 of 4 required transitions
    second = trainer.run_iteration()
    assert second.learner_updates == 3  # exactly the configured ratio, not more
    assert trainer.gradient_updates == 3
    assert trainer.network_version == 1  # one increment per update group
    assert second.updates_per_env_step == pytest.approx(3 / 2)
    assert all(np.isfinite(value) for value in second.learner.values())


def test_search_diagnostics_and_staleness_are_reported(tmp_path) -> None:
    trainer = _trainer(
        tmp_path,
        horizon=2,
        num_simulations=2,
        min_replay_transitions_before_training=2,
        learner_updates_per_iteration=1,
        max_env_steps=6,
        eval_every_env_steps=10_000,
        checkpoint_every_env_steps=10_000,
    )
    trainer.run_iteration()
    progress = trainer.run_iteration()
    search = progress.search
    for key in (
        "search_changed_argmax_fraction",
        "root_visit_entropy_mean",
        "mcts_network_kl_mean",
        "tree_depth_mean",
        "tree_depth_max",
        "reward_model_mae",
        "network_root_value_mean",
        "mcts_root_value_mean",
        "root_value_abs_delta_mean",
        "mean_valid_actions",
        "selected_hold_fraction",
        "selected_hold_mean_visits",
    ):
        assert key in search, key
    assert search["tree_depth_max"] >= 1
    assert 0.0 <= search["search_changed_argmax_fraction"] <= 1.0
    assert "mean_staleness" in progress.staleness


def test_checkpoints_are_separate_best_latest_and_step_files(tmp_path) -> None:
    trainer = _trainer(
        tmp_path,
        horizon=2,
        num_simulations=2,
        min_replay_transitions_before_training=2,
        learner_updates_per_iteration=1,
        max_env_steps=4,
        eval_every_env_steps=2,
        checkpoint_every_env_steps=2,
    )
    report = trainer.train()
    directory = trainer.output_dir
    assert (directory / "latest.pt").exists()
    assert (directory / "best.pt").exists()
    assert list(directory.glob("step_*.pt"))
    assert report["best_validation_score"] is not None
    assert report["selection_metric_name"] == "mean_episode_log_return"
    payload = report["counters"]
    assert payload["network_version"] >= 1
    assert payload["trajectories_collected"] == report["replay"]["num_trajectories"]


def test_resume_restores_counters_and_replay(tmp_path) -> None:
    first = _trainer(
        tmp_path,
        horizon=2,
        num_simulations=2,
        min_replay_transitions_before_training=2,
        learner_updates_per_iteration=1,
        max_env_steps=4,
        eval_every_env_steps=4,
        checkpoint_every_env_steps=4,
    )
    first.train()
    resumable = _trainer(
        tmp_path,
        horizon=2,
        num_simulations=2,
        min_replay_transitions_before_training=2,
        learner_updates_per_iteration=1,
        max_env_steps=6,
        eval_every_env_steps=4,
        checkpoint_every_env_steps=4,
    )
    report = resumable.resume(first.output_dir / "latest.pt")
    assert report["resumed"] is True
    assert report["replay_restored"] is True
    assert report["exact_continuation"] is True
    assert report["env_steps"] == first.env_steps
    assert report["gradient_updates"] == first.gradient_updates
    assert report["network_version"] == first.network_version
    assert len(resumable.replay) == len(first.replay)
    # Continuing produces valid updates from the restored replay.
    progress = resumable.run_iteration()
    assert progress.learner_updates == 1
    assert np.isfinite(progress.learner["total_loss"])


def test_trainer_refuses_a_non_train_split(tmp_path) -> None:
    with pytest.raises(ValueError, match="TRAIN"):
        _trainer(tmp_path, split="validation")


# --------------------------------------------------------------------------- #
# replay persistence (S30)
# --------------------------------------------------------------------------- #


def test_replay_store_round_trip_preserves_order_and_metadata(tmp_path) -> None:
    trainer = _trainer(
        tmp_path,
        horizon=2,
        num_simulations=2,
        min_replay_transitions_before_training=2,
        learner_updates_per_iteration=1,
        max_env_steps=4,
        eval_every_env_steps=10_000,
        checkpoint_every_env_steps=10_000,
    )
    trainer.run_iteration()
    trainer.run_iteration()
    directory = trainer.replay_dir
    save_replay(trainer.replay, directory)
    restored, report = load_replay(directory, config=trainer.replay.config)
    assert report.trajectories_loaded == len(trainer.replay)
    assert report.trajectories_dropped_on_load == 0
    assert [t.metadata.trajectory_id for t in restored.trajectories] == [
        t.metadata.trajectory_id for t in trainer.replay.trajectories
    ]
    assert [t.metadata.network_version for t in restored.trajectories] == [
        t.metadata.network_version for t in trainer.replay.trajectories
    ]
    for original, copy in zip(trainer.replay.trajectories, restored.trajectories, strict=True):
        assert np.array_equal(original.observations, copy.observations)
        assert np.array_equal(original.root_policies, copy.root_policies)
        assert np.array_equal(original.action_masks, copy.action_masks)
        copy.validate()
    assert replay_store_report(directory)["num_trajectories"] == len(restored)


def test_replay_store_rejects_a_missing_or_foreign_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_replay(tmp_path / "nope")
    (tmp_path / "replay_index.json").write_text('{"format": "something-else"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a ForexMind"):
        load_replay(tmp_path)


# --------------------------------------------------------------------------- #
# diagnostics (S16-S22, S33-S35, S39)
# --------------------------------------------------------------------------- #


def _record(**overrides) -> RootSearchRecord:
    base: dict = dict(
        prior=np.array([0.5, 0.5, 0, 0, 0, 0], dtype=float),
        visits=np.array([3.0, 1.0, 0, 0, 0, 0]),
        policy=np.array([0.75, 0.25, 0, 0, 0, 0]),
        q_values=np.zeros(6),
        predicted_rewards=np.array([0.0, -0.1, -0.2, -0.05, 0.1, 0.2]),
        mask=np.array([True] * 6),
        action=0,
        network_value=0.1,
        search_value=0.2,
        tree_depth=2,
        real_reward=0.0,
    )
    base.update(overrides)
    return RootSearchRecord(**base)


def test_search_summary_reports_search_improvement_and_reward_fit() -> None:
    records = [
        _record(),
        _record(
            prior=np.array([1.0, 0, 0, 0, 0, 0]),
            visits=np.array([0.0, 2.0, 0, 0, 0, 0]),
            policy=np.array([0.0, 1.0, 0, 0, 0, 0]),
            action=1,
            real_reward=-0.1,
            network_value=0.3,
            search_value=0.1,
            tree_depth=4,
        ),
    ]
    summary = search_summary(records)
    assert summary["search_changed_argmax_fraction"] == pytest.approx(0.5)
    assert summary["tree_depth_mean"] == pytest.approx(3.0)
    assert summary["tree_depth_max"] == 4.0
    assert summary["reward_model_mae"] == pytest.approx(0.0)  # predictions match reality
    assert summary["root_value_abs_delta_mean"] == pytest.approx(0.15)
    assert summary["selected_hold_fraction"] == pytest.approx(0.5)
    assert summary["selected_hold_mean_visits"] == pytest.approx(3.0)
    assert summary["mean_valid_actions"] == pytest.approx(6.0)
    assert summary["fraction_flat_masked"] == 0.0
    assert 0.0 <= summary["root_visit_entropy_mean"] <= np.log(6) + 1e-9


def test_search_summary_fails_loudly_on_mask_violations() -> None:
    with pytest.raises(ValueError, match="invalid"):
        assert_records_are_legal(
            [_record(action=1, mask=np.array([True, False, True, True, True, True]))]
        )
    visits_on_invalid = np.array([3.0, 1.0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="visits"):
        assert_records_are_legal(
            [
                _record(
                    visits=visits_on_invalid,
                    mask=np.array([True, False, False, False, False, False]),
                )
            ]
        )
    with pytest.raises(ValueError, match="HOLD"):
        assert_records_are_legal([_record(mask=np.array([False] * 6))])


def test_staleness_summary_reports_mean_median_and_max() -> None:
    summary = staleness_summary(np.array([0.0, 1.0, 2.0, 5.0]))
    assert summary["mean_staleness"] == pytest.approx(2.0)
    assert summary["median_staleness"] == pytest.approx(1.5)
    assert summary["max_staleness"] == 5.0
    assert staleness_summary(np.array([]))["max_staleness"] == 0.0


def test_validation_uses_the_ppo_selection_metric(tmp_path) -> None:
    trainer = _trainer(
        tmp_path,
        horizon=2,
        num_simulations=2,
        min_replay_transitions_before_training=2,
        learner_updates_per_iteration=1,
        max_env_steps=4,
        eval_every_env_steps=4,
        checkpoint_every_env_steps=4,
    )
    evaluation = trainer.evaluate()
    assert evaluation.metrics["selection_metric_name"] == "mean_episode_log_return"
    headline = evaluation.headline()
    for key in (
        "mean_episode_log_return",
        "mean_episode_return",
        "median_episode_return",
        "profitable_episode_fraction",
        "p10_episode_return",
        "p90_episode_return",
        "mean_turnover",
        "mean_executions",
    ):
        assert key in headline, key
    assert evaluation.score == pytest.approx(float(headline["mean_episode_log_return"]))
    steps = sum(t.n_steps for values in evaluation.trajectories.values() for t in values)
    assert evaluation.search["roots"] == pytest.approx(float(steps))
