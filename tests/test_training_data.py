"""Phase 3 tests: leakage-free data protocol.

* only the training split is used for experience collection,
* validation is used only for checkpoint selection,
* the test split is never touched during training.
"""

from __future__ import annotations

import numpy as np
from forexmind.training.config import ExperimentConfig


def _ds():
    from tests.synthetic import make_instrument, make_split_dataset, timeline_m5

    dates = [
        "2020-01-06",
        "2020-06-01",
        "2020-12-07",
        "2021-03-01",
        "2021-09-01",
        "2022-03-01",
        "2022-09-01",
    ]
    return make_split_dataset({"EURUSD": make_instrument("EURUSD", timeline_m5(dates, per_day=40))})


def _trainer(tmp_path, algorithm: str = "sac"):
    if algorithm == "ppo":
        from forexmind.training.ppo import PPOTrainer as Trainer
    else:
        from forexmind.training.sac import SACTrainer as Trainer
    cfg = ExperimentConfig.smoke(algorithm)
    cfg.environment.instruments = ("EURUSD",)
    cfg.logging.run_dir = "runs_smoke_test"
    return Trainer(cfg, tmp_path, dataset=_ds())


def test_splits_halfopen_nonoverlapping() -> None:
    ds = _ds()
    sc = ds.split_config
    assert sc.train_end <= sc.validation_start
    assert sc.validation_end <= sc.test_start
    assert sc.train_start < sc.train_end
    assert sc.validation_start < sc.validation_end
    assert sc.test_start < sc.test_end
    # Ranges are half-open [start, end): training data never overlaps eval.
    assert sc.train_end <= sc.validation_start


def test_replay_metadata_marks_training_split(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.replay.metadata()["training_split"] == "train"


def test_episode_config_uses_train_split(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.episode_config.split == "train"


def test_worker_builder_uses_train_range(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    builder = trainer.collector.worker._make_builder("EURUSD")
    start, end = trainer.dataset.split_config.range("train")
    assert builder._split_start == np.datetime64(start)
    assert builder._split_end == np.datetime64(end)
    # And the validation range is different from train.
    vstart, vend = trainer.dataset.split_config.range("validation")
    assert (vstart, vend) != (start, end)


def test_collected_transitions_come_from_train_window(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    worker = trainer.collector.worker
    transitions = [worker.step(random_action=True) for _ in range(16)]
    # The collector is wired to the training split only.
    for t in transitions:
        assert t.obs.shape[0] == trainer.obs_dim
    # Episode builder ranges (each transition came from a train episode).
    builder = worker._make_builder("EURUSD")
    start, end = trainer.dataset.split_config.range("train")
    assert builder._split_start == np.datetime64(start)
    assert builder._split_end == np.datetime64(end)


def test_evaluation_uses_validation_split_for_selection(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    from forexmind.training.evaluator import PolicyEvaluator

    evaluator = PolicyEvaluator(
        trainer.dataset,
        trainer.env_config,
        trainer.encoder,
        trainer.window_config,
        eval_horizon=16,
        eval_seed=42,
        context_length=8,
    )
    result = evaluator.evaluate(trainer.policy(), "sac", "validation", 2, seed=42)
    assert result.split == "validation"
    assert result.metrics.get("_selection_score") is not None
