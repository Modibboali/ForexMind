"""Phase 3 tests: reproducibility.

Same seed + same config + same dataset must produce identical training runs
in the deterministic sync backend (worker sampling, replay sampling, and
network init are all seeded).
"""

from __future__ import annotations

from forexmind.training.config import ExperimentConfig


def _volatile_dataset():
    from tests.synthetic import m5_mean_reverting, make_instrument, make_split_dataset

    m5 = m5_mean_reverting("2020-01-06", n=2400, mean=1.10, amplitude=0.01, period=50)
    return make_split_dataset({"EURUSD": make_instrument("EURUSD", m5)})


def _small_cfg(seed: int) -> ExperimentConfig:
    cfg = ExperimentConfig.smoke("sac")
    cfg.training.total_env_steps = 256
    cfg.training.warmup_steps = 60
    cfg.training.batch_size = 16
    cfg.training.collect_batch = 32
    cfg.training.replay_capacity = 2000
    cfg.training.gradient_updates_per_step = 1
    cfg.logging.log_every_env_steps = 80
    cfg.logging.evaluate_every_env_steps = 10**9  # no validation in this test
    cfg.logging.checkpoint_every_env_steps = 10**9
    cfg.logging.run_dir = "runs_smoke_test"
    cfg.environment.instruments = ("EURUSD",)
    cfg.compute.seed = seed
    return cfg


def _run(tmp_path, name: str, seed: int):
    from forexmind.training.sac import SACTrainer

    trainer = SACTrainer(_small_cfg(seed), tmp_path / name, dataset=_volatile_dataset())
    summary = trainer.train()
    lc = (tmp_path / name / "learning_curve.csv").read_text(encoding="utf-8")
    return summary, lc


def test_same_seed_same_run(tmp_path) -> None:
    s1, lc1 = _run(tmp_path, "a", seed=42)
    s2, lc2 = _run(tmp_path, "b", seed=42)
    assert lc1 == lc2
    assert s1["env_steps"] == s2["env_steps"] == 256
    assert s1["gradient_updates"] == s2["gradient_updates"]
    assert s1["best_validation_score"] == s2["best_validation_score"]


def test_different_seeds_differ(tmp_path) -> None:
    s1, lc1 = _run(tmp_path, "a", seed=1)
    s2, lc2 = _run(tmp_path, "b", seed=2)
    assert s1["env_steps"] == s2["env_steps"] == 256
    # On volatile data different exploration seeds produce different returns.
    assert lc1 != lc2


def test_episode_specs_deterministic() -> None:
    from forexmind.training.benchmark import build_test_episode_specs

    ds = _volatile_dataset()
    a = build_test_episode_specs(ds, split="train", n_episodes=5, horizon=16,
                                 context_length=8, seed=7)
    b = build_test_episode_specs(ds, split="train", n_episodes=5, horizon=16,
                                 context_length=8, seed=7)
    for x, y in zip(a, b, strict=True):
        assert x.instrument == y.instrument
        assert x.start_index == y.start_index
        assert x.seed == y.seed


def test_training_summary_and_curve_files_written(tmp_path) -> None:
    summary, _ = _run(tmp_path, "c", seed=42)
    run_dir = tmp_path / "c"
    assert (run_dir / "learning_curve.csv").is_file()
    assert (run_dir / "training_log.jsonl").is_file()
    assert (run_dir / "training_summary.json").is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert summary["status"] == "completed"
    assert "warnings" in summary
