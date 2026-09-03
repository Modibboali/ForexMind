"""Phase 3 tests: policy evaluation, checkpoint selection, and the final
benchmark tables (trained policy vs baselines on the untouched test split)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from forexmind.training.benchmark import (
    benchmark_test_split,
    load_checkpoint_policy,
    write_benchmark_results,
)
from forexmind.training.config import ExperimentConfig, ModelConfig
from forexmind.training.evaluator import PolicyEvaluator, selection_score
from forexmind.training.networks import SquashedGaussianActor
from forexmind.training.policies import PolicyAgent, sample_action


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


def _env_encoder():
    from forexmind.config import default_config
    from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
    from forexmind.observation.window import WindowConfig

    env_config = default_config(
        initial_balance="10000",
        leverage=50,
        spread_value=0.0002,
        sizing_mode="equity_fraction",
    )
    encoder = ObservationEncoder(EncoderConfig(context_length=8, initial_balance="10000"))
    return env_config, encoder, WindowConfig(context_length=8)


def _policy() -> SquashedGaussianActor:
    return SquashedGaussianActor(71, 1, ModelConfig(hidden_dim=16, num_layers=2))


# ---------------------------------------------------------------------------
# Selection score
# ---------------------------------------------------------------------------


def test_selection_score_default() -> None:
    m = {"sharpe": 1.5, "max_drawdown_pct": 0.1}
    assert selection_score(m, "sharpe_drawdown", 1.0) == pytest.approx(1.5 - 1.0 * 0.1)
    assert selection_score(m, "sharpe_drawdown", 2.0) == pytest.approx(1.5 - 2.0 * 0.1)
    assert selection_score(m, "sharpe") == pytest.approx(1.5)
    assert selection_score(m, "total_return") == pytest.approx(0.0)


def test_selection_score_unknown_metric_raises() -> None:
    with pytest.raises(ValueError):
        selection_score({}, "nope")


# ---------------------------------------------------------------------------
# Deterministic policy action selection
# ---------------------------------------------------------------------------


def test_policy_agent_deterministic() -> None:
    policy = _policy()
    obs = np.zeros(71, dtype=np.float32)
    a1 = sample_action(policy, obs, "sac", deterministic=True)
    a2 = sample_action(policy, obs, "sac", deterministic=True)
    assert a1 == a2
    assert -1.0 <= a1 <= 1.0
    agent = PolicyAgent(policy, "sac")
    assert agent.name == "sac"
    assert agent.policy is policy


# ---------------------------------------------------------------------------
# PolicyEvaluator integration
# ---------------------------------------------------------------------------


def test_policy_evaluator_returns_metrics() -> None:
    ds = _ds()
    env_config, encoder, window_config = _env_encoder()
    evaluator = PolicyEvaluator(
        ds,
        env_config,
        encoder,
        window_config,
        eval_horizon=16,
        eval_seed=42,
        context_length=8,
    )
    result = evaluator.evaluate(_policy(), "sac", "validation", 2, seed=42)
    assert result.split == "validation"
    for key in (
        "total_return",
        "sharpe",
        "sortino",
        "max_drawdown_pct",
        "_selection_score",
        "turnover",
        "mean_reward",
    ):
        assert key in result.metrics
    assert np.isfinite(result.score)


def test_policy_evaluator_deterministic_across_runs() -> None:
    ds = _ds()
    env_config, encoder, window_config = _env_encoder()
    evaluator = PolicyEvaluator(
        ds,
        env_config,
        encoder,
        window_config,
        eval_horizon=16,
        eval_seed=42,
        context_length=8,
    )
    policy = _policy()
    r1 = evaluator.evaluate(policy, "sac", "validation", 2, seed=42)
    r2 = evaluator.evaluate(policy, "sac", "validation", 2, seed=42)
    assert r1.score == pytest.approx(r2.score)
    assert r1.metrics["total_return"] == pytest.approx(r2.metrics["total_return"])


# ---------------------------------------------------------------------------
# Final benchmark tables
# ---------------------------------------------------------------------------


def test_benchmark_test_split_compares_all_agents() -> None:
    ds = _ds()
    env_config, encoder, window_config = _env_encoder()
    bench = benchmark_test_split(
        dataset=ds,
        env_config=env_config,
        encoder=encoder,
        window_config=window_config,
        policy=_policy(),
        algorithm="sac",
        split="test",
        n_episodes=2,
        horizon=16,
        seed=42,
    )
    assert bench["split"] == "test"
    agents = [r["agent"] for r in bench["results"]]
    assert agents[0] == "sac_trained"
    expected = {"flat", "long", "short", "random", "momentum", "mean_reversion", "sma_crossover"}
    assert expected <= set(agents)
    assert len(bench["results"]) == 8
    for r in bench["results"]:
        assert "sharpe" in r["metrics"]
        assert r["per_year"]  # per-year table is populated


def test_write_benchmark_results_files(tmp_path) -> None:
    ds = _ds()
    env_config, encoder, window_config = _env_encoder()
    bench = benchmark_test_split(
        dataset=ds,
        env_config=env_config,
        encoder=encoder,
        window_config=window_config,
        policy=_policy(),
        algorithm="sac",
        split="test",
        n_episodes=2,
        horizon=16,
        seed=42,
    )
    paths = write_benchmark_results(bench, tmp_path)
    assert paths["json"].is_file()
    assert paths["agent"].is_file()
    assert paths["per_instrument"].is_file()
    assert paths["per_year"].is_file()
    assert paths["text"].is_file()
    text = paths["text"].read_text(encoding="utf-8")
    assert "AGENT" in text
    assert "sac_trained" in text
    assert "flat" in text
    csv_text = paths["agent"].read_text(encoding="utf-8")
    assert "agent" in csv_text and "sharpe" in csv_text


def test_load_checkpoint_policy_roundtrip(tmp_path) -> None:
    from forexmind.training.sac import SACTrainer

    cfg = ExperimentConfig.smoke("sac")
    cfg.environment.instruments = ("EURUSD",)
    cfg.logging.run_dir = "runs_smoke_test"
    trainer = SACTrainer(cfg, tmp_path, dataset=_ds())
    trainer._save_checkpoint("x")
    policy, algorithm = load_checkpoint_policy(
        tmp_path / "checkpoints" / "x.pt", trainer.obs_dim, cfg.model
    )
    assert algorithm == "sac"
    for p1, p2 in zip(trainer.actor.parameters(), policy.parameters(), strict=True):
        assert np.allclose(p1.detach().numpy(), p2.detach().numpy())


def test_load_checkpoint_policy_roundtrip_ppo(tmp_path) -> None:
    """A PPO checkpoint must load a GaussianPolicy (was a SAC actor -> bug)."""
    from forexmind.training.networks import GaussianPolicy
    from forexmind.training.ppo import PPOTrainer

    cfg = ExperimentConfig.smoke("ppo")
    cfg.environment.instruments = ("EURUSD",)
    cfg.logging.run_dir = "runs_smoke_test"
    trainer = PPOTrainer(cfg, tmp_path, dataset=_ds())
    trainer._save_checkpoint("x")
    policy, algorithm = load_checkpoint_policy(
        tmp_path / "checkpoints" / "x.pt", trainer.obs_dim, cfg.model
    )
    assert algorithm == "ppo"
    assert isinstance(policy, GaussianPolicy)
    # And it must actually be usable through the PPO action path.
    action = sample_action(policy, np.zeros(trainer.obs_dim, dtype=np.float32), "ppo")
    assert -1.0 <= action <= 1.0
    for p1, p2 in zip(trainer.actor.parameters(), policy.parameters(), strict=True):
        assert np.allclose(p1.detach().numpy(), p2.detach().numpy())


# ---------------------------------------------------------------------------
# Turnover metric (execution_price is a str in the trade log)
# ---------------------------------------------------------------------------


def test_pooled_turnover_coerces_string_price() -> None:
    """execution_price is stored as a str; turnover must still be computed."""
    from forexmind.training.evaluator import _f, _pooled_turnover

    assert _f("1.20088") == pytest.approx(1.20088)
    assert _f("not-a-number", default=0.0) == 0.0
    assert _f(1.25) == pytest.approx(1.25)

    class _Traj:
        def __init__(self, trade_log):
            self.info = {"initial_balance": 10000.0}
            self.trade_log = trade_log

    class _Ev:
        def __init__(self, trajs):
            self.trajectories_by_instrument = {"EURUSD": trajs}

    ev = _Ev(
        [
            _Traj(
                [
                    {"units_delta": -1725.7325883395993, "execution_price": "1.20088"},
                    {"units_delta": 6603.40551283013, "execution_price": "1.20055"},
                ]
            )
        ]
    )
    turnover = _pooled_turnover(ev)
    assert turnover > 0.0  # was 0.0 before the _f string-coercion fix


# ---------------------------------------------------------------------------
# Checkpoint discovery (forgiving --checkpoint resolution)
# ---------------------------------------------------------------------------


def _make_run_checkpoints(root: Path) -> None:
    """Create a fake run dir with a best.pt so resolver tests can find it."""
    (root / "sac_cpu_seed42" / "checkpoints").mkdir(parents=True)
    (root / "sac_cpu_seed42" / "checkpoints" / "best.pt").write_bytes(b"x")
    (root / "sac_cpu_seed42" / "checkpoints" / "step_1000.pt").write_bytes(b"x")


def test_resolve_checkpoint_exact_path(tmp_path) -> None:
    from forexmind.training.checkpoint import resolve_checkpoint

    _make_run_checkpoints(tmp_path)
    p = tmp_path / "sac_cpu_seed42" / "checkpoints" / "best.pt"
    assert resolve_checkpoint(p) == p


def test_resolve_checkpoint_run_dir(tmp_path) -> None:
    from forexmind.training.checkpoint import resolve_checkpoint

    _make_run_checkpoints(tmp_path)
    run_dir = tmp_path / "sac_cpu_seed42"
    assert resolve_checkpoint(run_dir) == run_dir / "checkpoints" / "best.pt"


def test_resolve_checkpoint_run_name(tmp_path) -> None:
    from forexmind.training.checkpoint import resolve_checkpoint

    _make_run_checkpoints(tmp_path)
    resolved = resolve_checkpoint("sac_cpu_seed42", run_root=tmp_path)
    assert resolved == tmp_path / "sac_cpu_seed42" / "checkpoints" / "best.pt"


def test_resolve_checkpoint_missing_lists_candidates(tmp_path) -> None:
    from forexmind.training.checkpoint import resolve_checkpoint

    _make_run_checkpoints(tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        resolve_checkpoint("nonexistent_run", run_root=tmp_path)
    msg = str(exc.value)
    assert "not found" in msg
    assert "sac_cpu_seed42" in msg  # lists what is actually available
    assert "best.pt" in msg


def test_resolve_checkpoint_no_runs_clear_hint(tmp_path) -> None:
    from forexmind.training.checkpoint import resolve_checkpoint

    with pytest.raises(FileNotFoundError) as exc:
        resolve_checkpoint("sac_cpu_seed42", run_root=tmp_path)
    assert "Train first" in str(exc.value)


def test_resume_checkpoint_resolves_marker_and_latest_alias(tmp_path) -> None:
    from forexmind.training.checkpoint import resolve_resume_checkpoint

    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "step_100.pt"
    checkpoint.write_bytes(b"x")
    marker = checkpoint_dir / "latest.txt"
    marker.write_text("step_100.pt\n", encoding="utf-8")

    assert resolve_resume_checkpoint(marker) == checkpoint
    assert resolve_resume_checkpoint(checkpoint_dir / "latest.pt") == checkpoint
    assert resolve_resume_checkpoint(tmp_path / "run") == checkpoint
