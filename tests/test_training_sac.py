"""Phase 3 tests: SAC networks, action range, update, entropy temperature,
replay buffer, and checkpointing."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from forexmind.training.config import ExperimentConfig, ModelConfig
from forexmind.training.networks import (
    SquashedGaussianActor,
    TwinQCritic,
    build_sac_networks,
    count_parameters,
    soft_update,
)
from forexmind.training.replay import ReplayBuffer
from forexmind.training.sac import SACTrainer

OBS_DIM = 351
ACTION_DIM = 1


def _model() -> ModelConfig:
    return ModelConfig(hidden_dim=16, num_layers=2, activation="relu")


def _obs(batch: int = 4) -> torch.Tensor:
    return torch.randn(batch, OBS_DIM)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def test_sac_actor_action_range() -> None:
    actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM, _model())
    obs = _obs()
    sampled, logp = actor.sample(obs)
    det = actor.deterministic(obs)
    for a in (sampled, det):
        assert a.shape == (4, ACTION_DIM)
        assert torch.all(a >= -1.0).item()
        assert torch.all(a <= 1.0).item()
    assert logp.shape == (4, 1)
    assert torch.isfinite(logp).all().item()


def test_sac_actor_sample_differs_from_mean() -> None:
    actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM, _model())
    obs = _obs()
    det = actor.deterministic(obs)
    samples = [actor.sample(obs)[0] for _ in range(5)]
    # With a nonzero log-std the samples should not all equal the mean.
    std = actor.log_std.detach().exp()
    assert torch.all(std > 1e-3).item()
    pooled = torch.cat(samples)
    assert not torch.allclose(pooled, det.repeat(5, 1), atol=1e-4)


def test_sac_actor_log_std_bounds() -> None:
    actor = SquashedGaussianActor(OBS_DIM, ACTION_DIM, _model())
    with torch.no_grad():
        actor.log_std.fill_(100.0)
        _mean, log_std = actor.forward(_obs())
    assert torch.all(log_std <= 2.0).item()
    assert torch.all(log_std >= -5.0).item()


def test_twin_critic_shapes() -> None:
    critic = TwinQCritic(OBS_DIM, ACTION_DIM, _model())
    obs, act = _obs(), torch.randn(4, ACTION_DIM)
    q1, q2 = critic(obs, act)
    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)
    assert torch.isfinite(q1).all().item()
    assert torch.isfinite(q2).all().item()


def test_target_critic_init_equals_critic() -> None:
    nets = build_sac_networks(OBS_DIM, ACTION_DIM, _model())
    obs, act = _obs(), torch.randn(4, ACTION_DIM)
    t1, t2 = nets.target_critic(obs, act)
    c1, c2 = nets.critic(obs, act)
    assert torch.allclose(t1, c1)
    assert torch.allclose(t2, c2)


def test_soft_update_full_copy() -> None:
    nets = build_sac_networks(OBS_DIM, ACTION_DIM, _model())
    with torch.no_grad():
        for p in nets.critic.parameters():
            p.add_(1.0)
    soft_update(nets.target_critic, nets.critic, tau=1.0)
    obs, act = _obs(), torch.randn(4, ACTION_DIM)
    t1, _ = nets.target_critic(obs, act)
    c1, _ = nets.critic(obs, act)
    assert torch.allclose(t1, c1)


def test_count_parameters_positive() -> None:
    nets = build_sac_networks(OBS_DIM, ACTION_DIM, _model())
    assert count_parameters(nets.actor) > 0
    assert count_parameters(nets.critic) > 0


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


def _fake_transition(
    i: int, obs_dim: int = OBS_DIM
) -> tuple[np.ndarray, float, float, np.ndarray, bool, bool]:
    obs = np.full((obs_dim,), float(i), dtype=np.float32)
    return obs, float(i % 3 - 1), float(i) * 0.01, obs + 1.0, False, False


def test_replay_ring_capacity() -> None:
    cap = 32
    buf = ReplayBuffer(OBS_DIM, cap, 1)
    for i in range(100):
        buf.push(*_fake_transition(i))
    assert buf.size == cap
    assert buf.full
    assert buf.metadata()["transitions_collected"] == 100


def test_replay_sample_shapes_and_finite() -> None:
    buf = ReplayBuffer(OBS_DIM, 100, 1)
    for i in range(50):
        buf.push(*_fake_transition(i))
    batch = buf.sample(16)
    assert batch.obs.shape == (16, OBS_DIM)
    assert batch.action.shape == (16, 1)
    assert batch.reward.shape == (16,)
    assert batch.next_obs.shape == (16, OBS_DIM)
    assert batch.terminated.shape == (16,)
    assert np.isfinite(batch.obs).all()


def test_replay_sample_seeded_deterministic() -> None:
    buf = ReplayBuffer(OBS_DIM, 100, 1)
    for i in range(50):
        buf.push(*_fake_transition(i))
    a = buf.sample_seeded(16, seed=7)
    b = buf.sample_seeded(16, seed=7)
    assert np.array_equal(a.obs, b.obs)
    c = buf.sample_seeded(16, seed=8)
    assert not np.array_equal(a.obs, c.obs)


def test_replay_metadata_roundtrip() -> None:
    buf = ReplayBuffer(OBS_DIM, 100, 1)
    for i in range(20):
        buf.push(*_fake_transition(i))
    meta = buf.state_meta()
    buf2 = ReplayBuffer(OBS_DIM, 100, 1)
    buf2.load_meta(meta)
    assert buf2._total_pushed == 20
    assert buf2.size == 0  # buffer itself is intentionally not restored


def test_replay_sample_too_large_raises() -> None:
    buf = ReplayBuffer(OBS_DIM, 10, 1)
    buf.push(*_fake_transition(0))
    with pytest.raises(ValueError):
        buf.sample(16)


# ---------------------------------------------------------------------------
# SAC trainer update / alpha / checkpoint
# ---------------------------------------------------------------------------


def _sac_trainer(tmp_path, *, alpha: float | str = "auto") -> SACTrainer:
    from tests.synthetic import make_instrument, make_split_dataset, timeline_m5

    dates = [
        "2020-01-06", "2020-06-01", "2020-12-07",
        "2021-03-01", "2021-09-01", "2022-03-01", "2022-09-01",
    ]
    ds = make_split_dataset(
        {"EURUSD": make_instrument("EURUSD", timeline_m5(dates, per_day=40))}
    )
    cfg = ExperimentConfig.smoke("sac")
    cfg.training.alpha = alpha
    cfg.environment.instruments = ("EURUSD",)
    cfg.logging.run_dir = "runs_smoke_test"
    return SACTrainer(cfg, tmp_path, dataset=ds)


def test_sac_trainer_auto_alpha_tunes(tmp_path) -> None:
    trainer = _sac_trainer(tmp_path, alpha="auto")
    assert trainer.auto_alpha
    assert trainer.alpha_opt is not None
    assert trainer.target_entropy == -1.0  # action_dim == 1
    # log_alpha is a leaf with requires_grad
    assert trainer.log_alpha.requires_grad


def test_sac_trainer_fixed_alpha(tmp_path) -> None:
    trainer = _sac_trainer(tmp_path, alpha=0.5)
    assert not trainer.auto_alpha
    assert trainer.alpha_opt is None
    assert trainer.log_alpha.detach().item() == pytest.approx(np.log(0.5))


def test_sac_update_produces_finite_losses(tmp_path) -> None:
    trainer = _sac_trainer(tmp_path)
    # Warm up the replay buffer manually with random transitions.
    for i in range(64):
        trainer.replay.push(
            np.random.default_rng(i).normal(size=trainer.obs_dim).astype(np.float32),
            float(np.random.default_rng(i).uniform(-1, 1)),
            0.001,
            np.random.default_rng(i).normal(size=trainer.obs_dim).astype(np.float32),
            False,
            False,
        )
    trainer._env_steps = trainer.config.training.warmup_steps + 1
    diag = trainer.update()
    for key in ("actor_loss", "critic_loss", "alpha", "entropy", "q1", "q2"):
        assert key in diag
        assert np.isfinite(diag[key]), key
    # The actor received a gradient update.
    assert any(p.grad is not None for p in trainer.actor.parameters())


def test_sac_checkpoint_roundtrip(tmp_path) -> None:
    trainer = _sac_trainer(tmp_path)
    with torch.no_grad():
        for p in trainer.actor.parameters():
            p.add_(0.5)
    before = [p.detach().clone() for p in trainer.actor.parameters()]
    trainer._save_checkpoint("roundtrip")

    trainer2 = _sac_trainer(tmp_path)
    state = trainer2.checkpoints.load(tmp_path / "checkpoints" / "roundtrip.pt")
    trainer2.load_state(state)
    after = [p.detach() for p in trainer2.actor.parameters()]
    for b, a in zip(before, after, strict=True):
        assert torch.allclose(b, a)


def test_sac_consume_transitions_updates_gradients(tmp_path) -> None:
    trainer = _sac_trainer(tmp_path)
    for i in range(trainer.config.training.warmup_steps + 32):
        trainer.replay.push(*_fake_transition(i, obs_dim=trainer.obs_dim))
    trainer._env_steps = trainer.config.training.warmup_steps + 32
    from forexmind.training.collector import Transition

    transitions = [
        Transition(
            obs=np.zeros(trainer.obs_dim, dtype=np.float32),
            action=0.1,
            reward=0.0,
            next_obs=np.zeros(trainer.obs_dim, dtype=np.float32),
            terminated=False,
            truncated=False,
        )
        for _ in range(16)
    ]
    g0 = trainer._gradient_updates
    trainer._consume_transitions(transitions)
    assert trainer._gradient_updates > g0
