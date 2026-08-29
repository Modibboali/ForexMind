"""Phase 3 PPO numerical-stability tests (docs/ppo_numerical_stability_audit.md).

Covers:
* the finite reward floor on equity collapse (was ``-inf`` -> NaN pipeline),
* hand-computed GAE (terminal / truncation bootstrap / mid-batch terminal),
* advantage normalization (standard + degenerate zero-std),
* first-non-finite detection (``finite_check`` raises ``FiniteError`` with
  instrument/timestamp context),
* gradient clipping, diagnostics keys, log-std bounds,
* worker/trainer log-prob consistency (clamped Gaussian, no tanh correction).
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
import torch
from forexmind.config import (
    EnvironmentConfig,
    ExecutionConfig,
    MarginConfig,
    PositionSizingConfig,
    RewardConfig,
)
from forexmind.data.dataset import InstrumentData, MarketDataset
from forexmind.environment import ForexEnvironment
from forexmind.training.collector import Transition
from forexmind.training.config import ExperimentConfig, ModelConfig
from forexmind.training.networks import GaussianPolicy
from forexmind.training.numerics import FiniteError, assert_finite, tensor_stats

from tests.synthetic import ladder_m1


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


def _trainer(tmp_path, **cfg_overrides):
    from forexmind.training.ppo import PPOTrainer

    cfg = ExperimentConfig.smoke("ppo")
    cfg.environment.instruments = ("EURUSD",)
    for key, value in cfg_overrides.items():
        setattr(cfg.training, key, value)
    return PPOTrainer(cfg, tmp_path, dataset=_ds())


# ---------------------------------------------------------------------------
# Reward finite floor (the -inf bug)
# ---------------------------------------------------------------------------


def test_reward_service_floor_on_collapse() -> None:
    rw = RewardConfig(min_reward=-50.0)
    from forexmind.environment.reward import RewardService

    svc = RewardService(rw)
    assert svc.reward("10000", "10100") == pytest.approx(np.log(1.01), rel=1e-6)
    # Equity collapse (curr <= 0) -> finite floor, NOT -inf.
    assert svc.reward("10000", "0") == -50.0
    assert svc.reward("10000", "-5000") == -50.0
    assert np.isfinite(svc.reward("10000", "-5000"))
    assert svc.collapse_count == 3


def test_reward_finite_floor_on_equity_collapse_end_to_end() -> None:
    """A price crash that drives equity <= 0 must yield a finite reward."""
    cfg = EnvironmentConfig(
        execution=ExecutionConfig(spread_value=0.0),
        margin=MarginConfig(
            initial_balance=Decimal("10000"),
            leverage=Decimal("100"),
            maintenance_margin_ratio=Decimal("0.5"),
        ),
        sizing=PositionSizingConfig(mode="fixed_units", fixed_units=Decimal("100000")),
        horizon=10,
        reward=RewardConfig(min_reward=-50.0),
    )
    # Two M5 bars at 1.10 (exec + hold), then a crash to 0.95.
    prices = [1.10] * 10 + [0.95] * 20
    ds = MarketDataset()
    ds.add(InstrumentData.from_m1("EURUSD", ladder_m1("2025-01-06 00:00", prices)))
    env = ForexEnvironment(ds, cfg)
    env.reset(seed=0, start_index=0)
    _obs, _r0, t0, _tr0, _i0 = env.step(4)  # go long at 1.10
    assert not t0
    _obs, r1, t1, _tr1, info1 = env.step(4)  # crash -> liquidation
    assert t1 and info1["liquidation"] is True
    assert np.isfinite(r1)
    assert r1 == -50.0  # finite floor replaces -inf


# ---------------------------------------------------------------------------
# GAE hand-computed
# ---------------------------------------------------------------------------


def test_gae_terminal_at_end(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    rew = np.array([1.0, 1.0, 1.0])
    val = np.zeros(3)
    next_val = np.zeros(3)
    done = np.array([False, False, True])  # terminal at the end
    adv, ret = trainer._compute_gae(rew, done, val, next_val)
    # gamma=0.99, lam=0.95
    np.testing.assert_allclose(adv, [2.82504025, 1.9405, 1.0], rtol=1e-5)
    np.testing.assert_allclose(ret, adv, rtol=1e-5)


def test_gae_truncation_bootstraps(tmp_path) -> None:
    """Truncated (not terminated) final step must bootstrap with V(s')."""
    trainer = _trainer(tmp_path)
    rew = np.array([1.0, 1.0, 1.0])
    val = np.zeros(3)
    next_val = np.full(3, 10.0)
    done = np.array([False, False, False])  # all truncated -> bootstrap
    adv, _ret = trainer._compute_gae(rew, done, val, next_val)
    np.testing.assert_allclose(adv, [30.792938725, 21.15145, 10.9], rtol=1e-5)


def test_gae_terminal_mid_batch_resets(tmp_path) -> None:
    """A terminal in the middle must stop the advantage recursion."""
    trainer = _trainer(tmp_path)
    rew = np.array([1.0, 1.0, 1.0, 1.0])
    val = np.zeros(4)
    next_val = np.zeros(4)
    done = np.array([False, False, True, False])  # terminal at index 2
    adv, _ret = trainer._compute_gae(rew, done, val, next_val)
    # indices 0,1 see the terminal at 2; index 3 is a fresh episode.
    np.testing.assert_allclose(adv, [2.82504025, 1.9405, 1.0, 1.0], rtol=1e-5)


def test_gae_normalization_standard(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    rew = np.array([1.0, 2.0, 3.0])
    done = np.array([True, True, True])
    val = np.zeros(3)
    next_val = np.zeros(3)
    adv, ret = trainer._gae(rew, done, val, next_val)
    assert abs(float(adv.mean())) < 1e-6
    assert abs(float(adv.std()) - 1.0) < 1e-4
    np.testing.assert_allclose(ret, rew, rtol=1e-5)


def test_gae_normalization_degenerate_zero_std(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    # Every step is a terminal: each advantage is just its own delta == 1,
    # so the batch has zero variance -> normalization yields zero signal.
    rew = np.array([1.0, 1.0])
    done = np.array([True, True])
    val = np.zeros(2)
    next_val = np.zeros(2)
    adv, ret = trainer._gae(rew, done, val, next_val)
    np.testing.assert_allclose(adv, [0.0, 0.0])
    np.testing.assert_allclose(ret, [1.0, 1.0])


# ---------------------------------------------------------------------------
# First-non-finite detection
# ---------------------------------------------------------------------------


def test_tensor_stats_counts() -> None:
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    st = tensor_stats(arr)
    assert st["nan_count"] == 1
    assert st["inf_count"] == 2
    assert st["finite_count"] == 2
    assert st["total_count"] == 5
    assert st["min"] == 1.0
    assert st["max"] == 2.0


def test_assert_finite_strict_raises() -> None:
    x = torch.tensor([1.0, float("nan")])
    with pytest.raises(FiniteError) as ei:
        assert_finite("test_component", x, strict=True)
    assert "test_component" in str(ei.value)
    assert "nan_count" in str(ei.value)


def test_finite_check_raises_on_nan_obs_with_context(tmp_path) -> None:
    trainer = _trainer(tmp_path, finite_check=True)
    obs = np.zeros(trainer.obs_dim, dtype=np.float32)
    obs[0] = np.nan
    t = Transition(
        obs=obs,
        action=0.0,
        reward=0.0,
        next_obs=np.zeros(trainer.obs_dim, dtype=np.float32),
        terminated=False,
        truncated=False,
        worker_id=0,
        instrument="EURUSD",
        timestamp="2020-01-06",
    )
    trainer._rollout = [t]
    with pytest.raises(FiniteError) as ei:
        trainer.update()
    msg = str(ei.value)
    assert "observation" in msg
    assert "EURUSD" in msg
    assert "2020-01-06" in msg


def test_finite_check_raises_on_inf_reward(tmp_path) -> None:
    """A -inf reward (the pre-fix liquidation bug) must be caught at 'reward'."""
    trainer = _trainer(tmp_path, finite_check=True)
    obs = np.zeros(trainer.obs_dim, dtype=np.float32)
    t = Transition(
        obs=obs,
        action=0.0,
        reward=float("-inf"),
        next_obs=obs.copy(),
        terminated=True,
        truncated=False,
        worker_id=0,
        instrument="EURUSD",
    )
    trainer._rollout = [t]
    with pytest.raises(FiniteError) as ei:
        trainer.update()
    assert "reward" in str(ei.value)


def test_inf_reward_poisons_gae_without_guard(tmp_path) -> None:
    """Documents the root cause: one -inf reward NaN's the whole GAE batch.

    With the pre-fix env behavior (liquidation -> reward=-inf), the raw GAE
    advantages become -inf, and advantage normalization (subtract mean, divide
    by std) produces NaN everywhere -> NaN gradients -> NaN actor weights ->
    Normal(loc=nan).  This is why the reward floor is a concrete-bug fix, not
    a cosmetic clip.
    """
    trainer = _trainer(tmp_path)
    rew = np.array([1.0, float("-inf"), 1.0])
    done = np.array([False, False, False])
    val = np.zeros(3)
    next_val = np.zeros(3)
    raw_adv, _ = trainer._compute_gae(rew, done, val, next_val)
    assert not np.isfinite(raw_adv).all()  # -inf poisons the recursion
    with np.errstate(invalid="ignore"):
        std = float(raw_adv.std())
        normalized = (raw_adv - raw_adv.mean()) / (std + trainer._adv_epsilon)
    assert np.isnan(normalized).any()  # -> NaN advantages


# ---------------------------------------------------------------------------
# Gradient clipping + diagnostics
# ---------------------------------------------------------------------------


def test_ppo_update_diagnostics_and_grad_clip(tmp_path) -> None:
    trainer = _trainer(tmp_path, max_grad_norm=0.5, finite_check=False)
    transitions = [trainer.collector.worker.step(random_action=False) for _ in range(64)]
    trainer._rollout = transitions
    diag = trainer.update()
    expected = (
        "actor_loss",
        "critic_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "actor_grad_norm",
        "critic_grad_norm",
        "actor_param_max",
        "critic_param_max",
        "reward_min",
        "reward_max",
        "reward_mean",
        "reward_std",
        "return_min",
        "return_max",
        "advantage_std",
        "mean_action",
        "std_action",
        "mean_abs_action",
        "frac_near_minus_one",
        "frac_near_zero",
        "frac_near_plus_one",
    )
    for key in expected:
        assert key in diag, key
    assert np.isfinite(diag["actor_loss"])
    assert np.isfinite(diag["critic_loss"])
    # Post-clip gradient norms respect the configured max_grad_norm.
    assert diag["actor_grad_norm"] <= 0.5 + 1e-3
    assert diag["critic_grad_norm"] <= 0.5 + 1e-3
    # Parameters stay finite after an update.
    assert np.isfinite(diag["actor_param_max"])
    assert np.isfinite(diag["critic_param_max"])


def test_ppo_grad_clip_disabled_allows_large_norm(tmp_path) -> None:
    trainer = _trainer(tmp_path, max_grad_norm=0.0, finite_check=False)
    transitions = [trainer.collector.worker.step(random_action=False) for _ in range(64)]
    trainer._rollout = transitions
    diag = trainer.update()
    assert "actor_grad_norm" in diag  # mechanism runs without clipping


# ---------------------------------------------------------------------------
# log-std bounds + worker/trainer log-prob consistency
# ---------------------------------------------------------------------------


def test_gaussian_policy_log_std_bounds() -> None:
    model = ModelConfig(hidden_dim=16, num_layers=2)
    policy = GaussianPolicy(5, 1, model, log_std_min=-5.0, log_std_max=2.0)
    with torch.no_grad():
        policy.log_std.fill_(100.0)  # would overflow exp() if unconstrained
    dist = policy.dist(torch.zeros(4, 5))
    assert torch.isfinite(dist.scale).all()
    assert float(dist.scale.max().detach()) <= float(np.exp(2.0)) + 1e-6
    assert float(dist.scale.min().detach()) >= float(np.exp(-5.0)) - 1e-6


def test_ppo_worker_logprob_consistent_with_trainer(tmp_path) -> None:
    """Stored old_log_prob must be N(u; mu, sigma) at the RAW sample u."""
    trainer = _trainer(tmp_path)
    trainer._sync_policy_to_workers()
    t = trainer.collector.worker.step(random_action=False)
    obs_t = torch.as_tensor(t.obs, dtype=torch.float32).unsqueeze(0)
    raw_t = torch.as_tensor([[t.action_raw]], dtype=torch.float32)
    dist = trainer.actor.dist(obs_t)
    expected = float(dist.log_prob(raw_t).sum().item())
    # The stored log-prob is the density of the actual pre-clamp sample u.
    assert abs(t.log_prob - expected) < 1e-5
    # The env-facing action is the clamped projection of the raw sample.
    assert -1.0 <= t.action <= 1.0
    assert abs(t.action - float(np.clip(t.action_raw, -1.0, 1.0))) < 1e-6


def test_ppo_minibatch_math_hand_computed(tmp_path) -> None:
    """Mandatory tiny PPO mathematical test (docs/ppo_math_audit.md §18).

    Manually computes old/new log-prob, ratio, clipped objective, KL, and
    entropy for a 4-sample batch and compares against the implementation.
    """
    import math

    from torch.distributions import Normal

    trainer = _trainer(tmp_path, ppo_epochs=1)
    mean = torch.tensor([0.5, -0.2, 0.1, 0.7], dtype=torch.float32).view(-1, 1)
    std = torch.tensor([1.0, 1.2, 0.8, 1.5], dtype=torch.float32).view(-1, 1)
    dist = Normal(mean, std)
    act_raw = torch.tensor([[0.3], [-0.5], [0.2], [0.9]], dtype=torch.float32)
    old_logp = torch.tensor([[-1.0], [-0.5], [-1.5], [-0.3]], dtype=torch.float32)
    adv = torch.tensor([[0.5], [-0.3], [0.8], [-0.6]], dtype=torch.float32)
    eps = 0.2
    ent_coef = 0.01

    def gauss_logpdf(a: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        return -0.5 * ((a - mu) / sig) ** 2 - torch.log(sig) - 0.5 * math.log(2 * math.pi)

    new_logp = torch.cat(
        [gauss_logpdf(act_raw[i], mean[i], std[i]) for i in range(4)]
    ).view(-1, 1)
    entropy_manual = (0.5 * math.log(2 * math.pi * math.e) + torch.log(std)).mean()

    log_ratio_manual = torch.clamp(new_logp - old_logp, -10.0, 10.0)
    ratio_manual = log_ratio_manual.exp()
    clip_frac_manual = float(((ratio_manual - 1.0).abs() > eps).float().mean().item())
    approx_kl_manual = float((0.5 * (ratio_manual - 1.0).pow(2)).mean().item())
    approx_kl_log_manual = float(((ratio_manual - 1.0) - log_ratio_manual).mean().item())
    surr1 = ratio_manual * adv
    surr2 = ratio_manual.clamp(1.0 - eps, 1.0 + eps) * adv
    actor_loss_manual = -(torch.min(surr1, surr2).mean() + ent_coef * entropy_manual)

    actor_loss, ratio, approx_kl, approx_kl_log, clip_frac, entropy = (
        trainer._minibatch_actor(dist, act_raw, old_logp, adv, eps=eps, ent_coef=ent_coef)
    )
    np.testing.assert_allclose(ratio.detach().numpy(), ratio_manual.numpy(), rtol=1e-6)
    assert approx_kl == pytest.approx(approx_kl_manual, rel=1e-6)
    assert approx_kl_log == pytest.approx(approx_kl_log_manual, rel=1e-6)
    assert clip_frac == pytest.approx(clip_frac_manual, rel=1e-6)
    np.testing.assert_allclose(entropy.detach().numpy(), entropy_manual.numpy(), rtol=1e-6)
    np.testing.assert_allclose(
        actor_loss.detach().numpy(), actor_loss_manual.detach().numpy(), rtol=1e-6
    )


def test_ppo_target_kl_early_stop(tmp_path) -> None:
    """A tiny target-KL must stop the remaining PPO epochs for the rollout."""
    trainer = _trainer(tmp_path, ppo_epochs=8, ppo_target_kl=1e-6, finite_check=False)
    transitions = [trainer.collector.worker.step(random_action=False) for _ in range(64)]
    trainer._rollout = transitions
    diag = trainer.update()
    assert diag["early_stop_kl"] == 1.0
    assert diag["kl_stop_epoch"] == 1.0  # stopped after the first epoch
    # Diagnostics keys are still populated.
    assert "approx_kl_log" in diag
    assert "mean_abs_parameter_update" in diag


def test_gaussian_policy_evaluate_matches_dist_logprob() -> None:
    model = ModelConfig(hidden_dim=16, num_layers=2)
    policy = GaussianPolicy(5, 1, model)
    obs = torch.randn(3, 5)
    a = torch.clamp(torch.randn(3, 1), -1.0, 1.0)
    lp, _ = policy.evaluate(obs, a)
    dist = policy.dist(obs)
    np.testing.assert_allclose(
        lp.detach().numpy(),
        dist.log_prob(a).sum(-1, keepdim=True).detach().numpy(),
        rtol=1e-5,
    )
