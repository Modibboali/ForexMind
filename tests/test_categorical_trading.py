"""Stage 3.4 action-system correctness, including economic and PPO invariants."""

from dataclasses import replace
from decimal import Decimal
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from forexmind.config import ExecutionConfig, PositionSizingConfig
from forexmind.environment import ForexEnvironment
from forexmind.environment.actions import ACTION_NAMES, valid_action_mask
from forexmind.training.action_diagnostics import ActionDiagnostics
from forexmind.training.config import ModelConfig
from forexmind.training.networks import CategoricalPolicy, TanhGaussianPolicy
from gymnasium.spaces import Discrete

from tests.test_environment import _config, _dataset


def make_env(pair="EURUSD", price=1.1, costs=False):
    cfg = replace(
        _config(close_at_episode_end=False), sizing=PositionSizingConfig(mode="equity_fraction")
    )
    if costs:
        cfg = replace(
            cfg,
            execution=ExecutionConfig(
                spread_value=0.0002,
                commission_per_unit=0.00001,
                slippage_mode="fixed",
                slippage_value=0.00005,
            ),
        )
    env = ForexEnvironment(_dataset(instrument=pair, price=price), cfg)
    env.reset(start_index=0, horizon=8)
    return env


@pytest.mark.parametrize("pair,price", [("EURUSD", 1.1), ("USDJPY", 150.0)])
@pytest.mark.parametrize(
    "index,target", list(enumerate([0, 0, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1]))[2:]
)
def test_all_exposure_targets_and_masks(pair, price, index, target):
    env = make_env(pair, price)
    assert isinstance(env.action_space, Discrete) and env.action_space.n == 10
    assert not env.action_masks()[1] and env.action_masks()[0]
    obs, *_ = env.step(index)
    signed = float(obs.account.gross_exposure / obs.account.equity) * np.sign(
        float(obs.account.position_units)
    )
    assert signed == pytest.approx(target)
    assert not env.action_masks()[index]
    assert env.action_masks()[0]


@pytest.mark.parametrize("index", [3, 7])
def test_hold_skips_sizing_and_execution_even_after_equity_change(index):
    env = make_env(costs=True)
    env.step(index)
    units = env.portfolio.position.units
    env.portfolio.apply_cash_adjustment(Decimal("100"))
    env._target_units = Mock(side_effect=AssertionError("HOLD attempted sizing"))
    env._engine.execute = Mock(side_effect=AssertionError("HOLD attempted execution"))
    obs, reward, _, _, info = env.step(0)
    assert obs.account.position_units == units
    assert info["units_delta"] == info["trade_cost"] == info["execution_commission"] == 0
    assert info["execution_price"] is None
    assert info["action_diagnostics"]["actual_executions"] == 0
    assert np.isfinite(reward)


@pytest.mark.parametrize("index", [3, 7])
def test_flat_closes_long_and_short_with_normal_costs(index):
    env = make_env(costs=True)
    env.step(index)
    units = env.portfolio.position.units
    obs, _, _, _, info = env.step(1)
    assert obs.account.position_units == 0
    assert info["units_delta"] == -units
    assert info["execution_commission"] > 0
    _, _, _, _, info = env.step(1)
    assert info["units_delta"] == 0 and info["execution_price"] is None


def test_mask_tolerance_and_residual_flat():
    assert not valid_action_mask(0.5005, is_flat=False)[7]
    assert valid_action_mask(0.502, is_flat=False)[7]
    assert not valid_action_mask(-0.7505, is_flat=False)[3]
    assert valid_action_mask(-0.748, is_flat=False)[3]
    assert valid_action_mask(1e-12, is_flat=False)[1]
    for exposure in [-2, -1, -0.75, 0, 0.5, 1, 2]:
        assert valid_action_mask(exposure, is_flat=False)[0]


def test_categorical_probability_sampling_and_ratio():
    torch.manual_seed(42)
    policy = CategoricalPolicy(5, ModelConfig(hidden_dim=16, num_layers=1))
    obs = torch.zeros(4096, 5)
    masks = torch.ones(4096, 10, dtype=torch.bool)
    masks[:2048, 1] = False
    masks[2048:, 7] = False
    dist = policy.dist(obs, masks)
    assert torch.all(dist.probs[~masks] == 0)
    torch.testing.assert_close(dist.probs.sum(-1), torch.ones(4096))
    action, logp = policy.sample(obs, masks)
    assert masks.gather(1, action[:, None]).all()
    assert set(action.tolist()) == set(range(10))
    chosen = policy.act(obs, masks, deterministic=True)
    assert masks.gather(1, chosen[:, None]).all()
    reconstructed, entropy = policy.evaluate(obs, action, masks)
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
    torch.testing.assert_close((reconstructed - logp).exp(), torch.ones_like(logp))
    assert not any("log_std" in name or "mean" in name for name in policy.state_dict())


def test_old_checkpoint_rejected_before_any_weights_change():
    model = ModelConfig(hidden_dim=16, num_layers=1)
    policy = CategoricalPolicy(5, model)
    old = TanhGaussianPolicy(5, 1, model)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    with pytest.raises(ValueError, match="Start a new categorical PPO run"):
        policy.load_state_dict(old.state_dict())
    for k, v in policy.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_durations_and_worker_isolation_across_fragments():
    diag = ActionDiagnostics()

    def record(action, before, after, step, worker=0, done=False):
        diag.record(
            action,
            {
                "units_before": before,
                "units_after_policy": after,
                "actual_executions": int(before != after),
            },
            step=step,
            worker=worker,
            done=done,
        )

    record(7, 0, 50, 0)
    record(0, 50, 50, 1)
    record(3, 0, -75, 0, worker=1)
    record(0, 50, 50, 2)
    record(0, -75, -75, 1, worker=1, done=True)
    record(1, 50, 0, 3, done=True)
    out = diag.summary()
    assert out["mean_consecutive_hold"] == 1.5
    assert out["median_consecutive_hold"] == 1.5
    assert out["max_hold_streak"] == 2
    assert out["mean_position_holding_duration"] == 2.5
    assert out["median_position_holding_duration"] == 2.5
    assert out["hold_executions"] == 0
    assert out["transitions_FLAT_LONG"] == out["transitions_FLAT_SHORT"] == 1
    assert out["transitions_LONG_HOLD"] == 2
    assert out["transitions_LONG_FLAT"] == 1
    assert sum(out[f"count_{a}"] for a in ACTION_NAMES) == 6


def test_clipped_objective_matches_hand_computed_categorical_probabilities(tmp_path):
    from tests.test_ppo_parallel_correctness import _trainer

    trainer = _trainer(tmp_path)
    for p in trainer.actor.parameters():
        p.data.zero_()
    obs = torch.zeros(4, trainer.obs_dim)
    mask = torch.ones(4, 10, dtype=torch.bool)
    mask[:, 1] = False
    action = torch.tensor([0, 3, 7, 9])
    old = torch.log(torch.tensor([[1 / 18], [2 / 9], [1 / 9], [1 / 9]]))
    advantage = torch.tensor([[1.0], [-1.0], [2.0], [-2.0]])
    loss, ratio, _, _, clip, entropy = trainer._minibatch_actor(
        obs, action, old, advantage, action_mask=mask, eps=0.2, ent_coef=0.01
    )
    torch.testing.assert_close(ratio, torch.tensor([[2.0], [0.5], [1.0], [1.0]]))
    # Positive advantage clips ratio 2 to 1.2; negative clips .5 to .8.
    expected = -(1.2 - 0.8 + 2 - 2) / 4 - 0.01 * np.log(9)
    assert loss.item() == pytest.approx(expected)
    assert clip == 0.5 and entropy.item() == pytest.approx(np.log(9))


def test_resume_restores_categorical_state_without_joining_fresh_episodes(tmp_path):
    from tests.test_ppo_parallel_correctness import _trainer

    trainer = _trainer(tmp_path)
    trainer._sync_policy_to_workers()
    transitions = trainer.collector.collect(66)
    trainer._ingest(transitions)
    trainer._consume_transitions(transitions)
    trainer._save_checkpoint("categorical")
    restored = _trainer(tmp_path / "restored")
    restored._restore_from_checkpoint(tmp_path / "checkpoints/categorical.pt")
    assert restored._env_steps == 66
    assert restored.action_diagnostics.summary()["policy_decisions"] == 66
    for key, value in trainer.actor.state_dict().items():
        torch.testing.assert_close(restored.actor.state_dict()[key], value)
    restored._sync_policy_to_workers()
    fresh = restored.collector.collect(16)
    restored._ingest(fresh)
    assert restored._episode_lengths[-1] == 16
    assert restored._episode_returns[-1] == pytest.approx(sum(t.reward for t in fresh))


def test_only_hold_mask_has_zero_entropy_and_invalid_mask_is_rejected():
    policy = CategoricalPolicy(3, ModelConfig(hidden_dim=8, num_layers=1))
    obs = torch.zeros(1, 3)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, 0] = True
    action, logp = policy.sample(obs, mask)
    assert action.item() == 0 and logp.item() == 0
    assert policy.dist(obs, mask).entropy().item() == 0
    mask[0, 0] = False
    with pytest.raises(ValueError, match="always allow HOLD"):
        policy.dist(obs, mask)


def test_stored_invalid_action_is_rejected():
    policy = CategoricalPolicy(3, ModelConfig(hidden_dim=8, num_layers=1))
    mask = torch.ones(1, 10, dtype=torch.bool)
    mask[0, 1] = False
    with pytest.raises(ValueError, match="invalid under its sampling mask"):
        policy.evaluate(torch.zeros(1, 3), torch.tensor([1]), mask)


def test_hold_reward_is_log_equity_change_after_price_move():
    prices = [1.10] * 5 + [1.11] * 5 + [1.12] * 20
    cfg = replace(
        _config(close_at_episode_end=False), sizing=PositionSizingConfig(mode="equity_fraction")
    )
    env = ForexEnvironment(_dataset(prices=prices), cfg)
    env.reset(start_index=0, horizon=4)
    obs, *_ = env.step(7)
    before = float(obs.account.equity)
    units = obs.account.position_units
    obs, reward, _, _, info = env.step(0)
    assert obs.account.position_units == units
    assert reward == pytest.approx(np.log(float(obs.account.equity) / before))
    assert info["trade_cost"] == info["units_delta"] == 0


def test_forced_episode_closure_is_separate_from_hold_execution():
    env = make_env()
    env.config = replace(env.config, close_at_episode_end=True)
    env.reset(start_index=0, horizon=2)
    env.step(7)
    obs, _, _, truncated, info = env.step(0)
    assert truncated and obs.account.position_units == 0
    assert info["action_diagnostics"]["actual_executions"] == 0
    assert info["action_diagnostics"]["forced_executions"] == 1
    assert info["action_diagnostics"]["forced_turnover"] > 0


def test_decimal_rounding_does_not_count_as_forced_execution():
    env = make_env()
    # Repeated fractional resizes can round the final Decimal digit when
    # portfolio units are computed as old_units + delta.
    for action in [7, 3, 6, 9, 4, 8, 5, 2]:
        _, _, _, _, info = env.step(action)
        assert not info["liquidation"]
        assert info["action_diagnostics"]["forced_executions"] == 0
        assert info["action_diagnostics"]["forced_turnover"] == 0


def test_update_history_survives_periodic_log_window_reset(tmp_path):
    from tests.test_ppo_parallel_correctness import _trainer

    trainer = _trainer(tmp_path)
    trainer._record_diagnostics({"entropy": 2.0})
    trainer._diag_history.clear()
    trainer._record_diagnostics({"entropy": 1.0})
    assert [d["entropy"] for d in trainer.update_history] == [2.0, 1.0]
