"""Unit tests for tanh-squashed Gaussian action distribution (Stage 3.2).

Tests verify:
1. Mathematical correctness of tanh-squashed Gaussian distribution
2. Jacobian correction for log-probability
3. PPO importance ratio consistency
4. Numerical stability (no NaN/Inf)
5. Old/new log-prob equivalence when policy unchanged
6. Action bounds and deterministic action = tanh(mu)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from forexmind.training.config import ModelConfig
from forexmind.training.networks import TanhGaussianPolicy, GaussianPolicy


class TestTanhGaussianPolicyBasics:
    """Test basic properties of TanhGaussianPolicy."""

    @pytest.fixture
    def policy(self) -> TanhGaussianPolicy:
        """Create a small policy for testing."""
        config = ModelConfig(
            hidden_dim=64,
            num_layers=1,
            activation="relu",
        )
        return TanhGaussianPolicy(
            obs_dim=10,
            action_dim=1,
            config=config,
            log_std_min=-5.0,
            log_std_max=2.0,
        )

    def test_tanh_action_bounded_deterministic(self, policy: TanhGaussianPolicy) -> None:
        """Deterministic action = tanh(mu) must be in (-1, 1)."""
        obs = torch.randn(8, 10)
        action = policy.act(obs, deterministic=True)
        
        assert action.shape == (8, 1)
        assert torch.all(action > -1.0)
        assert torch.all(action < 1.0)
        
        # Verify it's exactly tanh(mu)
        dist = policy.dist(obs)
        expected = torch.tanh(dist.mean)
        assert torch.allclose(action, expected)

    def test_tanh_action_bounded_stochastic(self, policy: TanhGaussianPolicy) -> None:
        """Stochastic samples must be in (-1, 1)."""
        obs = torch.randn(16, 10)
        torch.manual_seed(42)
        
        for _ in range(10):
            action = policy.act(obs, deterministic=False)
            assert action.shape == (16, 1)
            assert torch.all(action > -1.0), "Action should be > -1"
            assert torch.all(action < 1.0), "Action should be < 1"
            # Extremely unlikely to be exactly 0
            assert not torch.all(action == 0.0)

    def test_log_prob_and_raw_consistency(self, policy: TanhGaussianPolicy) -> None:
        """log_prob_and_raw() should produce consistent action and log_prob."""
        obs = torch.randn(4, 10)
        torch.manual_seed(42)
        
        action, log_prob, raw = policy.log_prob_and_raw(obs, deterministic=False)
        
        # Action must be tanh(raw)
        expected_action = torch.tanh(raw)
        assert torch.allclose(action, expected_action)
        
        # Action must be bounded
        assert torch.all(action > -1.0)
        assert torch.all(action < 1.0)
        
        # Log-prob must be finite
        assert torch.all(torch.isfinite(log_prob))
        assert log_prob.shape == (4, 1)

    def test_log_prob_finite_near_boundaries(self, policy: TanhGaussianPolicy) -> None:
        """Log-prob must remain finite even for raw actions near tanh boundaries."""
        obs = torch.randn(4, 10)
        dist = policy.dist(obs)
        
        # Extreme raw actions (before tanh)
        extreme_raw = torch.tensor([[5.0], [10.0], [-5.0], [-10.0]])
        log_prob, entropy = policy.evaluate(obs, extreme_raw)
        
        assert torch.all(torch.isfinite(log_prob)), "Log-prob must be finite for extreme raw actions"
        assert torch.all(torch.isfinite(entropy)), "Entropy must be finite"

    def test_deterministic_vs_stochastic_action_mean(self, policy: TanhGaussianPolicy) -> None:
        """Stochastic actions should have mean ≈ deterministic action (large sample)."""
        obs = torch.randn(1, 10)
        policy.eval()
        
        with torch.no_grad():
            det_action = policy.act(obs, deterministic=True)
            
            # Sample many times
            torch.manual_seed(42)
            stochastic_actions = []
            for _ in range(100):
                s_action = policy.act(obs, deterministic=False)
                stochastic_actions.append(s_action.item())
            
            mean_stochastic = np.mean(stochastic_actions)
            det_value = float(det_action.item())
            
            # Mean should be reasonably close to deterministic
            assert abs(mean_stochastic - det_value) < 0.2


class TestPPOImportanceRatioCorrectness:
    """Test PPO importance ratio and log-prob equivalence."""

    @pytest.fixture
    def policy(self) -> TanhGaussianPolicy:
        config = ModelConfig(hidden_dim=64, num_layers=1, activation="relu")
        return TanhGaussianPolicy(obs_dim=10, action_dim=1, config=config)

    def test_old_and_new_logprob_identical_when_policy_unchanged(
        self, policy: TanhGaussianPolicy
    ) -> None:
        """When policy params are unchanged, old and new log-probs must match exactly."""
        obs = torch.randn(8, 10)
        
        # Sample actions
        torch.manual_seed(42)
        _, old_log_prob, raw_action = policy.log_prob_and_raw(obs, deterministic=False)
        
        # Policy unchanged, re-evaluate
        with torch.no_grad():
            new_log_prob, _ = policy.evaluate(obs, raw_action)
        
        assert torch.allclose(old_log_prob, new_log_prob, atol=1e-6), \
            "Old and new log-probs must match when policy unchanged"

    def test_importance_ratio_approximately_one_when_policies_same(
        self, policy: TanhGaussianPolicy
    ) -> None:
        """PPO ratio exp(new_log_prob - old_log_prob) should be ≈ 1.0 if policies identical."""
        obs = torch.randn(8, 10)
        
        torch.manual_seed(42)
        _, old_log_prob, raw_action = policy.log_prob_and_raw(obs, deterministic=False)
        
        with torch.no_grad():
            new_log_prob, _ = policy.evaluate(obs, raw_action)
        
        ratio = torch.exp(new_log_prob - old_log_prob)
        expected = torch.ones_like(ratio)
        
        assert torch.allclose(ratio, expected, atol=1e-5), \
            "Ratio should be 1.0 when policies are identical"

    def test_importance_ratio_changes_after_gradient_update(
        self, policy: TanhGaussianPolicy
    ) -> None:
        """Importance ratio should differ after a gradient update."""
        obs = torch.randn(8, 10)
        
        torch.manual_seed(42)
        _, old_log_prob, raw_action = policy.log_prob_and_raw(obs, deterministic=False)
        old_log_prob = old_log_prob.detach()
        
        # Apply a single gradient step (minimal update)
        opt = torch.optim.Adam(policy.parameters(), lr=0.1)
        new_log_prob, entropy = policy.evaluate(obs, raw_action)
        loss = -new_log_prob.mean() - 0.01 * entropy.mean()
        loss.backward()
        opt.step()
        
        # Re-evaluate after update
        with torch.no_grad():
            updated_log_prob, _ = policy.evaluate(obs, raw_action)
        
        ratio = torch.exp(updated_log_prob - old_log_prob)
        
        # Ratio should have changed (at least one value differs from 1.0)
        assert not torch.allclose(ratio, torch.ones_like(ratio), atol=0.05), \
            "Ratio should change after gradient update"


class TestNumericalStability:
    """Test numerical stability of the implementation."""

    @pytest.fixture
    def policy(self) -> TanhGaussianPolicy:
        config = ModelConfig(hidden_dim=64, num_layers=1, activation="relu")
        return TanhGaussianPolicy(obs_dim=10, action_dim=1, config=config)

    def test_no_nan_inf_forward_pass(self, policy: TanhGaussianPolicy) -> None:
        """Forward pass should never produce NaN/Inf."""
        obs = torch.randn(32, 10)
        
        with torch.no_grad():
            action = policy.act(obs, deterministic=False)
            assert torch.all(torch.isfinite(action))
            
            _, log_prob, raw = policy.log_prob_and_raw(obs, deterministic=False)
            assert torch.all(torch.isfinite(log_prob))
            assert torch.all(torch.isfinite(raw))

    def test_no_nan_inf_backward_pass(self, policy: TanhGaussianPolicy) -> None:
        """Backward pass should never produce NaN/Inf in gradients."""
        obs = torch.randn(16, 10)
        adv = torch.randn(16, 1)
        
        _, old_log_prob, raw_action = policy.log_prob_and_raw(obs, deterministic=False)
        new_log_prob, entropy = policy.evaluate(obs, raw_action)
        
        ratio = torch.exp(new_log_prob - old_log_prob)
        loss = -(ratio * adv).mean() - 0.01 * entropy.mean()
        loss.backward()
        
        # Check all gradients are finite
        for param in policy.parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad))

    def test_log_std_clamping(self, policy: TanhGaussianPolicy) -> None:
        """log_std parameter must stay within bounds."""
        obs = torch.randn(8, 10)
        
        # Do many forward passes
        for _ in range(10):
            _ = policy.act(obs, deterministic=False)
        
        log_std = policy.log_std
        assert torch.all(log_std >= policy.log_std_min)
        assert torch.all(log_std <= policy.log_std_max)

    def test_extreme_log_std_values(self, policy: TanhGaussianPolicy) -> None:
        """Extreme log_std values should not cause NaN/Inf."""
        obs = torch.randn(4, 10)
        
        # Manually set extreme log_std
        with torch.no_grad():
            policy.log_std.fill_(policy.log_std_max)  # Very high std
        
        action, log_prob, raw = policy.log_prob_and_raw(obs)
        assert torch.all(torch.isfinite(action))
        assert torch.all(torch.isfinite(log_prob))
        
        with torch.no_grad():
            policy.log_std.fill_(policy.log_std_min)  # Very low std
        
        action, log_prob, raw = policy.log_prob_and_raw(obs)
        assert torch.all(torch.isfinite(action))
        assert torch.all(torch.isfinite(log_prob))


class TestActionTransformationDeterminism:
    """Test that action transformation is deterministic under fixed seed."""

    @pytest.fixture
    def policy(self) -> TanhGaussianPolicy:
        config = ModelConfig(hidden_dim=64, num_layers=1, activation="relu")
        p = TanhGaussianPolicy(obs_dim=10, action_dim=1, config=config)
        p.eval()
        return p

    def test_deterministic_action_reproducibility(self, policy: TanhGaussianPolicy) -> None:
        """Deterministic action should be reproducible."""
        obs = torch.randn(4, 10)
        
        action1 = policy.act(obs, deterministic=True)
        action2 = policy.act(obs, deterministic=True)
        
        assert torch.allclose(action1, action2)

    def test_stochastic_action_seed_reproducibility(self, policy: TanhGaussianPolicy) -> None:
        """Stochastic action should be reproducible with same seed."""
        obs = torch.randn(4, 10)
        
        torch.manual_seed(42)
        action1 = policy.act(obs, deterministic=False)
        
        torch.manual_seed(42)
        action2 = policy.act(obs, deterministic=False)
        
        assert torch.allclose(action1, action2)


class TestBackwardCompatibility:
    """Test that GaussianPolicy alias works."""

    def test_gaussian_policy_alias_exists(self) -> None:
        """GaussianPolicy should be an alias for TanhGaussianPolicy."""
        assert GaussianPolicy is TanhGaussianPolicy

    def test_gaussian_policy_import_works(self) -> None:
        """GaussianPolicy should be importable."""
        from forexmind.training.networks import GaussianPolicy as GP
        assert GP is TanhGaussianPolicy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
