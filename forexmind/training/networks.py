"""Neural network definitions for model-free RL (Phase 3).

The model interface takes the flat Phase-2 encoded observation ``(obs_dim,)``
(``MLP(351)`` baseline) so a later experiment can swap in a structured
temporal encoder without changing the environment.  The networks are kept
modest (``hidden_dim=256``, ``num_layers=2`` by default).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from forexmind.training.config import ModelConfig

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _activation(name: str) -> type[nn.Module]:
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"unsupported activation {name!r}")


class MLP(nn.Module):
    """Simple feed-forward MLP."""

    def __init__(self, in_dim: int, out_dim: int, config: ModelConfig) -> None:
        super().__init__()
        act = _activation(config.activation)
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(config.num_layers):
            layers.append(nn.Linear(dim, config.hidden_dim))
            layers.append(act())
            dim = config.hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquashedGaussianActor(nn.Module):
    """SAC actor: Gaussian policy squashed to [-1, 1] via tanh.

    ``deterministic()`` returns the mean action (used for evaluation); the
    exploration action is ``sample()`` with a log-probability.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.mean_net = MLP(obs_dim, action_dim, config)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_net(obs)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean, log_std = self.forward(obs)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(squashed_action, log_prob_of_action)``."""
        dist = self._dist(obs)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self._dist(obs)
        return dist.entropy().sum(-1, keepdim=True)


class TwinQCritic(nn.Module):
    """Twin Q critics (clipped double Q)."""

    def __init__(self, obs_dim: int, action_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.q1 = MLP(obs_dim + action_dim, 1, config)
        self.q2 = MLP(obs_dim + action_dim, 1, config)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class TanhGaussianPolicy(nn.Module):
    """PPO tanh-squashed Gaussian policy (mean + learned log-std), continuous actions.

    **Mathematically correct bounded continuous policy:**

    1. Sample raw: ``u ~ N(mean, stddev)`` from ``policy_net(obs)``
    2. Transform: ``a = tanh(u)`` where ``a ∈ (-1, 1)`` (strictly bounded)
    3. Log-prob with Jacobian: ``log π(a|s) = log N(u) - Σ log(1 - tanh²(u) + ε)``

    The environment receives ``a`` directly (no clamping to boundary).

    **Deterministic action:** ``a = tanh(μ)`` (mean of the raw Gaussian)

    **Why tanh-squashing instead of clamping:**
    - Gaussian + clamp loses information (multiple raw samples → same boundary action)
    - Tanh is differentiable everywhere, smooth log-prob gradient
    - Jacobian correction ensures log-prob matches actual action distribution
    - Numerically stable with ε in log(1 - tanh²(u) + ε)

    **Storage requirements:**
    - Store raw sample ``u`` for replay/PPO updates (can reconstruct log-prob)
    - Store transformed action ``a`` for environment (trading semantics)
    - Store log-prob for efficiency (avoid recomputation during PPO update)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: ModelConfig,
        *,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.mean_net = MLP(obs_dim, action_dim, config)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Return the base Normal distribution over the raw (pre-tanh) action."""
        return self.dist(obs)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """Build the Normal distribution for raw actions."""
        mean = self.mean_net(obs)
        log_std = self.log_std.expand_as(mean).clamp(self.log_std_min, self.log_std_max)
        return torch.distributions.Normal(mean, log_std.exp())

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action (deterministic or stochastic).

        Returns the transformed action a = tanh(u), ready for the environment.
        Clamping is NOT applied; tanh naturally bounds to (-1, 1).
        """
        dist = self.dist(obs)
        raw = dist.mean if deterministic else dist.rsample()
        return torch.tanh(raw)

    def log_prob_and_raw(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample raw action and compute log-probability with Jacobian correction.

        Returns:
            (action, log_prob, raw_action) where:
            - action: tanh-transformed action ∈ (-1, 1), ready for environment
            - log_prob: log π(a|s) = log N(u) - Σ log(1 - tanh²(u) + ε)
            - raw_action: sample from N(mean, stddev), stored for replay/PPO updates
        """
        dist = self.dist(obs)
        raw = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(raw)

        # Log-probability with Jacobian correction for tanh transformation.
        # log π(a) = log N(u) - log |∂a/∂u|
        # where |∂a/∂u| = 1 - tanh²(u)
        log_prob_raw = dist.log_prob(raw)
        log_det_jacobian = torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob_raw - log_det_jacobian
        return action, log_prob.sum(-1, keepdim=True), raw

    def evaluate(
        self, obs: torch.Tensor, raw_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-prob at a given raw action (for PPO updates).

        During PPO training, we must evaluate log π(a|s) using the raw action ``u``
        that was originally sampled, not the transformed action ``a``.  This is
        critical for the importance ratio to be defined over the actual sampling
        distribution.

        Args:
            obs: Observation batch (batch_size, obs_dim)
            raw_action: Raw pre-tanh action (batch_size, action_dim)

        Returns:
            (log_prob, entropy) where log_prob includes Jacobian correction
        """
        dist = self.dist(obs)

        # Compute log-prob of the raw action under the base Gaussian
        log_prob_raw = dist.log_prob(raw_action)

        # Apply tanh Jacobian correction using the transformed action
        action = torch.tanh(raw_action)
        log_det_jacobian = torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob_raw - log_det_jacobian

        return log_prob.sum(-1, keepdim=True), dist.entropy().sum(-1, keepdim=True)


# For backward compatibility: GaussianPolicy is now TanhGaussianPolicy
GaussianPolicy = TanhGaussianPolicy


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.net = MLP(obs_dim, 1, config)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-averaged target update."""
    for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@dataclass(frozen=True)
class SACNetworks:
    """Bundle of SAC networks."""

    actor: SquashedGaussianActor
    critic: TwinQCritic
    target_critic: TwinQCritic

    def to_device(self, device: torch.device) -> None:
        self.actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)


def build_sac_networks(obs_dim: int, action_dim: int, model: ModelConfig) -> SACNetworks:
    actor = SquashedGaussianActor(obs_dim, action_dim, model)
    critic = TwinQCritic(obs_dim, action_dim, model)
    target_critic = TwinQCritic(obs_dim, action_dim, model)
    target_critic.load_state_dict(critic.state_dict())
    return SACNetworks(actor=actor, critic=critic, target_critic=target_critic)
