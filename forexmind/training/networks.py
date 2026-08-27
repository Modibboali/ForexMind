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


class GaussianPolicy(nn.Module):
    """PPO Gaussian policy (mean + learned log-std), continuous actions.

    The action is a clamped Gaussian sample (clamped to [-1, 1] at execution).
    ``log_std`` is a single learned parameter bounded to ``[log_std_min,
    log_std_max]`` so ``exp(log_std)`` cannot silently overflow.
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
        return self.dist(obs)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_net(obs)
        log_std = self.log_std.expand_as(mean).clamp(self.log_std_min, self.log_std_max)
        return torch.distributions.Normal(mean, log_std.exp())

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.dist(obs)
        action = dist.mean if deterministic else dist.sample()
        return torch.clamp(action, -1.0, 1.0)

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Plain Gaussian density on the (clamped) action.  No tanh-squash
        # correction: the action is clamped, not tanh-squashed, so applying a
        # tanh Jacobian correction here would be inconsistent with the density
        # used during the PPO update.
        dist = self.dist(obs)
        raw_action = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
        log_prob = dist.log_prob(raw_action)
        return log_prob.sum(-1, keepdim=True), dist.entropy().sum(-1, keepdim=True)


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
