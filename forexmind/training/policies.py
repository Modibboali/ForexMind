"""Categorical masked PPO and continuous SAC policy construction."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn

from forexmind.environment.actions import Action
from forexmind.observation.schema import EncodedObservation
from forexmind.training.config import ModelConfig
from forexmind.training.networks import (
    CategoricalPolicy,
    SquashedGaussianActor,
)


def build_policy_network(
    algorithm: str,
    obs_dim: int,
    action_dim: int,
    model: ModelConfig,
) -> nn.Module:
    if algorithm == "sac":
        return SquashedGaussianActor(obs_dim, action_dim, model)
    if algorithm == "ppo":
        return CategoricalPolicy(obs_dim, model)
    raise ValueError(f"unsupported algorithm {algorithm!r}; use 'sac' or 'ppo'")


@torch.no_grad()
def sample_action(
    policy: nn.Module,
    obs_flat: np.ndarray,
    algorithm: str,
    *,
    deterministic: bool = False,
    device: str | torch.device = "cpu",
    action_mask: np.ndarray | None = None,
) -> int | float:
    """Return a masked PPO integer index or a continuous SAC exposure."""
    obs = torch.as_tensor(np.asarray(obs_flat, dtype=np.float32), device=device).unsqueeze(0)
    if algorithm == "sac":
        sac_policy = cast(SquashedGaussianActor, policy)
        action = sac_policy.deterministic(obs) if deterministic else sac_policy.sample(obs)[0]
    elif algorithm == "ppo":
        if action_mask is None:
            raise ValueError("PPO requires the current environment action mask")
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
        ppo_policy = cast(CategoricalPolicy, policy)
        action = ppo_policy.act(obs, mask, deterministic=deterministic)
        return int(action.item())
    else:  # pragma: no cover - guarded in build_policy_network
        raise ValueError(f"unsupported algorithm {algorithm!r}")
    return float(action.item())


class PolicyAgent:
    """A Phase-2 :class:`TradingAgent` wrapper around a torch policy.

    Used for evaluation (validation/test) through the existing
    ``EvaluationRunner`` with deterministic action selection (masked argmax for PPO).
    """

    def __init__(
        self,
        policy: nn.Module,
        algorithm: str,
        *,
        name: str | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.policy = policy
        self.algorithm = algorithm
        self.name = name or algorithm
        self._device = device
        self.policy.eval()
        self.action_mask: np.ndarray | None = None
        self.last_action_index: int | None = None

    def reset(self, seed: int | None = None) -> None:
        self.action_mask = None
        self.last_action_index = None

    def set_action_mask(self, mask: np.ndarray) -> None:
        self.action_mask = mask.copy()

    def act(self, observation: EncodedObservation) -> Action:
        action = sample_action(
            self.policy,
            observation.encoded,
            self.algorithm,
            deterministic=True,
            device=self._device,
            action_mask=self.action_mask,
        )
        from forexmind.environment.actions import resolve_action

        self.last_action_index = int(action) if self.algorithm == "ppo" else None
        return resolve_action(action)
