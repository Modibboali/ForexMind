"""Policy construction and action sampling (Phase 3).

A single :func:`build_policy_network` returns the right torch module for the
algorithm (SAC actor or PPO Gaussian policy).  :func:`sample_action` converts a
flat encoded observation to a target exposure in [-1, 1]; ``deterministic``
uses the policy mean (evaluation), otherwise it samples (exploration).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn

from forexmind.environment.actions import Action
from forexmind.observation.schema import EncodedObservation
from forexmind.training.config import ModelConfig
from forexmind.training.networks import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    GaussianPolicy,
    SquashedGaussianActor,
    TanhGaussianPolicy,
)


def build_policy_network(
    algorithm: str,
    obs_dim: int,
    action_dim: int,
    model: ModelConfig,
    *,
    log_std_min: float | None = None,
    log_std_max: float | None = None,
) -> nn.Module:
    if algorithm == "sac":
        return SquashedGaussianActor(obs_dim, action_dim, model)
    if algorithm == "ppo":
        return GaussianPolicy(
            obs_dim,
            action_dim,
            model,
            log_std_min=LOG_STD_MIN if log_std_min is None else log_std_min,
            log_std_max=LOG_STD_MAX if log_std_max is None else log_std_max,
        )
    raise ValueError(f"unsupported algorithm {algorithm!r}; use 'sac' or 'ppo'")


@torch.no_grad()
def sample_action(
    policy: nn.Module,
    obs_flat: np.ndarray,
    algorithm: str,
    *,
    deterministic: bool = False,
    device: str | torch.device = "cpu",
) -> float:
    """Sample (or deterministically select) a target exposure from a flat obs.

    For PPO: The tanh-squashed Gaussian policy naturally produces actions in (-1, 1).
    No clamping is required (tanh is strictly bounded).

    For SAC: The SquashedGaussianActor also uses tanh and produces actions in (-1, 1).

    Returns a float in (-1, 1).
    """
    obs = torch.as_tensor(np.asarray(obs_flat, dtype=np.float32), device=device).unsqueeze(0)
    if algorithm == "sac":
        sac_policy = cast(SquashedGaussianActor, policy)
        action = sac_policy.deterministic(obs) if deterministic else sac_policy.sample(obs)[0]
    elif algorithm == "ppo":
        ppo_policy = cast(TanhGaussianPolicy, policy)
        action = ppo_policy.act(obs, deterministic=deterministic)
    else:  # pragma: no cover - guarded in build_policy_network
        raise ValueError(f"unsupported algorithm {algorithm!r}")
    return float(action.item())


class PolicyAgent:
    """A Phase-2 :class:`TradingAgent` wrapper around a torch policy.

    Used for evaluation (validation/test) through the existing
    ``EvaluationRunner`` with deterministic action selection (policy mean).
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

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observation: EncodedObservation) -> Action:
        action = sample_action(
            self.policy,
            observation.encoded,
            self.algorithm,
            deterministic=True,
            device=self._device,
        )
        return Action(action)
