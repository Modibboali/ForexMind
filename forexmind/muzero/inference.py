"""MuZero inference API (Stage 4.1).

:class:`MuZeroNetwork` composes the representation/dynamics/prediction
sub-networks and exposes the two public entry points that MCTS (Stage 4.2) will
call::

    root = model.initial_inference(observation, action_mask)
    child = model.recurrent_inference(root.latent_state, action, next_mask)

Responsibilities of this layer (and only this layer):

* normalizing observation/action/latent inputs into ``[B, ...]`` tensors,
* decoding support logits into scalars,
* applying the optional causal ``action_mask`` to policy logits.

The core networks never apply the mask themselves, so training can still read
raw logits by passing ``action_mask=None``.

``recurrent_inference`` is a *pure neural* transition: it never calls
``env.step()`` and never touches the environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from forexmind.muzero.config import MuZeroConfig
from forexmind.muzero.networks import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
    build_subnetworks,
    parameter_report,
)
from forexmind.muzero.support import support_to_scalar
from forexmind.muzero.types import NetworkOutput

__all__ = [
    "MuZeroNetwork",
    "apply_action_mask",
    "build_muzero_network",
    "decode_scalar",
    "masked_policy_probs",
]


def decode_scalar(
    raw: torch.Tensor,
    *,
    use_support: bool,
    support_size: int,
    scale: float,
    epsilon: float,
) -> torch.Tensor:
    """Decode a reward/value head output into economic units ``[..., 1]``.

    Shared by inference and by the Stage 4.4 learner so training-time and
    inference-time predictions are decoded identically.
    """
    if use_support:
        scalar = support_to_scalar(raw, support_size, scale=scale, epsilon=epsilon)
        return scalar.unsqueeze(-1)
    if raw.shape[-1] != 1:
        raise ValueError(f"scalar head must output 1 value, got {raw.shape[-1]}")
    return raw * scale


def apply_action_mask(
    policy_logits: torch.Tensor,
    action_mask: torch.Tensor | np.ndarray,
    *,
    hold_index: int = 0,
    validate: bool = True,
) -> torch.Tensor:
    """Return policy logits with invalid actions set to a large negative value.

    The mask is applied here (the search/inference-facing layer), never inside
    the prediction network.  Requirements enforced when ``validate`` is true:

    * the mask matches the logits' ``num_actions`` (a 1-D mask is broadcast),
    * ``HOLD`` (``hold_index=0``) stays valid,
    * every batch row keeps at least one valid action.
    """
    mask = action_mask if isinstance(action_mask, torch.Tensor) else torch.as_tensor(action_mask)
    mask = mask.to(device=policy_logits.device, dtype=torch.bool)
    if mask.ndim == 1 and mask.shape[0] == policy_logits.shape[-1]:
        mask = mask.unsqueeze(0).expand_as(policy_logits)
    if mask.shape != policy_logits.shape:
        raise ValueError(
            f"action_mask must have shape {tuple(policy_logits.shape)}, got {tuple(mask.shape)}"
        )
    if validate:
        if mask.numel() == 0:
            raise ValueError("action_mask must not be empty")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("action_mask must leave at least one valid action per row")
        if hold_index is not None and not bool(mask[..., hold_index].all()):
            raise ValueError("action_mask must always keep HOLD (index 0) valid")
    return policy_logits.masked_fill(~mask, torch.finfo(policy_logits.dtype).min)


def masked_policy_probs(
    policy_logits: torch.Tensor, action_mask: torch.Tensor | np.ndarray
) -> torch.Tensor:
    """Softmax of masked policy logits (convenience for callers/tests)."""
    return torch.softmax(apply_action_mask(policy_logits, action_mask), dim=-1)


class MuZeroNetwork(nn.Module):
    """Composable MuZero core network with a structured inference API."""

    def __init__(self, config: MuZeroConfig) -> None:
        super().__init__()
        self.config = config
        representation, dynamics, prediction = build_subnetworks(config)
        self.representation: RepresentationNetwork = representation
        self.dynamics: DynamicsNetwork = dynamics
        self.prediction: PredictionNetwork = prediction

    # -- helpers --------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _as_observation_batch(self, observation: Any) -> torch.Tensor:
        obs = observation if isinstance(observation, torch.Tensor) else torch.as_tensor(observation)
        obs = obs.to(device=self.device, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2:
            raise ValueError(
                f"observation must be [obs_dim] or [B, obs_dim], got {tuple(obs.shape)}"
            )
        if obs.shape[-1] != self.config.obs_dim:
            raise ValueError(
                f"observation last dim must be {self.config.obs_dim}, got {obs.shape[-1]}"
            )
        return obs

    def _as_latent_batch(self, latent_state: Any) -> torch.Tensor:
        latent = (
            latent_state
            if isinstance(latent_state, torch.Tensor)
            else torch.as_tensor(latent_state)
        )
        latent = latent.to(device=self.device, dtype=torch.float32)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.ndim != 2 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError(
                f"latent_state must be [B, {self.config.latent_dim}], got {tuple(latent.shape)}"
            )
        return latent

    def _as_action_batch(self, action: Any, batch_size: int) -> torch.Tensor:
        if isinstance(action, bool):
            raise TypeError("action must be a discrete index, not a bool")
        if isinstance(action, (int, np.integer)):
            return torch.full((batch_size,), int(action), dtype=torch.long, device=self.device)
        tensor = action if isinstance(action, torch.Tensor) else torch.as_tensor(action)
        tensor = tensor.to(device=self.device)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1).expand(batch_size)
        elif tensor.ndim == 1 and tensor.shape[0] == 1:
            tensor = tensor.expand(batch_size)
        elif tensor.ndim != 1 or tensor.shape[0] != batch_size:
            raise ValueError(
                f"action must be a scalar or shape [{batch_size}], got {tuple(tensor.shape)}"
            )
        if tensor.is_floating_point():
            if not bool(torch.all(tensor == tensor.round())):
                raise ValueError("action contains non-integral float values")
            tensor = tensor.round()
        return tensor.to(dtype=torch.long)

    def _decode(
        self,
        raw: torch.Tensor,
        *,
        support_size: int,
        scale: float,
        epsilon: float,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Turn a reward/value head output into ``(scalar [B, 1], logits or None)``."""
        if self.config.use_support:
            scalar = support_to_scalar(raw, support_size, scale=scale, epsilon=epsilon)
            return scalar.unsqueeze(-1), raw
        if raw.shape[-1] != 1:
            raise ValueError(f"{name} scalar head must output 1 value, got {raw.shape[-1]}")
        return raw * scale, None

    # -- public inference API -------------------------------------------------

    def initial_inference(
        self, observation: Any, action_mask: torch.Tensor | np.ndarray | None = None
    ) -> NetworkOutput:
        """``observation -> h_theta -> f_theta -> (policy_logits, value, reward=0)``.

        Returns a :class:`NetworkOutput` whose ``reward`` is exactly zero (no
        dynamics transition has occurred) and whose ``reward_logits`` is ``None``.
        """
        obs = self._as_observation_batch(observation)
        latent = self.representation(obs)
        policy_logits, value_raw = self.prediction(latent)
        value, value_logits = self._decode(
            value_raw,
            support_size=self.config.value_support_size,
            scale=self.config.value_scale,
            epsilon=self.config.value_epsilon,
            name="value",
        )
        if action_mask is not None:
            policy_logits = apply_action_mask(policy_logits, action_mask)
        reward = torch.zeros((latent.shape[0], 1), dtype=latent.dtype, device=latent.device)
        return NetworkOutput(
            latent_state=latent,
            policy_logits=policy_logits,
            value=value,
            reward=reward,
            value_logits=value_logits,
            reward_logits=None,
        )

    def recurrent_inference(
        self,
        latent_state: Any,
        action: Any,
        action_mask: torch.Tensor | np.ndarray | None = None,
    ) -> NetworkOutput:
        """``latent_state + action -> g_theta -> f_theta -> NetworkOutput``.

        ``action`` is the discrete action *index* in ``[0, num_actions)``, not a
        target exposure.  The returned ``latent_state`` is the **next** latent
        state so callers can chain calls.  This is a pure neural transition and
        never calls ``env.step()``.
        """
        latent = self._as_latent_batch(latent_state)
        actions = self._as_action_batch(action, latent.shape[0])
        next_latent, reward_raw = self.dynamics(latent, actions)
        policy_logits, value_raw = self.prediction(next_latent)
        value, value_logits = self._decode(
            value_raw,
            support_size=self.config.value_support_size,
            scale=self.config.value_scale,
            epsilon=self.config.value_epsilon,
            name="value",
        )
        reward, reward_logits = self._decode(
            reward_raw,
            support_size=self.config.reward_support_size,
            scale=self.config.reward_scale,
            epsilon=self.config.reward_epsilon,
            name="reward",
        )
        if action_mask is not None:
            policy_logits = apply_action_mask(policy_logits, action_mask)
        return NetworkOutput(
            latent_state=next_latent,
            policy_logits=policy_logits,
            value=value,
            reward=reward,
            value_logits=value_logits,
            reward_logits=reward_logits,
        )

    # -- reporting ------------------------------------------------------------

    def parameter_report(self) -> dict[str, int]:
        """Parameter counts for representation / dynamics / prediction / total."""
        return parameter_report(self.representation, self.dynamics, self.prediction)


def build_muzero_network(config: MuZeroConfig) -> MuZeroNetwork:
    """Construct a :class:`MuZeroNetwork` from a :class:`MuZeroConfig`."""
    return MuZeroNetwork(config)
