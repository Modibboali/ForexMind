"""MuZero core networks (Stage 4.1).

Implements the three standard MuZero functions without any of the surrounding
search/training machinery (no MCTS, replay, self-play, or loss):

* ``h_theta`` :class:`RepresentationNetwork`  ``o_t -> s_t^0``
* ``g_theta`` :class:`DynamicsNetwork`        ``(s_t^k, a_{t+k}) -> (r_hat, s_t^{k+1})``
* ``f_theta`` :class:`PredictionNetwork`      ``s_t^k -> (policy_logits, value)``

Design notes
------------
* The latent state is normalized (``LayerNorm``) and, by default, the dynamics
  predict a residual ``delta`` (``next = normalize(s + delta)``) for stability.
* Discrete actions are encoded with a learnable ``nn.Embedding``; the raw index
  is never concatenated directly onto the latent state.
* Prediction returns **raw logits** (no softmax) and a value head whose output
  width depends on :class:`MuZeroConfig` (support logits or a scalar).
* Dynamics predicts its *own* latent transition and reward head.  It never
  touches the environment and never predicts raw market observations.
"""

from __future__ import annotations

import torch
from torch import nn

from forexmind.muzero.config import MuZeroConfig

__all__ = [
    "DynamicsNetwork",
    "MLPStack",
    "PredictionNetwork",
    "RepresentationNetwork",
    "count_parameters",
    "parameter_report",
]


def _make_activation(name: str) -> type[nn.Module]:
    if name == "silu":
        return nn.SiLU
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "gelu":
        return nn.GELU
    if name == "elu":
        return nn.ELU
    raise ValueError(f"unsupported activation {name!r}")


class MLPStack(nn.Module):
    """``num_layers`` hidden blocks of ``Linear -> [LayerNorm] -> activation``.

    The final projection to the output dimension is *not* included here; it is
    added by each sub-network so the last hidden width is explicit.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        *,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        act = _make_activation(activation)
        blocks: list[nn.Module] = []
        dim = in_dim
        for _ in range(num_layers):
            blocks.append(nn.Linear(dim, hidden_dim))
            if layer_norm:
                blocks.append(nn.LayerNorm(hidden_dim))
            blocks.append(act())
            dim = hidden_dim
        self.net = nn.Sequential(*blocks)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RepresentationNetwork(nn.Module):
    """``h_theta(o_t) -> s_t^0`` — encodes a flat observation into latent state.

    A compact MLP is sufficient for Stage 4.1.  No transformer/recurrent/world
    model stack is used because the current Phase-2 observation is a flat
    vector, not a sequence.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        *,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.trunk = MLPStack(obs_dim, hidden_dim, num_layers, activation, layer_norm=layer_norm)
        self.proj = nn.Linear(self.trunk.out_dim, latent_dim)
        self.latent_norm: nn.Module = nn.LayerNorm(latent_dim) if layer_norm else nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"representation expects [B, {self.obs_dim}] observations, got {tuple(obs.shape)}"
            )
        return self.latent_norm(self.proj(self.trunk(obs)))


class DynamicsNetwork(nn.Module):
    """``g_theta(s, a) -> (next_latent, reward_head_output)``.

    The action is embedded, concatenated with the latent state, passed through a
    residual transition, and the same hidden features feed a separate reward
    head.  Rewards target the unchanged Forex reward
    ``log(equity[t+1] / equity[t])``; only the *representation* of that target
    (scalar vs categorical support) is configurable at the head.
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        action_embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        *,
        reward_output_dim: int,
        layer_norm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.action_embedding_dim = action_embedding_dim
        self.residual = residual
        self.action_embedding = nn.Embedding(num_actions, action_embedding_dim)
        self.trunk = MLPStack(
            latent_dim + action_embedding_dim,
            hidden_dim,
            num_layers,
            activation,
            layer_norm=layer_norm,
        )
        self.transition = nn.Linear(self.trunk.out_dim, latent_dim)
        self.reward_head = nn.Linear(self.trunk.out_dim, reward_output_dim)
        self.next_norm: nn.Module = nn.LayerNorm(latent_dim) if layer_norm else nn.Identity()

    # -- action encoding ------------------------------------------------------

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """Validate and embed a batch of discrete action indices ``[B]``.

        Raises a clear ``ValueError`` for non-integer input, wrong rank, or any
        index outside ``[0, num_actions)`` (e.g. ``-1``, ``10``, ``11``).
        """
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"action must be a torch.Tensor, got {type(action).__name__}")
        if action.ndim != 1:
            raise ValueError(f"action must be 1-D [B], got shape {tuple(action.shape)}")
        if action.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            if action.is_floating_point() and bool(torch.all(action == action.round())):
                action = action.long()
            else:
                raise ValueError(f"action must have an integer dtype, got {action.dtype}")
        if action.numel() > 0:
            lo = int(action.min())
            hi = int(action.max())
            if lo < 0 or hi >= self.num_actions:
                raise ValueError(
                    f"action index out of range [0, {self.num_actions}): min={lo}, max={hi}"
                )
        return self.action_embedding(action)

    # -- forward --------------------------------------------------------------

    def forward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"dynamics expects [B, {self.latent_dim}] latent states, got {tuple(latent.shape)}"
            )
        embedding = self.encode_action(action)
        if embedding.shape[0] != latent.shape[0]:
            raise ValueError(
                f"action batch {embedding.shape[0]} does not match latent batch {latent.shape[0]}"
            )
        features = self.trunk(torch.cat([latent, embedding], dim=-1))
        delta = self.transition(features)
        next_latent = self.next_norm(latent + delta) if self.residual else self.next_norm(delta)
        reward = self.reward_head(features)
        return next_latent, reward


class PredictionNetwork(nn.Module):
    """``f_theta(s) -> (policy_logits, value_head_output)``.

    Returns raw policy logits (no softmax) and the value head output; decoding
    the value head into a scalar is the inference layer's responsibility.
    """

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        *,
        value_output_dim: int,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.value_output_dim = value_output_dim
        self.policy_trunk = MLPStack(
            latent_dim, hidden_dim, num_layers, activation, layer_norm=layer_norm
        )
        self.policy_head = nn.Linear(self.policy_trunk.out_dim, num_actions)
        self.value_trunk = MLPStack(
            latent_dim, hidden_dim, num_layers, activation, layer_norm=layer_norm
        )
        self.value_head = nn.Linear(self.value_trunk.out_dim, value_output_dim)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"prediction expects [B, {self.latent_dim}] latents, got {tuple(latent.shape)}"
            )
        policy_logits = self.policy_head(self.policy_trunk(latent))
        value = self.value_head(self.value_trunk(latent))
        return policy_logits, value


def count_parameters(module: nn.Module) -> int:
    """Total number of trainable scalar parameters in ``module``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parameter_report(
    representation: nn.Module, dynamics: nn.Module, prediction: nn.Module
) -> dict[str, int]:
    """Return a parameter-count breakdown for the three MuZero sub-networks."""
    r = count_parameters(representation)
    d = count_parameters(dynamics)
    p = count_parameters(prediction)
    return {"representation": r, "dynamics": d, "prediction": p, "total": r + d + p}


def build_subnetworks(
    config: MuZeroConfig,
) -> tuple[RepresentationNetwork, DynamicsNetwork, PredictionNetwork]:
    """Construct the three sub-networks from a :class:`MuZeroConfig`."""
    representation = RepresentationNetwork(
        obs_dim=config.obs_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        activation=config.activation,
        layer_norm=config.layer_norm,
    )
    dynamics = DynamicsNetwork(
        latent_dim=config.latent_dim,
        num_actions=config.num_actions,
        action_embedding_dim=config.action_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        activation=config.activation,
        reward_output_dim=config.reward_output_dim,
        layer_norm=config.layer_norm,
        residual=config.residual_dynamics,
    )
    prediction = PredictionNetwork(
        latent_dim=config.latent_dim,
        num_actions=config.num_actions,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        activation=config.activation,
        value_output_dim=config.value_output_dim,
        layer_norm=config.layer_norm,
    )
    return representation, dynamics, prediction
