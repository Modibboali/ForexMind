"""MuZero core networks and inference API (Stage 4.1).

This package deliberately contains **only** the neural architecture and the
inference contracts: representation ``h_theta``, dynamics ``g_theta``, and
prediction ``f_theta``.  It does not implement MCTS, replay, self-play, the
MuZero loss, or any training loop — those belong to later stages.

Quick start::

    from forexmind.muzero import MuZeroConfig, build_muzero_network

    config = MuZeroConfig.from_encoder_config()  # obs_dim derived from the encoder
    model = build_muzero_network(config)

    root = model.initial_inference(observation, action_mask)
    child = model.recurrent_inference(root.latent_state, action, next_mask)
"""

from __future__ import annotations

from forexmind.muzero.config import MuZeroConfig, observation_dim
from forexmind.muzero.inference import (
    MuZeroNetwork,
    apply_action_mask,
    build_muzero_network,
    masked_policy_probs,
)
from forexmind.muzero.networks import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
    count_parameters,
    parameter_report,
)
from forexmind.muzero.support import scalar_to_support, support_to_scalar
from forexmind.muzero.types import NetworkOutput

__all__ = [
    "DynamicsNetwork",
    "MuZeroConfig",
    "MuZeroNetwork",
    "NetworkOutput",
    "PredictionNetwork",
    "RepresentationNetwork",
    "apply_action_mask",
    "build_muzero_network",
    "count_parameters",
    "masked_policy_probs",
    "observation_dim",
    "parameter_report",
    "scalar_to_support",
    "support_to_scalar",
]
