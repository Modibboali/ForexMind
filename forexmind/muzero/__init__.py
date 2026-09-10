"""MuZero for ForexMind (Stages 4.1-4.2).

Stage 4.1 added the neural core and inference contracts (``h_theta``,
``g_theta``, ``f_theta``).  Stage 4.2 adds the MCTS/PUCT search layer that plans
in the learned latent space over the frozen six-action space.

Still **not** implemented: MuZero loss, replay buffer, self-play, reanalysis,
distributed actors, or Stochastic MuZero.

Quick start::

    from forexmind.muzero import MuZeroConfig, build_muzero_network, MuZeroMCTS
    from forexmind.muzero import SearchConfig

    config = MuZeroConfig.from_encoder_config()  # obs_dim derived from the encoder
    model = build_muzero_network(config)
    search = MuZeroMCTS(model, SearchConfig(num_simulations=50).evaluation())

    result = search.search_from_env(env, observation)
    result.action, result.policy, result.visit_counts
"""

from __future__ import annotations

from forexmind.muzero.actions import (
    MUZERO_ACTION_NAMES,
    MUZERO_ENV_ACTION_INDICES,
    MUZERO_NUM_ACTIONS,
    MUZERO_TARGET_EXPOSURES,
    PlanningState,
    env_action_index,
    mu_zero_action_index,
    project_action_mask,
)
from forexmind.muzero.config import MuZeroConfig, SearchConfig, observation_dim
from forexmind.muzero.inference import (
    MuZeroNetwork,
    apply_action_mask,
    build_muzero_network,
    masked_policy_probs,
)
from forexmind.muzero.minmax import MinMaxStats
from forexmind.muzero.networks import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
    count_parameters,
    parameter_report,
)
from forexmind.muzero.node import Node
from forexmind.muzero.search import (
    MuZeroMCTS,
    SearchDiagnostics,
    SearchResult,
    discounted_backup,
    visit_count_policy,
)
from forexmind.muzero.support import scalar_to_support, support_to_scalar
from forexmind.muzero.types import NetworkOutput

__all__ = [
    "MUZERO_ACTION_NAMES",
    "MUZERO_ENV_ACTION_INDICES",
    "MUZERO_NUM_ACTIONS",
    "MUZERO_TARGET_EXPOSURES",
    "DynamicsNetwork",
    "MinMaxStats",
    "MuZeroConfig",
    "MuZeroMCTS",
    "MuZeroNetwork",
    "NetworkOutput",
    "Node",
    "PlanningState",
    "PredictionNetwork",
    "RepresentationNetwork",
    "SearchConfig",
    "SearchDiagnostics",
    "SearchResult",
    "apply_action_mask",
    "build_muzero_network",
    "count_parameters",
    "discounted_backup",
    "env_action_index",
    "masked_policy_probs",
    "mu_zero_action_index",
    "observation_dim",
    "parameter_report",
    "project_action_mask",
    "scalar_to_support",
    "support_to_scalar",
    "visit_count_policy",
]
