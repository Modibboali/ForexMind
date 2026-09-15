"""MuZero for ForexMind (Stages 4.1-4.3).

Stage 4.1 added the neural core and inference contracts (``h_theta``,
``g_theta``, ``f_theta``).  Stage 4.2 added the MCTS/PUCT search layer over the
frozen six-action space.  Stage 4.3 adds real trajectory collection,
trajectory-level replay, and MuZero policy/value/reward target construction.

Still **not** implemented: the MuZero loss, optimizer, target network,
reanalysis, prioritized replay, distributed actors, or Stochastic MuZero.

Quick start::

    from forexmind.muzero import MuZeroConfig, build_muzero_network, MuZeroMCTS
    from forexmind.muzero import SearchConfig, CollectorConfig, MuZeroCollector
    from forexmind.muzero import ReplayConfig, TrajectoryReplayBuffer, TargetConfig

    config = MuZeroConfig.from_encoder_config()  # obs_dim derived from the encoder
    model = build_muzero_network(config)
    search = MuZeroMCTS(model, SearchConfig(num_simulations=50).evaluation())

    result = search.search_from_env(env, observation)
    result.action, result.policy, result.visit_counts

    collector = MuZeroCollector(dataset, env_config, encoder_config, model, CollectorConfig())
    replay = TrajectoryReplayBuffer(ReplayConfig(max_trajectories=64))
    collector.collect_into(replay, 4)
    batch = replay.sample(8, target_config=TargetConfig(num_unroll_steps=5))
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
from forexmind.muzero.calibration import (
    TargetStatistics,
    calibration_report,
    describe_targets,
    propose_scale,
)
from forexmind.muzero.collector import (
    CollectedTrajectory,
    CollectionStats,
    CollectorConfig,
    EpisodeRun,
    MuZeroCollector,
    derive_decision_seed,
    derive_episode_seed,
    derive_search_seed,
)
from forexmind.muzero.config import MuZeroConfig, SearchConfig, observation_dim
from forexmind.muzero.diagnostics import (
    RootSearchRecord,
    search_summary,
    staleness_summary,
)
from forexmind.muzero.evaluation import (
    MuZeroAgent,
    MuZeroEvaluation,
    MuZeroEvaluator,
)
from forexmind.muzero.inference import (
    MuZeroNetwork,
    apply_action_mask,
    build_muzero_network,
    decode_scalar,
    masked_policy_probs,
)
from forexmind.muzero.inference_service import (
    BatchedInferenceServer,
    InferenceBackend,
    InferenceRequest,
    InferenceResponse,
    InferenceServiceError,
    InferenceStats,
    LocalInferenceBackend,
    RemoteInferenceBackend,
)
from forexmind.muzero.learner import (
    LearnerConfig,
    MuZeroLearner,
    OptimizerConfig,
    check_batch_shapes,
)
from forexmind.muzero.losses import (
    LossConfig,
    MuZeroLossResult,
    MuZeroPrediction,
    muzero_losses,
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
from forexmind.muzero.packed_replay import (
    PackedTrajectories,
    build_vectorized_batch,
    sample_reference,
)
from forexmind.muzero.parallel_collector import (
    CollectorPoolConfig,
    CollectorWorkerError,
    LockedReplay,
    MuZeroCollectorPool,
    ReplayWriter,
    WorkerDatasetSpec,
)
from forexmind.muzero.profiling import PhaseTimer, merge_phase_reports
from forexmind.muzero.replay import (
    SAMPLING_STRATEGIES,
    ReplayConfig,
    TrajectoryReplayBuffer,
)
from forexmind.muzero.replay_store import (
    load_replay,
    replay_store_report,
    save_replay,
)
from forexmind.muzero.search import (
    MuZeroMCTS,
    SearchDiagnostics,
    SearchResult,
    discounted_backup,
    visit_count_policy,
)
from forexmind.muzero.support import (
    SUPPORT_RANGE,
    inverse_transform_to_scalar,
    saturation_fraction,
    scalar_to_support,
    support_to_scalar,
    transform_to_scalar,
)
from forexmind.muzero.targets import (
    MuZeroBatch,
    MuZeroSample,
    TargetConfig,
    build_unroll_sample,
    collate_samples,
    value_target,
)
from forexmind.muzero.trainer import MuZeroTrainer, MuZeroTrainingConfig
from forexmind.muzero.trajectory import (
    MuZeroTrajectory,
    TrajectoryMetadata,
    model_version,
)
from forexmind.muzero.types import NetworkOutput

__all__ = [
    "MUZERO_ACTION_NAMES",
    "MUZERO_ENV_ACTION_INDICES",
    "MUZERO_NUM_ACTIONS",
    "MUZERO_TARGET_EXPOSURES",
    "SAMPLING_STRATEGIES",
    "SUPPORT_RANGE",
    "BatchedInferenceServer",
    "CollectedTrajectory",
    "CollectionStats",
    "CollectorConfig",
    "CollectorPoolConfig",
    "CollectorWorkerError",
    "DynamicsNetwork",
    "EpisodeRun",
    "InferenceBackend",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceServiceError",
    "InferenceStats",
    "LearnerConfig",
    "LocalInferenceBackend",
    "LockedReplay",
    "LossConfig",
    "MinMaxStats",
    "MuZeroAgent",
    "MuZeroBatch",
    "MuZeroCollector",
    "MuZeroCollectorPool",
    "MuZeroConfig",
    "MuZeroEvaluation",
    "MuZeroEvaluator",
    "MuZeroLearner",
    "MuZeroLossResult",
    "MuZeroMCTS",
    "MuZeroNetwork",
    "MuZeroPrediction",
    "MuZeroSample",
    "MuZeroTrainer",
    "MuZeroTrainingConfig",
    "MuZeroTrajectory",
    "NetworkOutput",
    "Node",
    "OptimizerConfig",
    "PackedTrajectories",
    "PhaseTimer",
    "PlanningState",
    "PredictionNetwork",
    "RemoteInferenceBackend",
    "ReplayConfig",
    "ReplayWriter",
    "RepresentationNetwork",
    "RootSearchRecord",
    "SearchConfig",
    "SearchDiagnostics",
    "SearchResult",
    "TargetConfig",
    "TargetStatistics",
    "TrajectoryMetadata",
    "TrajectoryReplayBuffer",
    "WorkerDatasetSpec",
    "apply_action_mask",
    "build_muzero_network",
    "build_unroll_sample",
    "build_vectorized_batch",
    "calibration_report",
    "check_batch_shapes",
    "collate_samples",
    "count_parameters",
    "decode_scalar",
    "derive_decision_seed",
    "derive_episode_seed",
    "derive_search_seed",
    "describe_targets",
    "discounted_backup",
    "env_action_index",
    "inverse_transform_to_scalar",
    "load_replay",
    "masked_policy_probs",
    "merge_phase_reports",
    "model_version",
    "mu_zero_action_index",
    "muzero_losses",
    "observation_dim",
    "parameter_report",
    "project_action_mask",
    "propose_scale",
    "replay_store_report",
    "sample_reference",
    "saturation_fraction",
    "save_replay",
    "scalar_to_support",
    "search_summary",
    "staleness_summary",
    "support_to_scalar",
    "transform_to_scalar",
    "value_target",
    "visit_count_policy",
]
