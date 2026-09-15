"""Stage 4.6 tests: batched inference, batched search, parallel collectors.

Coverage map (brief sections):

* S9-S13 batched inference  -> ``test_compute_batch_matches_individual_calls``,
  ``test_service_batching_preserves_values``, ``test_inference_stats_*``
* S14-S15 search concurrency -> ``test_batched_search_*``
* S4 worker seeding          -> ``test_worker_seed_derivation_*``
* S7/S35/S36 queues + shutdown -> ``test_pool_backpressure_and_clean_shutdown``
* S34 numerical equivalence  -> ``test_single_process_and_parallel_collectors_agree``
* S37/S38 failure handling   -> ``test_worker_failure_is_reported``,
  ``test_inference_service_failure_propagates``,
  ``test_inference_timeout_propagates``
"""

from __future__ import annotations

import queue
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import torch
from forexmind.config import (
    EnvironmentConfig,
    ExecutionConfig,
    MarginConfig,
    PositionSizingConfig,
)
from forexmind.data.splits import SplitConfig, SplitDataset
from forexmind.muzero.actions import MUZERO_NUM_ACTIONS, PlanningState
from forexmind.muzero.collector import (
    CollectorConfig,
    MuZeroCollector,
    derive_decision_seed,
    derive_episode_seed,
    derive_search_seed,
)
from forexmind.muzero.config import MuZeroConfig, SearchConfig
from forexmind.muzero.inference import build_muzero_network
from forexmind.muzero.inference_service import (
    INITIAL_INFERENCE,
    RECURRENT_INFERENCE,
    BatchedInferenceServer,
    InferenceRequest,
    InferenceServiceError,
    InferenceTimeoutError,
    LocalInferenceBackend,
    RemoteInferenceBackend,
    compute_batch,
    response_to_network_output,
)
from forexmind.muzero.parallel_collector import (
    CollectorPoolConfig,
    CollectorWorkerError,
    MuZeroCollectorPool,
    WorkerDatasetSpec,
    trajectory_from_payload,
    trajectory_to_payload,
)
from forexmind.muzero.replay import ReplayConfig, TrajectoryReplayBuffer
from forexmind.muzero.search import MuZeroMCTS
from forexmind.observation.encoder import EncoderConfig

from tests.muzero_synthetic import make_trajectory
from tests.synthetic import make_instrument, make_split_dataset, make_test_split_config, timeline_m5

ALL_VALID = np.ones(MUZERO_NUM_ACTIONS, dtype=bool)
DATES = ["2020-01-06", "2020-03-02", "2020-06-01", "2020-09-07", "2020-12-07"]
#: TRAIN + VALIDATION + TEST coverage for tests that also run validation.
ALL_DATES = [*DATES, "2021-03-01", "2021-09-01", "2022-03-01"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _encoder_config() -> EncoderConfig:
    return EncoderConfig(context_length=8)


def _env_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        execution=ExecutionConfig(spread_mode="fixed", spread_value=0.0),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")),
        sizing=PositionSizingConfig(mode="equity_fraction"),
    )


def _model_config(encoder_config: EncoderConfig | None = None) -> MuZeroConfig:
    config = encoder_config or _encoder_config()
    return MuZeroConfig(
        obs_dim=config.spec.encoded_shape[0],
        latent_dim=8,
        hidden_dim=8,
        num_layers=1,
    )


def _dataset() -> SplitDataset:
    return make_split_dataset(
        {"EURUSD": make_instrument("EURUSD", timeline_m5(DATES, per_day=40))}
    )


def _write_parquet_dataset(
    root: Path,
    instruments: tuple[str, ...] = ("EURUSD",),
    *,
    dates: list[str] | None = None,
) -> SplitConfig:
    """Small synthetic parquet dataset so spawned workers can rebuild it."""
    split_config = make_test_split_config()
    for instrument in instruments:
        data = make_instrument(instrument, timeline_m5(dates or DATES, per_day=40))
        directory = root / instrument
        directory.mkdir(parents=True, exist_ok=True)
        data.m5.to_parquet(directory / "m5.parquet")
        data.m1.to_parquet(directory / "m1.parquet")
    return split_config


def _collector_config(**overrides) -> CollectorConfig:
    base: dict = dict(
        split="train",
        horizon=3,
        num_simulations=2,
        seed=0,
        training=True,
        temperature=1.0,
        capture_diagnostics=True,
        per_decision_seed=True,
    )
    base.update(overrides)
    return CollectorConfig(**base)


def _pool(
    tmp_path: Path,
    *,
    workers: int = 2,
    collectors_per_worker: int = 1,
    queue_size: int = 4,
    model: object | None = None,
    **config_overrides,
) -> tuple[MuZeroCollectorPool, TrajectoryReplayBuffer]:
    encoder_config = _encoder_config()
    model_config = _model_config(encoder_config)
    if model is None:
        model = build_muzero_network(model_config)
        model.eval()
    # TRAIN-only coverage is enough here: these tests never run validation.
    split_config = _write_parquet_dataset(tmp_path / "processed")
    dataset_spec = WorkerDatasetSpec(
        processed_dir=str(tmp_path / "processed"),
        split_config=split_config.to_dict(),
        instruments=("EURUSD",),
        backend="parquet",
    )
    pool_config = CollectorPoolConfig(
        num_workers=workers,
        collectors_per_worker=collectors_per_worker,
        max_inference_batch_size=8,
        max_batch_wait_ms=2.0,
        trajectory_queue_size=queue_size,
        **config_overrides,
    )
    replay = TrajectoryReplayBuffer(ReplayConfig(max_trajectories=32, seed=0))
    pool = MuZeroCollectorPool(
        config=pool_config,
        dataset_spec=dataset_spec,
        env_config=_env_config(),
        encoder_config=encoder_config,
        collector_config=_collector_config(),
        model_factory=lambda: build_muzero_network(model_config),
        model_config=model_config,
        model=model,
        replay=replay,
        model_version_string="test-weights",
    )
    return pool, replay


# --------------------------------------------------------------------------- #
# batched inference (S9-S13)
# --------------------------------------------------------------------------- #


def test_compute_batch_matches_individual_calls() -> None:
    """One batched model call must equal the same rows sent one at a time."""
    torch.manual_seed(0)
    config = _model_config()
    model = build_muzero_network(config)
    model.eval()

    rng = np.random.default_rng(0)
    observations = rng.normal(size=(4, config.obs_dim)).astype(np.float32)
    masks = np.ones((4, MUZERO_NUM_ACTIONS), dtype=bool)
    masks[1, 3] = False
    masks[2, 0] = True

    batched = compute_batch(
        model,
        [
            InferenceRequest(
                worker_id=0,
                request_id=1,
                kind=INITIAL_INFERENCE,
                observations=observations,
                action_masks=masks,
            )
        ],
    )[0]
    batched_output = response_to_network_output(batched)
    for index in range(observations.shape[0]):
        single = compute_batch(
            model,
            [
                InferenceRequest(
                    worker_id=0,
                    request_id=index,
                    kind=INITIAL_INFERENCE,
                    observations=observations[index : index + 1],
                    action_masks=masks[index : index + 1],
                )
            ],
        )[0]
        single_output = response_to_network_output(single)
        assert torch.allclose(
            batched_output.latent_state[index : index + 1], single_output.latent_state, atol=1e-6
        )
        assert torch.allclose(
            batched_output.policy_logits[index : index + 1],
            single_output.policy_logits,
            atol=1e-6,
        )
        assert torch.allclose(
            batched_output.value[index : index + 1], single_output.value, atol=1e-6
        )

    latents = batched_output.latent_state.detach().numpy()
    actions = np.array([0, 1, 2, 3], dtype=np.int64)
    recurrent_batched = response_to_network_output(
        compute_batch(
            model,
            [
                InferenceRequest(
                    worker_id=0,
                    request_id=1,
                    kind=RECURRENT_INFERENCE,
                    latent_states=latents,
                    actions=actions,
                )
            ],
        )[0]
    )
    for index in range(latents.shape[0]):
        single = response_to_network_output(
            compute_batch(
                model,
                [
                    InferenceRequest(
                        worker_id=0,
                        request_id=index,
                        kind=RECURRENT_INFERENCE,
                        latent_states=latents[index : index + 1],
                        actions=actions[index : index + 1],
                    )
                ],
            )[0]
        )
        assert torch.allclose(
            recurrent_batched.latent_state[index : index + 1],
            single.latent_state,
            atol=1e-6,
        )
        assert torch.allclose(
            recurrent_batched.reward[index : index + 1], single.reward, atol=1e-6
        )


def test_service_batching_preserves_values() -> None:
    """The request/response service must not change priors, values or rewards."""
    torch.manual_seed(1)
    config = _model_config()
    model = build_muzero_network(config)
    model.eval()

    request_queue: queue.Queue = queue.Queue()
    response_queue: queue.Queue = queue.Queue()
    server = BatchedInferenceServer(
        lambda: build_muzero_network(config),
        request_queue=request_queue,
        response_queues={0: response_queue},
        max_batch_size=8,
        max_batch_wait_ms=2.0,
    )
    server.sync_weights(model, version=3, model_version_string="weights")
    server.start()
    client = RemoteInferenceBackend(
        worker_id=0, request_queue=request_queue, response_queue=response_queue, timeout_s=30.0
    )
    try:
        rng = np.random.default_rng(2)
        observations = rng.normal(size=(3, config.obs_dim)).astype(np.float32)
        masks = np.ones((3, MUZERO_NUM_ACTIONS), dtype=bool)
        local = LocalInferenceBackend(model)
        reference = local.initial_inference(torch.as_tensor(observations), torch.as_tensor(masks))
        served = client.initial_inference(observations, masks)
        assert torch.allclose(reference.latent_state, served.latent_state, atol=1e-6)
        assert torch.allclose(reference.policy_logits, served.policy_logits, atol=1e-6)
        assert torch.allclose(reference.value, served.value, atol=1e-6)
        assert client.network_version == 3

        for index in range(observations.shape[0]):
            single = client.initial_inference(observations[index], masks[index])
            assert torch.allclose(
                served.latent_state[index : index + 1], single.latent_state, atol=1e-6
            )
            assert torch.allclose(
                served.policy_logits[index : index + 1], single.policy_logits, atol=1e-6
            )
    finally:
        server.stop()

    diagnostics = server.diagnostics()
    assert diagnostics["calls"] >= 4
    assert diagnostics["max_batch_size_observed"] >= 2
    assert diagnostics["errors"] == 0
    assert diagnostics["mean_batch_size"] >= 1.0
    assert diagnostics["network_version"] == 3


def test_inference_stats_track_batch_sizes() -> None:
    config = _model_config()
    model = build_muzero_network(config)
    model.eval()
    backend = LocalInferenceBackend(model)
    observations = np.zeros((5, config.obs_dim), dtype=np.float32)
    backend.initial_inference(observations, np.ones((5, MUZERO_NUM_ACTIONS), dtype=bool))
    backend.initial_inference(observations[:1], ALL_VALID)
    snapshot = backend.stats.snapshot()
    assert snapshot["initial_calls"] == 2
    assert snapshot["initial_items"] == 6
    assert snapshot["max_batch_size"] == 5.0
    assert snapshot["mean_batch_size"] == pytest.approx(3.0)
    assert snapshot["mean_inference_latency_ms"] > 0.0


def test_inference_service_failure_propagates() -> None:
    class BrokenModel:
        def __init__(self, config: MuZeroConfig) -> None:
            self.config = config
            self.network_version = 0

        def eval(self) -> None:
            return None

        def to(self, device: object) -> None:
            return None

        def parameters(self) -> list:
            return []

        def state_dict(self) -> dict:
            return {}

        def load_state_dict(self, state: dict, strict: bool = True) -> None:
            return None

        def initial_inference(self, observations: object, action_masks: object) -> object:
            raise RuntimeError("dynamics exploded")

    config = _model_config()
    request_queue: queue.Queue = queue.Queue()
    response_queue: queue.Queue = queue.Queue()
    server = BatchedInferenceServer(
        lambda: BrokenModel(config),
        request_queue=request_queue,
        response_queues={0: response_queue},
        max_batch_size=4,
    )
    server.start()
    client = RemoteInferenceBackend(
        worker_id=0, request_queue=request_queue, response_queue=response_queue, timeout_s=10.0
    )
    try:
        with pytest.raises(InferenceServiceError, match="dynamics exploded"):
            client.initial_inference(
                np.zeros(config.obs_dim, dtype=np.float32), ALL_VALID
            )
        assert server.fatal_error is not None
        # A second request must fail fast rather than deadlock.
        with pytest.raises(InferenceServiceError):
            client.initial_inference(np.zeros(config.obs_dim, dtype=np.float32), ALL_VALID)
    finally:
        server.stop()


def test_inference_timeout_propagates() -> None:
    client = RemoteInferenceBackend(
        worker_id=0,
        request_queue=queue.Queue(),
        response_queue=queue.Queue(),
        timeout_s=0.2,
    )
    with pytest.raises(InferenceTimeoutError):
        client.initial_inference(np.zeros(4, dtype=np.float32), ALL_VALID)


# --------------------------------------------------------------------------- #
# batched search (S14-S15, S31)
# --------------------------------------------------------------------------- #


def _search(model: object, **overrides) -> MuZeroMCTS:
    config = SearchConfig(num_simulations=4, seed=0, **overrides)
    return MuZeroMCTS(model, config, rng=np.random.default_rng(7))


def test_batched_search_with_one_root_is_identical_to_search() -> None:
    torch.manual_seed(3)
    model = build_muzero_network(_model_config())
    model.eval()
    observation = np.zeros(model.config.obs_dim, dtype=np.float32)

    single = _search(model, temperature=1.0, add_root_noise=True)
    result = single.search(observation, ALL_VALID)
    batched = _search(model, temperature=1.0, add_root_noise=True)
    results = batched.search_batch([observation], [ALL_VALID])

    assert len(results) == 1
    assert results[0].action == result.action
    assert np.array_equal(results[0].visit_counts, result.visit_counts)
    assert np.allclose(results[0].policy, result.policy, atol=0.0)
    assert results[0].root_value == result.root_value
    assert np.allclose(results[0].root_priors, result.root_priors, atol=0.0)
    assert results[0].diagnostics.recurrent_inference_calls == (
        result.diagnostics.recurrent_inference_calls
    )


def test_batched_search_matches_independent_per_root_searches() -> None:
    """Roots in a batch must see exactly what they would see alone."""
    torch.manual_seed(4)
    model = build_muzero_network(_model_config())
    model.eval()
    rng = np.random.default_rng(11)
    observations = [rng.normal(size=model.config.obs_dim).astype(np.float32) for _ in range(3)]
    masks = [ALL_VALID.copy() for _ in range(3)]
    masks[1][2] = False

    batched = _search(model, temperature=1.0, add_root_noise=True)
    generators = [np.random.default_rng(100 + index) for index in range(3)]
    batched_results = batched.search_batch(
        observations, masks, rngs=[np.random.default_rng(100 + index) for index in range(3)]
    )

    for index in range(3):
        single = _search(model, temperature=1.0, add_root_noise=True)
        single.rng = generators[index]
        reference = single.search(observations[index], masks[index], add_root_noise=True)
        assert batched_results[index].action == reference.action
        assert np.array_equal(batched_results[index].visit_counts, reference.visit_counts)
        assert np.allclose(batched_results[index].policy, reference.policy, atol=0.0)
        # Model maths is identical modulo batch-shape reduction order (S13).
        assert batched_results[index].root_value == pytest.approx(
            reference.root_value, rel=1e-5, abs=1e-7
        )
        assert np.allclose(
            batched_results[index].root_priors, reference.root_priors, atol=0.0
        )


def test_batched_search_preserves_tree_invariants() -> None:
    torch.manual_seed(5)
    model = build_muzero_network(_model_config())
    model.eval()
    search = _search(model, temperature=0.0, add_root_noise=False)
    rng = np.random.default_rng(12)
    observations = [rng.normal(size=model.config.obs_dim).astype(np.float32) for _ in range(4)]
    masks = [PlanningState.from_exposure(0.5).action_mask() for _ in range(4)]
    results = search.search_batch(observations, masks)
    assert len(results) == 4
    for result in results:
        assert float(result.visit_counts.sum()) == pytest.approx(search.config.num_simulations)
        assert result.visit_counts[~result.root_action_mask].sum() == 0.0
        assert float(result.policy.sum()) == pytest.approx(1.0)
        assert result.root_action_mask[0]  # HOLD always valid
        assert result.root_action_mask[result.action]
        assert np.isfinite(result.root_value)


# --------------------------------------------------------------------------- #
# seeding (S4)
# --------------------------------------------------------------------------- #


def test_worker_seed_derivation_is_deterministic_and_distinct() -> None:
    assert derive_episode_seed(0, 0, 0) == derive_episode_seed(0, 0, 0)
    assert derive_episode_seed(0, 0, 0) != derive_episode_seed(0, 1, 0)
    assert derive_episode_seed(0, 0, 0) != derive_episode_seed(0, 0, 1)
    assert derive_episode_seed(1, 0, 0) != derive_episode_seed(0, 0, 0)
    assert derive_search_seed(0, 0, 0) != derive_search_seed(0, 1, 0)
    assert derive_decision_seed(5, 0) != derive_decision_seed(5, 1)
    assert derive_decision_seed(5, 0) == derive_decision_seed(5, 0)


def test_worker_rank_changes_episode_choices() -> None:
    dataset = _dataset()
    model = build_muzero_network(_model_config())
    model.eval()
    base = _collector_config()
    collector = MuZeroCollector(
        dataset, _env_config(), _encoder_config(), model, base, instruments=("EURUSD",)
    )
    rank0 = [collector.start_run(index, worker_rank=0).spec for index in range(3)]
    rank1 = [collector.start_run(index, worker_rank=1).spec for index in range(3)]
    assert [(spec.instrument, spec.start_index) for spec in rank0] != [
        (spec.instrument, spec.start_index) for spec in rank1
    ]
    again = [collector.start_run(index, worker_rank=0).spec for index in range(3)]
    assert [spec.to_dict() for spec in rank0] == [spec.to_dict() for spec in again]


def test_collected_trajectories_store_planning_exposure_in_double_precision() -> None:
    """Regression: a float32 exposure can flip a mask near the 1e-3 tolerance.

    The mask is derived from the exposure, so the stored planning state must
    reproduce the stored masks exactly; parallel collection hit an intermittent
    ``action_masks[t] disagrees with the deterministic planning state`` failure
    until the stored exposure was widened to float64.
    """
    model = build_muzero_network(_model_config())
    model.eval()
    collector = MuZeroCollector(
        _dataset(),
        _env_config(),
        _encoder_config(),
        model,
        _collector_config(),
        instruments=("EURUSD",),
    )
    trajectory = collector.collect_with_diagnostics(0).trajectory
    assert trajectory.planning_exposure.dtype == np.float64
    trajectory.validate()
    for index in range(len(trajectory)):
        assert np.array_equal(
            trajectory.planning_state(index).action_mask(),
            trajectory.action_masks[index],
        )


# --------------------------------------------------------------------------- #
# trajectory wire format (S20)
# --------------------------------------------------------------------------- #


def test_trajectory_payload_round_trip() -> None:
    trajectory = make_trajectory(actions=[4, 1, 0], rewards=[0.0, -0.1, 0.2], trajectory_id=5)
    payload = trajectory_to_payload(trajectory, worker_id=2)
    restored, records = trajectory_from_payload(payload)
    assert records == []
    assert restored.metadata == trajectory.metadata
    # The planning exposure must survive the wire in double precision, because
    # the six-action mask is derived from it (float32 rounding can flip a mask).
    assert restored.planning_exposure.dtype == np.float64
    assert np.array_equal(restored.planning_exposure, trajectory.planning_exposure)
    assert np.array_equal(restored.observations, trajectory.observations)
    assert np.array_equal(restored.actions, trajectory.actions)
    assert np.array_equal(restored.root_policies, trajectory.root_policies)
    assert np.array_equal(restored.action_masks, trajectory.action_masks)
    assert restored.boundary_value == trajectory.boundary_value
    restored.validate()


# --------------------------------------------------------------------------- #
# parallel collectors (S3-S8, S34-S38)
# --------------------------------------------------------------------------- #


def test_single_process_and_parallel_collectors_agree(tmp_path) -> None:
    """Identical weights, seeds and episode spec must give identical results (S34)."""
    encoder_config = _encoder_config()
    model_config = _model_config(encoder_config)
    model = build_muzero_network(model_config)
    model.eval()
    collector_config = _collector_config(horizon=3, num_simulations=2)

    dataset = _dataset()
    sequential = MuZeroCollector(
        dataset,
        _env_config(),
        encoder_config,
        model,
        collector_config,
        instruments=("EURUSD",),
    )
    reference = sequential.collect_with_diagnostics(0)

    pool, replay = _pool(
        tmp_path,
        workers=1,
        collectors_per_worker=1,
        inference_mode="server",
        # Identical weights on both sides: the server copies this model.
        model=model,
    )
    try:
        collected = pool.collect_trajectories(1, timeout_s=120.0)
        diagnostics = pool.diagnostics(fresh=True)
    finally:
        shutdown = pool.close()

    assert replay.num_trajectories == 1
    assert shutdown["alive_workers"] == 0
    assert diagnostics["aggregate"]["dataset_backends"] == {"parquet": 1}
    assert diagnostics["writer"]["trajectories_written"] == 1

    produced = collected[0].trajectory
    expected = reference.trajectory
    assert len(produced) == len(expected)
    assert produced.metadata.episode_seed == expected.metadata.episode_seed
    assert produced.metadata.search_seed == expected.metadata.search_seed
    assert np.array_equal(produced.actions, expected.actions)
    assert np.array_equal(produced.action_masks, expected.action_masks)
    assert np.allclose(produced.rewards, expected.rewards, atol=0.0)
    assert np.allclose(produced.root_policies, expected.root_policies, atol=1e-6)
    assert np.allclose(produced.root_values, expected.root_values, atol=1e-5)
    assert np.allclose(produced.observations, expected.observations, atol=1e-6)
    assert produced.boundary_value == pytest.approx(expected.boundary_value, abs=1e-5)


def test_parallel_pool_collects_with_batching_and_accounting(tmp_path) -> None:
    pool, replay = _pool(tmp_path, workers=2, collectors_per_worker=2, queue_size=2)
    try:
        collected = pool.collect_trajectories(4, timeout_s=180.0)
        diagnostics = pool.diagnostics()
    finally:
        shutdown = pool.close()

    assert len(collected) == 4
    assert replay.num_trajectories >= 4
    assert len({item.trajectory.metadata.trajectory_id for item in collected}) == 4
    aggregate = diagnostics["aggregate"]
    assert aggregate["env_steps"] > 0
    assert aggregate["searches"] >= aggregate["env_steps"]
    # Worker-reported counters are snapshots and may lag a round behind the
    # writer (a slow-starting worker can legitimately have produced nothing when
    # its snapshot was taken); the authoritative count is the writer's.
    assert aggregate["trajectories_written"] >= 4
    written_by_worker = aggregate["trajectories_written_by_worker"]
    assert sum(written_by_worker.values()) == aggregate["trajectories_written"]
    assert set(written_by_worker) <= {0, 1}
    assert len(diagnostics["workers"]) == 2
    assert all(worker["pid"] for worker in diagnostics["workers"])
    assert diagnostics["config"]["num_collectors"] == 4
    inference = diagnostics["inference_server"]
    assert inference["calls"] > 0
    assert inference["errors"] == 0
    assert inference["max_batch_size_observed"] >= 1
    assert inference["fatal_error"] is None
    assert shutdown["alive_workers"] == 0
    assert shutdown["inference_server_stopped"] is True
    assert shutdown["writer_stopped"] is True
    assert shutdown["dropped"] == 0


def test_pool_backpressure_and_clean_shutdown(tmp_path) -> None:
    pool, replay = _pool(tmp_path, workers=2, collectors_per_worker=1, queue_size=1)
    try:
        pool.collect_trajectories(2, timeout_s=120.0)
        size = pool.queue_size()
        assert size is None or size <= pool.config.trajectory_queue_size
        for process in pool._processes:
            assert process.is_alive()
    finally:
        shutdown = pool.close()
    assert shutdown["workers_joined"] == 2
    assert shutdown["alive_workers"] == 0
    assert shutdown["queue_drained"] is True
    assert all(not process.is_alive() for process in pool._processes)
    assert replay.num_trajectories >= 2


def test_worker_failure_is_reported(tmp_path) -> None:
    pool, _replay = _pool(tmp_path, workers=2, collectors_per_worker=1, queue_size=4)
    try:
        pool.collect_trajectories(1, timeout_s=120.0)
        pool._processes[0].terminate()
        pool._processes[0].join(timeout=10.0)
        with pytest.raises(CollectorWorkerError) as error:
            pool.collect_trajectories(4, timeout_s=30.0)
        details = error.value.details
        assert details["worker_id"] == 0
        assert details["exitcode"] is not None
    finally:
        pool.close()


def test_pool_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        CollectorPoolConfig(num_workers=0)
    with pytest.raises(ValueError):
        CollectorPoolConfig(collectors_per_worker=0)
    with pytest.raises(ValueError, match="inference_mode"):
        CollectorPoolConfig(inference_mode="telepathy")
    with pytest.raises(ValueError, match="trajectory_queue_size"):
        CollectorPoolConfig(trajectory_queue_size=0)


# --------------------------------------------------------------------------- #
# trainer integration (S8, S16-S18, S36, S40)
# --------------------------------------------------------------------------- #


def _parallel_trainer(tmp_path: Path, **overrides):
    from forexmind.muzero.trainer import MuZeroTrainer, MuZeroTrainingConfig

    split_config = _write_parquet_dataset(tmp_path / "processed", dates=ALL_DATES)
    dataset = SplitDataset(
        split_config,
        lambda key: _lake_instrument(tmp_path / "processed", key),
        ("EURUSD",),
    )
    base: dict = dict(
        horizon=2,
        num_simulations=2,
        trajectories_per_iteration=2,
        min_replay_transitions_before_training=2,
        learner_updates_per_iteration=1,
        batch_size=2,
        unroll_steps=2,
        td_steps=2,
        latent_dim=8,
        hidden_dim=8,
        num_layers=1,
        max_trajectories=8,
        max_env_steps=6,
        eval_every_env_steps=6,
        eval_episodes=2,
        eval_horizon=2,
        checkpoint_every_env_steps=6,
        output_dir=tmp_path / "run",
        seed=0,
        num_collectors=2,
        collectors_per_worker=1,
        inference_mode="server",
        trajectory_queue_size=2,
        dataset_backend="parquet",
        processed_dir=tmp_path / "processed",
        profile=True,
    )
    base.update(overrides)
    return MuZeroTrainer(
        dataset,
        _env_config(),
        EncoderConfig(context_length=8),
        MuZeroTrainingConfig(**base),
    )


def _lake_instrument(processed_dir: Path, instrument: str):
    import pandas as pd
    from forexmind.data.dataset import InstrumentData

    directory = processed_dir / instrument
    return InstrumentData(
        instrument=instrument,
        m1=pd.read_parquet(directory / "m1.parquet"),
        m5=pd.read_parquet(directory / "m5.parquet"),
    )


def test_trainer_parallel_mode_collects_learns_and_shuts_down(tmp_path) -> None:
    trainer = _parallel_trainer(tmp_path)
    report = trainer.train()
    collection = report["collection"]
    assert collection["mode"] == "parallel"
    assert collection["num_workers"] == 2
    assert collection["num_collectors"] == 2
    assert report["counters"]["env_steps"] >= 6
    assert report["counters"]["trajectories_collected"] >= 2
    assert report["counters"]["gradient_updates"] >= 1
    assert report["counters"]["network_version"] >= 1
    inference = collection["inference"]
    assert inference is not None
    assert inference["calls"] > 0
    assert inference["errors"] == 0
    assert inference["network_version"] == report["counters"]["network_version"]
    writer = collection["writer"]
    assert writer["trajectories_written"] >= report["counters"]["trajectories_collected"]
    shutdown = collection["shutdown"]
    assert shutdown is not None
    assert shutdown["alive_workers"] == 0
    assert shutdown["queue_drained"] is True
    # Profiling phases are reported separately (brief S2).
    phases = report["profiling"]["phases"]
    for name in (
        "environment_step",
        "observation_encode",
        "initial_inference",
        "recurrent_inference",
    ):
        assert name in phases, name
    # Every checkpoint still carries the versioning/staleness contract (S16-S18).
    assert report["resume"]["resumed"] is False
    trainer.close()


def test_trainer_ratio_guard_uses_measured_transitions(tmp_path) -> None:
    trainer = _parallel_trainer(
        tmp_path, learner_updates_per_iteration=0, target_updates_per_env_step=0.5
    )
    assert trainer.updates_for_iteration(4) == 2
    assert trainer.updates_for_iteration(3) == 2  # round(1.5)
    assert trainer.updates_for_iteration(0) == 0
    trainer.close()
