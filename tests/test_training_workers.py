"""Phase 3 tests: EnvWorker + collectors (sync/process), transition validity,
seed derivation, and worker lifecycle."""

from __future__ import annotations

import numpy as np
from forexmind.episodes.config import EpisodeConfig
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig
from forexmind.training.collector import (
    EnvWorker,
    SyncCollector,
    worker_episode_seed,
)
from forexmind.training.config import ModelConfig
from forexmind.training.policies import build_policy_network


def _worker(dataset, *, algorithm: str = "sac") -> EnvWorker:
    from forexmind.config import default_config

    env_config = default_config(
        initial_balance="10000",
        leverage=50,
        spread_value=0.0002,
        sizing_mode="equity_fraction",
    )
    encoder = ObservationEncoder(EncoderConfig(context_length=8, initial_balance="10000"))
    window_config = WindowConfig(context_length=8)
    episode_config = EpisodeConfig(split="train", horizon=16, context_length=8, seed=1)
    model = ModelConfig(hidden_dim=16, num_layers=2)
    policy = build_policy_network(algorithm, encoder.config.spec.encoded_shape[0], 1, model)
    return EnvWorker(
        dataset=dataset,
        env_config=env_config,
        encoder_config=encoder.config,
        window_config=window_config,
        episode_config=episode_config,
        algorithm=algorithm,
        model_config=model,
        obs_dim=encoder.config.spec.encoded_shape[0],
        action_dim=1,
        worker_id=0,
        global_seed=123,
        policy=policy,
    )


def _dataset():
    from tests.synthetic import make_instrument, make_split_dataset, timeline_m5

    dates = [
        "2020-01-06", "2020-06-01", "2020-12-07",
        "2021-03-01", "2021-09-01", "2022-03-01", "2022-09-01",
    ]
    return make_split_dataset(
        {"EURUSD": make_instrument("EURUSD", timeline_m5(dates, per_day=40))}
    )


def test_worker_episode_seed_deterministic() -> None:
    a = worker_episode_seed(42, 0, 0)
    b = worker_episode_seed(42, 0, 0)
    assert a == b
    assert worker_episode_seed(42, 0, 0) != worker_episode_seed(42, 1, 0)
    assert worker_episode_seed(42, 0, 0) != worker_episode_seed(43, 0, 0)


def test_worker_steps_produce_valid_transitions() -> None:
    ds = _dataset()
    worker = _worker(ds)
    transitions = [worker.step(random_action=False) for _ in range(40)]
    assert len(transitions) == 40
    obs_dim = transitions[0].obs.shape[0]
    assert obs_dim == worker.obs_dim
    assert obs_dim == 8 * 5 + 10 + 14 + 7  # context=8 encoded observation
    for t in transitions:
        assert -1.0 <= t.action <= 1.0
        assert t.obs.shape == (obs_dim,)
        assert t.next_obs.shape == (obs_dim,)
        assert np.isfinite(t.obs).all()
        assert isinstance(t.terminated, bool)
        assert isinstance(t.truncated, bool)
    assert worker.completed_episodes >= 1  # 40 steps / horizon 16


def test_worker_random_warmup_actions() -> None:
    ds = _dataset()
    worker = _worker(ds)
    actions = [worker.step(random_action=True).action for _ in range(20)]
    assert all(-1.0 <= a <= 1.0 for a in actions)
    # Random actions should not be all identical.
    assert len(set(actions)) > 1


def test_sync_collector_deterministic() -> None:
    ds = _dataset()
    # Both workers share the same policy instance so sampling is comparable.
    from forexmind.config import default_config
    from forexmind.episodes.config import EpisodeConfig
    from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
    from forexmind.observation.window import WindowConfig
    from forexmind.training.config import ModelConfig
    from forexmind.training.policies import build_policy_network

    env_config = default_config(
        initial_balance="10000", leverage=50, spread_value=0.0002,
        sizing_mode="equity_fraction",
    )
    encoder = ObservationEncoder(EncoderConfig(context_length=8, initial_balance="10000"))
    window_config = WindowConfig(context_length=8)
    episode_config = EpisodeConfig(split="train", horizon=16, context_length=8, seed=1)
    model = ModelConfig(hidden_dim=16, num_layers=2)
    policy = build_policy_network("sac", encoder.config.spec.encoded_shape[0], 1, model)

    def make():
        return EnvWorker(
            dataset=ds, env_config=env_config, encoder_config=encoder.config,
            window_config=window_config, episode_config=episode_config,
            algorithm="sac", model_config=model, obs_dim=encoder.config.spec.encoded_shape[0],
            action_dim=1, worker_id=0, global_seed=123, policy=policy,
        )

    c1 = SyncCollector(make())
    c2 = SyncCollector(make())
    t1 = c1.collect(20, random_action=False)
    t2 = c2.collect(20, random_action=False)
    for a, b in zip(t1, t2, strict=True):
        assert a.action == b.action
        assert np.array_equal(a.obs, b.obs)
    c1.close()
    c2.close()


def test_worker_ppo_transitions_carry_logprob_value() -> None:
    from forexmind.training.networks import ValueNet

    ds = _dataset()
    worker = _worker(ds, algorithm="ppo")
    worker.set_policy(
        worker._policy,
        ValueNet(worker.obs_dim, ModelConfig(hidden_dim=16, num_layers=2)),
    )
    t = worker.step(random_action=False)
    assert isinstance(t.log_prob, float)
    assert isinstance(t.value, float)
    assert isinstance(t.next_value, float)
    assert np.isfinite(t.log_prob)


def test_process_collector_start_stop_collect(tmp_path) -> None:
    """Small process-backend round trip (spawn) to validate lifecycle."""
    from forexmind.config import default_config
    from forexmind.episodes.config import EpisodeConfig
    from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
    from forexmind.observation.window import WindowConfig
    from forexmind.training.collector import ProcessCollector

    # Write synthetic parquet so spawned workers can load it independently.
    from tests.synthetic import make_instrument, make_split_dataset, timeline_m5

    dates = ["2020-01-06", "2020-06-01", "2020-12-07"]
    inst = make_instrument("EURUSD", timeline_m5(dates, per_day=40))
    proc_dir = tmp_path / "processed"
    (proc_dir / "EURUSD").mkdir(parents=True)
    inst.m1.to_parquet(proc_dir / "EURUSD" / "m1.parquet")
    inst.m5.to_parquet(proc_dir / "EURUSD" / "m5.parquet")
    ds = make_split_dataset({"EURUSD": inst})

    env_config = default_config(
        initial_balance="10000", leverage=50, spread_value=0.0002,
        sizing_mode="equity_fraction",
    )
    encoder = ObservationEncoder(EncoderConfig(context_length=8, initial_balance="10000"))
    window_config = WindowConfig(context_length=8)
    episode_config = EpisodeConfig(split="train", horizon=16, context_length=8, seed=1)
    model = ModelConfig(hidden_dim=16, num_layers=2)
    policy = build_policy_network("sac", encoder.config.spec.encoded_shape[0], 1, model)

    collector = ProcessCollector(
        processed_dir=str(proc_dir),
        split_config=ds.split_config,
        instruments=ds.instruments,
        env_config=env_config,
        encoder_config=encoder.config,
        window_config=window_config,
        episode_config=episode_config,
        algorithm="sac",
        model=model,
        obs_dim=encoder.config.spec.encoded_shape[0],
        action_dim=1,
        global_seed=123,
        num_workers=2,
    )
    collector.set_policy(policy)
    try:
        batch = collector.collect(32, random_action=False)
    finally:
        collector.close()
    assert len(batch) == 32
    for t in batch:
        assert -1.0 <= t.action <= 1.0
