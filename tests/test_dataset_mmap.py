"""Shared memory-mapped dataset tests (PPO memory audit).

Verifies that the memory-mapped backend is bit-identical to the reference
parquet backend and that worker environments produce identical transitions
(observations, rewards, actions, done flags) for identical seeds/episodes.

Uses a small synthetic dataset so the test is self-contained and fast.
"""

from __future__ import annotations

import numpy as np
from forexmind.training.data import load_processed_from_dir, make_training_dataset
from forexmind.training.dataset_mmap import (
    build_shared_store,
    make_mmap_dataset,
    open_instrument_data,
)


def _write_synthetic_processed(tmp_path) -> None:
    from tests.synthetic import make_instrument, timeline_m5

    dates = [
        "2020-01-06",
        "2020-06-01",
        "2020-12-07",
        "2021-03-01",
        "2021-09-01",
        "2022-03-01",
    ]
    for instr in ("EURUSD", "GBPUSD"):
        inst = make_instrument(instr, timeline_m5(dates, per_day=40))
        d = tmp_path / "processed" / instr
        d.mkdir(parents=True)
        inst.m1.to_parquet(d / "m1.parquet")
        inst.m5.to_parquet(d / "m5.parquet")


def test_store_roundtrip_bit_exact(tmp_path) -> None:
    _write_synthetic_processed(tmp_path)
    store = build_shared_store(tmp_path / "processed", ("EURUSD", "GBPUSD"))
    for instr in ("EURUSD", "GBPUSD"):
        ref = load_processed_from_dir(instr, tmp_path / "processed")
        mm = open_instrument_data(store, instr)
        for tf in ("m1", "m5"):
            rf, mf = getattr(ref, tf), getattr(mm, tf)
            assert list(rf.columns) == list(mf.columns)
            assert len(rf) == len(mf)
            for col in rf.columns:
                a, b = rf[col].to_numpy(), mf[col].to_numpy()
                assert a.dtype == b.dtype, (instr, tf, col, a.dtype, b.dtype)
                assert np.array_equal(a, b, equal_nan=True), (instr, tf, col)


def _workers(tmp_path, *, backend: str):
    from forexmind.config import default_config
    from forexmind.episodes.config import EpisodeConfig
    from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
    from forexmind.observation.window import WindowConfig
    from forexmind.training.collector import EnvWorker
    from forexmind.training.config import ModelConfig

    from tests.synthetic import make_test_split_config

    instruments = ("EURUSD", "GBPUSD")
    split_config = make_test_split_config()
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
    obs_dim = encoder.config.spec.encoded_shape[0]

    if backend == "parquet":
        dataset = make_training_dataset(tmp_path / "processed", split_config, instruments)
    else:
        dataset = make_mmap_dataset(
            build_shared_store(tmp_path / "processed", instruments),
            split_config,
            instruments,
        )
    return EnvWorker(
        dataset=dataset,
        env_config=env_config,
        encoder_config=encoder.config,
        window_config=window_config,
        episode_config=episode_config,
        algorithm="ppo",
        model_config=model,
        obs_dim=obs_dim,
        action_dim=1,
        worker_id=3,
        global_seed=42,
        policy=None,
    )


def test_mmap_worker_matches_parquet_worker(tmp_path) -> None:
    _write_synthetic_processed(tmp_path)
    w_ref = _workers(tmp_path, backend="parquet")
    w_mm = _workers(tmp_path, backend="mmap")
    n_steps = 160
    for _ in range(n_steps):
        a = w_ref.step(random_action=True)
        b = w_mm.step(random_action=True)
        assert a.action == b.action
        assert a.reward == b.reward
        assert a.terminated == b.terminated
        assert a.truncated == b.truncated
        assert a.instrument == b.instrument
        np.testing.assert_array_equal(a.obs, b.obs)
        np.testing.assert_array_equal(a.next_obs, b.next_obs)
