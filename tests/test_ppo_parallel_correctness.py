"""Stage 3.3 PPO rollout correctness regressions.

These tests use synthetic transitions so the expected GAE and episode
accounting behavior is mathematical and deterministic.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from forexmind.training.collector import Transition
from forexmind.training.config import ExperimentConfig


def _ds():
    from tests.synthetic import make_instrument, make_split_dataset, timeline_m5

    dates = [
        "2020-01-06",
        "2020-06-01",
        "2020-12-07",
        "2021-03-01",
        "2021-09-01",
        "2022-03-01",
        "2022-09-01",
    ]
    return make_split_dataset({"EURUSD": make_instrument("EURUSD", timeline_m5(dates, per_day=40))})


def _trainer(tmp_path):
    from forexmind.training.ppo import PPOTrainer

    cfg = ExperimentConfig.smoke("ppo")
    cfg.environment.instruments = ("EURUSD",)
    cfg.logging.run_dir = "runs_smoke_test"
    return PPOTrainer(cfg, tmp_path, dataset=_ds())


def _transition(
    trainer,
    *,
    worker_id: int,
    trajectory_id: int,
    trajectory_step: int,
    reward: float,
    rollout_fragment_id: int,
    terminated: bool = False,
    truncated: bool = False,
    value: float = 0.0,
    next_value: float = 0.0,
) -> Transition:
    obs = np.full(trainer.obs_dim, worker_id + trajectory_id * 0.01, dtype=np.float32)
    return Transition(
        obs=obs,
        action=0.0,
        reward=reward,
        next_obs=obs + 1.0,
        terminated=terminated,
        truncated=truncated,
        log_prob=0.0,
        value=value,
        next_value=next_value,
        next_bootstrap=not terminated,
        worker_id=worker_id,
        trajectory_id=trajectory_id,
        trajectory_step=trajectory_step,
        rollout_fragment_id=rollout_fragment_id,
        action_raw=0.0,
    )


def _reference_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    bootstrap: np.ndarray,
    segment_ids: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        mask = bool(bootstrap[i])
        delta = rewards[i] + gamma * (next_values[i] if mask else 0.0) - values[i]
        same_segment = i + 1 < len(rewards) and segment_ids[i + 1] == segment_ids[i]
        last = delta + gamma * lam * (last if mask and same_segment else 0.0)
        adv[i] = last
    return adv, adv + values


def test_gae_worker_isolation(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    rewards = np.asarray([1, 1, 1, 100, 100, 100], dtype=np.float32)
    segments = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    bootstrap = np.ones_like(rewards, dtype=bool)
    adv_a, _ = trainer._compute_gae(
        rewards,
        np.zeros_like(rewards, dtype=bool),
        np.zeros_like(rewards),
        np.zeros_like(rewards),
        bootstrap_mask=bootstrap,
        segment_ids=segments,
    )
    changed = rewards.copy()
    changed[3:] = [1000, -500, 33]
    adv_b, _ = trainer._compute_gae(
        changed,
        np.zeros_like(rewards, dtype=bool),
        np.zeros_like(rewards),
        np.zeros_like(rewards),
        bootstrap_mask=bootstrap,
        segment_ids=segments,
    )
    np.testing.assert_allclose(adv_a[:3], adv_b[:3], rtol=1e-6)


def test_gae_fragment_boundary_isolation(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    rewards = np.asarray([2, 2, 2, 50, 50, 50], dtype=np.float32)
    segments = np.asarray([10, 10, 10, 11, 11, 11], dtype=np.int64)
    zeros = np.zeros_like(rewards)
    adv_a, _ = trainer._compute_gae(
        rewards,
        np.zeros_like(rewards, dtype=bool),
        zeros,
        zeros,
        bootstrap_mask=np.ones_like(rewards, dtype=bool),
        segment_ids=segments,
    )
    rewards[3:] = -999.0
    adv_b, _ = trainer._compute_gae(
        rewards,
        np.zeros_like(rewards, dtype=bool),
        zeros,
        zeros,
        bootstrap_mask=np.ones_like(rewards, dtype=bool),
        segment_ids=segments,
    )
    np.testing.assert_allclose(adv_a[:3], adv_b[:3], rtol=1e-6)


def test_gae_terminal_does_not_bootstrap(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    adv, ret = trainer._compute_gae(
        np.asarray([1.0], dtype=np.float32),
        np.asarray([True]),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([100.0], dtype=np.float32),
    )
    np.testing.assert_allclose(adv, [1.0], rtol=1e-6)
    np.testing.assert_allclose(ret, [1.0], rtol=1e-6)


def test_gae_truncation_bootstrap_semantics(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    adv, _ = trainer._compute_gae(
        np.asarray([1.0], dtype=np.float32),
        np.asarray([False]),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([10.0], dtype=np.float32),
    )
    np.testing.assert_allclose(adv, [1.0 + trainer.config.training.gamma * 10.0], rtol=1e-6)


def test_gae_matches_reference_implementation(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    rewards = np.asarray([0.1, -0.2, 0.3, 1.0, 2.0], dtype=np.float32)
    values = np.asarray([0.5, 0.4, -0.1, 0.2, 0.3], dtype=np.float32)
    next_values = np.asarray([0.4, -0.1, 0.7, 0.3, 0.9], dtype=np.float32)
    bootstrap = np.asarray([True, True, False, True, True])
    segments = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    expected_adv, expected_ret = _reference_gae(
        rewards,
        values,
        next_values,
        bootstrap,
        segments,
        trainer.config.training.gamma,
        trainer.config.training.gae_lambda,
    )
    adv, ret = trainer._compute_gae(
        rewards,
        np.asarray([False, False, True, False, False]),
        values,
        next_values,
        bootstrap_mask=bootstrap,
        segment_ids=segments,
    )
    np.testing.assert_allclose(adv, expected_adv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ret, expected_ret, rtol=1e-6, atol=1e-6)


def test_parallel_rollout_order_invariant_and_episode_accounting(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    by_worker: list[list[Transition]] = []
    expected_returns: list[float] = []
    for wid in range(4):
        worker_batch: list[Transition] = []
        for ep in range(2):
            rewards = [float(wid * 100 + ep * 10 + step) for step in range(4)]
            expected_returns.append(sum(rewards))
            for step, reward in enumerate(rewards):
                worker_batch.append(
                    _transition(
                        trainer,
                        worker_id=wid,
                        trajectory_id=wid * 10 + ep,
                        trajectory_step=step,
                        reward=reward,
                        rollout_fragment_id=wid,
                        truncated=step == 3,
                    )
                )
        by_worker.append(worker_batch)

    rollout_a = [t for worker_batch in by_worker for t in worker_batch]
    rollout_b = [t for worker_batch in reversed(by_worker) for t in worker_batch]

    adv_a, ret_a, integ_a = trainer._gae_for_rollout(rollout_a)
    adv_b, ret_b, integ_b = trainer._gae_for_rollout(rollout_b)
    key_a = {
        (t.worker_id, t.trajectory_id, t.trajectory_step): (adv_a[i], ret_a[i])
        for i, t in enumerate(rollout_a)
    }
    key_b = {
        (t.worker_id, t.trajectory_id, t.trajectory_step): (adv_b[i], ret_b[i])
        for i, t in enumerate(rollout_b)
    }
    assert key_a.keys() == key_b.keys()
    for key in key_a:
        np.testing.assert_allclose(key_a[key], key_b[key], rtol=1e-6, atol=1e-6)
    assert integ_a["unique_trajectories"] == integ_b["unique_trajectories"] == 8.0
    assert integ_a["rollout_completed_episodes"] == 8.0

    trainer._ingest(rollout_b)
    assert trainer._episodes == 8
    assert sorted(trainer._episode_returns) == sorted(expected_returns)
    assert trainer._episode_lengths == [4] * 8


def test_checkpoint_restores_best_score_and_rng(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    trainer.best_score = 12.5
    trainer.best_checkpoint = "best"
    trainer._env_steps = 123
    trainer._gradient_updates = 7
    trainer._episodes = 3

    np.random.seed(991)
    random.seed(992)
    torch.manual_seed(993)
    trainer._rng = np.random.default_rng(994)

    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.get_rng_state()
    local_state = copy.deepcopy(trainer._rng.bit_generator.state)
    expected_np = np.random.random(3)
    expected_py = [random.random() for _ in range(3)]
    expected_torch = torch.rand(3)
    expected_local = trainer._rng.random(3)

    np.random.set_state(np_state)
    random.setstate(py_state)
    torch.set_rng_state(torch_state)
    trainer._rng.bit_generator.state = local_state
    trainer._save_checkpoint("resume")

    np.random.random(9)
    [random.random() for _ in range(9)]
    torch.rand(9)
    trainer._rng.random(9)

    restored = _trainer(tmp_path)
    restored._restore_from_checkpoint(tmp_path / "checkpoints" / "resume.pt")

    assert restored.best_score == 12.5
    assert restored.best_checkpoint == "best"
    assert restored._env_steps == 123
    assert restored._gradient_updates == 7
    assert restored._episodes == 3
    np.testing.assert_allclose(np.random.random(3), expected_np)
    np.testing.assert_allclose([random.random() for _ in range(3)], expected_py)
    torch.testing.assert_close(torch.rand(3), expected_torch)
    np.testing.assert_allclose(restored._rng.random(3), expected_local)
