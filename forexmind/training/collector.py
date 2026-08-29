"""Experience collection for multi-CPU RL training (Phase 3).

Two backends:

* ``sync``: a single in-process :class:`EnvWorker` steps environments
  deterministically (used for smoke/CI and deterministic small runs).
* ``process``: ``num_workers`` independent worker processes, each running its
  own environments and sending transitions back to the learner through a queue
  (``Learner -> Workers -> Experience -> Replay Buffer -> Learner``).

Worker RNGs are derived reproducibly from ``(global_seed, worker_id,
episode_index)``.  Workers load the processed parquet dataset themselves (the
practical approach on Windows ``spawn``); on Linux/fork the dataset can be
inherited copy-on-write.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from forexmind.config import EnvironmentConfig
from forexmind.data.splits import SplitConfig, SplitDataset
from forexmind.environment import ForexEnvironment
from forexmind.episodes.config import EpisodeConfig
from forexmind.episodes.sampler import EpisodeSampler
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import MarketWindowBuilder, WindowConfig
from forexmind.training.policies import build_policy_network, sample_action


@dataclass(frozen=True, slots=True)
class Transition:
    obs: np.ndarray
    action: float
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    log_prob: float = 0.0
    value: float = 0.0
    next_value: float = 0.0
    worker_id: int = -1
    worker_pid: int = 0
    # Diagnostic-only provenance for the first-non-finite audit.
    instrument: str = ""
    timestamp: object | None = None
    # For PPO: the raw pre-clamp Gaussian sample ``u ~ N(mean, std)``.  The
    # importance ratio must use the density at THIS value (the actual sample
    # of the policy's Gaussian); ``action`` remains the env-facing clamped
    # projection.  ``action_raw == action`` when the sample was not clamped.
    action_raw: float = 0.0


def worker_episode_seed(global_seed: int, worker_id: int, episode_index: int) -> int:
    return (global_seed * 1_000_003 + worker_id * 1_000_033 + episode_index * 7_919) % (2**31)


class EnvWorker:
    """Owns training environments and produces transitions.

    The same class is used in-process (sync) and as a subprocess target
    (process).  ``policy`` may be ``None`` for pure random collection (warmup).
    """

    def __init__(
        self,
        dataset: SplitDataset,
        env_config: EnvironmentConfig,
        encoder_config: EncoderConfig,
        window_config: WindowConfig,
        episode_config: EpisodeConfig,
        algorithm: str,
        model_config: object,
        obs_dim: int,
        action_dim: int,
        worker_id: int,
        global_seed: int,
        policy: nn.Module | None = None,
    ) -> None:
        self.dataset = dataset
        self.env_config = env_config
        self.encoder = ObservationEncoder(encoder_config)
        self.window_config = window_config
        self.episode_config = episode_config
        self.algorithm = algorithm
        self.model_config = model_config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.worker_id = worker_id
        self.global_seed = global_seed
        self._policy = policy
        self._value_net: nn.Module | None = None
        self._rng = np.random.default_rng(global_seed + worker_id * 1009)
        self._total_steps = 0
        self._sampler = EpisodeSampler(dataset, episode_config)
        self._envs: dict[str, ForexEnvironment] = {}
        self._builders: dict[str, MarketWindowBuilder] = {}
        self._active: tuple[ForexEnvironment, MarketWindowBuilder, object, np.ndarray] | None = None
        self._episode_index = 0

    # -- policy ---------------------------------------------------------------

    def set_policy(self, policy: nn.Module | None, value_net: nn.Module | None = None) -> None:
        self._policy = policy
        self._value_net = value_net

    # -- environment lifecycle ------------------------------------------------

    def _make_env(self, instrument: str) -> ForexEnvironment:
        if instrument not in self._envs:
            from forexmind.data.dataset import MarketDataset

            ds = MarketDataset()
            ds.add(self.dataset.load(instrument))
            self._envs[instrument] = ForexEnvironment(ds, self.env_config, instrument=instrument)
        return self._envs[instrument]

    def _make_builder(self, instrument: str) -> MarketWindowBuilder:
        key = instrument
        if key not in self._builders:
            start, end = self.dataset.split_config.range(self.episode_config.split)
            self._builders[key] = MarketWindowBuilder(
                instrument, self.dataset.m5(instrument), start, end, self.window_config
            )
        return self._builders[key]

    def _episode_seed(self) -> int:
        return worker_episode_seed(self.global_seed, self.worker_id, self._episode_index)

    def _start_episode(self) -> None:
        spec = self._sampler.sample(1, seed=self._episode_seed())[0]
        env = self._make_env(spec.instrument)
        builder = self._make_builder(spec.instrument)
        obs, _info = env.reset(
            seed=spec.seed,
            instrument=spec.instrument,
            start_index=spec.start_index,
            horizon=spec.horizon,
        )
        window = builder.build(env.current_obs_index)
        last_obs = self.encoder.encode(obs, window).encoded
        self._active = (env, builder, spec, last_obs)

    # -- stepping -------------------------------------------------------------

    def step(self, *, random_action: bool = False) -> Transition:
        if self._active is None:
            self._start_episode()
        assert self._active is not None
        env, builder, _spec, last_obs = self._active

        policy = self._policy
        value_net = self._value_net
        use_policy = policy is not None and not random_action
        if self.algorithm == "ppo":
            use_policy = policy is not None  # PPO is on-policy: no random warmup
        log_prob = 0.0
        value = 0.0
        action_f = 0.0
        action_raw_f = 0.0
        if use_policy and policy is not None:
            # Isolate this worker's policy sampling RNG so the learner's RNG is
            # untouched and the worker is deterministic in the sync backend.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(
                    worker_episode_seed(self.global_seed, self.worker_id, self._total_steps)
                )
                obs_t = torch.as_tensor(last_obs, dtype=torch.float32).unsqueeze(0)
                if self.algorithm == "ppo":
                    from forexmind.training.networks import GaussianPolicy

                    gauss = cast(GaussianPolicy, policy)
                    dist = gauss(obs_t)
                    raw = dist.sample()  # u ~ N(mean, std)
                    action_env = torch.clamp(raw, -1.0, 1.0)  # projection for the env
                    # The stored log-prob must be the density of the ACTUAL
                    # sample ``u`` (the policy is a plain Gaussian over u; the
                    # clamp is only an execution-time projection).  Evaluating
                    # N() at the clamped boundary instead makes old/new
                    # log-probs not correspond to the sampling distribution
                    # and can drive the PPO ratio to diverge.  The env action
                    # stays clamped - the trading semantics are unchanged.
                    log_prob = float(dist.log_prob(raw).sum().item())
                    value = float(value_net(obs_t).item()) if value_net is not None else 0.0
                    action_f = float(action_env.item())
                    action_raw_f = float(raw.item())
                else:
                    action_f = float(sample_action(policy, last_obs, self.algorithm))
        else:
            action_f = float(self._rng.uniform(-1.0, 1.0))

        obs, reward, terminated, truncated, _info = env.step(action_f)
        window = builder.build(env.current_obs_index)
        next_obs = self.encoder.encode(obs, window).encoded
        next_value = 0.0
        if self.algorithm == "ppo" and use_policy and value_net is not None:
            with torch.no_grad():
                nxt_t = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)
                next_value = float(value_net(nxt_t).item())
        t = Transition(
            obs=last_obs,
            action=action_f,
            reward=float(reward),
            next_obs=next_obs,
            terminated=bool(terminated),
            truncated=bool(truncated),
            log_prob=log_prob,
            value=value,
            next_value=next_value,
            worker_id=self.worker_id,
            worker_pid=os.getpid(),
            instrument=str(obs.instrument) if obs.instrument else "",
            timestamp=obs.timestamp,
            action_raw=action_raw_f,
        )
        self._active = (env, builder, _spec, next_obs)
        self._total_steps += 1
        if terminated or truncated:
            self._episode_index += 1
            self._active = None
        return t

    @property
    def completed_episodes(self) -> int:
        return self._episode_index


class SyncCollector:
    """Deterministic in-process collection (single logical worker)."""

    def __init__(self, worker: EnvWorker) -> None:
        self.worker = worker

    def set_policy(self, policy: nn.Module | None, value_net: nn.Module | None = None) -> None:
        self.worker.set_policy(policy, value_net)

    def collect(self, n_steps: int, *, random_action: bool = False) -> list[Transition]:
        return [self.worker.step(random_action=random_action) for _ in range(n_steps)]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Process backend
# ---------------------------------------------------------------------------


def _worker_process_main(cfg: dict[str, Any], q_in: Any, q_out: Any) -> None:  # pragma: no cover
    """Subprocess entry: rebuild worker from a picklable config dict."""
    worker_id = int(cfg["worker_id"])
    print(
        f"[collector-worker] start worker_id={worker_id} pid={os.getpid()} "
        f"parent_pid={os.getppid()}",
        flush=True,
    )
    split_config = SplitConfig.from_dict(cfg["split_config"])
    from forexmind.training.dataset_mmap import resolve_dataset

    dataset, _dataset_backend = resolve_dataset(
        processed_dir=cfg["processed_dir"],
        split_config=split_config,
        instruments=tuple(cfg["instruments"]),
        backend=cfg.get("dataset_backend", "auto"),
    )
    env_config = cfg["env_config"]
    encoder_config = cfg["encoder_config"]
    window_config = cfg["window_config"]
    episode_config = cfg["episode_config"]
    model = cfg["model"]
    algorithm = cfg["algorithm"]
    policy = build_policy_network(
        algorithm,
        cfg["obs_dim"],
        cfg["action_dim"],
        model,
        log_std_min=cfg.get("log_std_min"),
        log_std_max=cfg.get("log_std_max"),
    )
    policy.eval()
    value_net = None
    if algorithm == "ppo":
        from forexmind.training.networks import ValueNet

        value_net = ValueNet(cfg["obs_dim"], model)
        value_net.eval()

    worker = EnvWorker(
        dataset=dataset,
        env_config=env_config,
        encoder_config=encoder_config,
        window_config=window_config,
        episode_config=episode_config,
        algorithm=algorithm,
        model_config=model,
        obs_dim=cfg["obs_dim"],
        action_dim=cfg["action_dim"],
        worker_id=worker_id,
        global_seed=cfg["global_seed"],
        policy=policy,
    )
    worker.set_policy(policy, value_net)
    torch.set_num_threads(1)
    while True:
        msg = q_in.get()
        if msg is None:
            break
        kind = msg[0]
        if kind == "set_policy":
            policy.load_state_dict({k: torch.as_tensor(v) for k, v in msg[1].items()})
            if value_net is not None and len(msg) > 2 and msg[2] is not None:
                value_net.load_state_dict({k: torch.as_tensor(v) for k, v in msg[2].items()})
        elif kind == "collect":
            n = int(msg[1])
            random_action = bool(msg[2])
            out = [worker.step(random_action=random_action) for _ in range(n)]
            q_out.put(out)
        else:  # pragma: no cover
            raise ValueError(f"unknown worker message {kind!r}")
    print(f"[collector-worker] stop worker_id={worker_id} pid={os.getpid()}", flush=True)
    q_out.put(None)


class ProcessCollector:
    """Multiple independent worker processes collecting experience."""

    def __init__(
        self,
        *,
        processed_dir: str,
        split_config: SplitConfig,
        instruments: tuple[str, ...],
        env_config: EnvironmentConfig,
        encoder_config: EncoderConfig,
        window_config: WindowConfig,
        episode_config: EpisodeConfig,
        algorithm: str,
        model: object,
        obs_dim: int,
        action_dim: int,
        global_seed: int,
        num_workers: int,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        dataset_backend: str = "auto",
    ) -> None:
        self.num_workers = max(1, num_workers)
        ctx = mp.get_context("spawn")
        self._q_in: list[Any] = []
        self._q_out: list[Any] = []
        self._procs: list[Any] = []
        base_cfg: dict[str, Any] = {
            "processed_dir": processed_dir,
            "split_config": split_config.to_dict(),
            "instruments": list(instruments),
            "env_config": env_config,
            "encoder_config": encoder_config,
            "window_config": window_config,
            "episode_config": episode_config,
            "algorithm": algorithm,
            "model": model,
            "hidden_dim": int(getattr(model, "hidden_dim", 256)),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "global_seed": global_seed,
            "log_std_min": log_std_min,
            "log_std_max": log_std_max,
            "dataset_backend": dataset_backend,
        }
        for wid in range(self.num_workers):
            q_in = ctx.Queue()
            q_out = ctx.Queue()
            cfg = dict(base_cfg, worker_id=wid)
            p = ctx.Process(target=_worker_process_main, args=(cfg, q_in, q_out))
            p.start()
            self._q_in.append(q_in)
            self._q_out.append(q_out)
            self._procs.append(p)
        print(
            "[collector] process backend started "
            f"configured_workers={self.num_workers} "
            f"alive_workers={self.alive_workers} "
            f"worker_pids={self.worker_pids}",
            flush=True,
        )

    @property
    def worker_pids(self) -> list[int]:
        return [int(p.pid) for p in self._procs if p.pid is not None]

    @property
    def alive_workers(self) -> int:
        return sum(1 for p in self._procs if p.is_alive())

    def worker_status(self) -> list[dict[str, int | bool | None]]:
        return [
            {
                "worker_index": i,
                "pid": int(p.pid) if p.pid is not None else None,
                "alive": bool(p.is_alive()),
                "exitcode": p.exitcode,
            }
            for i, p in enumerate(self._procs)
        ]

    def set_policy(self, policy: nn.Module, value_net: nn.Module | None = None) -> None:
        state = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}
        vstate = None
        if value_net is not None:
            vstate = {k: v.detach().cpu().numpy() for k, v in value_net.state_dict().items()}
        for q in self._q_in:
            q.put(("set_policy", state, vstate))

    def collect(self, n_steps: int, *, random_action: bool = False) -> list[Transition]:
        per = n_steps // self.num_workers
        rem = n_steps % self.num_workers
        for i, q in enumerate(self._q_in):
            q.put(("collect", per + (1 if i < rem else 0), random_action))
        transitions: list[Transition] = []
        for q in self._q_out:
            batch = q.get()
            if batch is not None:
                transitions.extend(batch)
        return transitions

    def close(self) -> None:
        for q in self._q_in:
            q.put(None)
        for p in self._procs:
            p.join(timeout=30)
        print(
            "[collector] process backend stopped "
            f"configured_workers={self.num_workers} "
            f"alive_workers={self.alive_workers} "
            f"worker_pids={self.worker_pids}",
            flush=True,
        )
        for q in self._q_in:
            q.close()
        for q in self._q_out:
            q.close()
