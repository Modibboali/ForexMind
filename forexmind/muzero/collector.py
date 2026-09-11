"""MuZero trajectory collection (Stage 4.3).

Drives the real Forex environment with Stage 4.2 MCTS and records correctly
indexed trajectories::

    obs, info = env.reset(...)
    while not done:
        mask   = env.action_masks()                       # real environment mask
        result = mcts.search(obs, mask, planning_state=...)   # training search
        action = result.action                            # MCTS visit distribution
        next_obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(...)
        obs = next_obs

Only **real** environment transitions are recorded.  Imagined MCTS transitions
stay inside search and are never written to a trajectory.

Split protection: the collector only ever uses the split it was configured with,
and :meth:`TrajectoryReplayBuffer.add` independently refuses non-train
trajectories, so validation/test data cannot leak into training replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forexmind.data.splits import SPLIT_NAMES, SplitDataset
from forexmind.environment.forex_env import ForexEnvironment
from forexmind.episodes.config import EpisodeConfig
from forexmind.episodes.sampler import EpisodeSampler, EpisodeSpec
from forexmind.muzero.actions import (
    MUZERO_NUM_ACTIONS,
    PlanningState,
    env_action_index,
    project_action_mask,
)
from forexmind.muzero.config import SearchConfig
from forexmind.muzero.search import MuZeroMCTS
from forexmind.muzero.trajectory import MuZeroTrajectory, TrajectoryMetadata, model_version
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import MarketWindowBuilder, WindowConfig

__all__ = ["CollectionStats", "CollectorConfig", "MuZeroCollector"]


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    """Everything that controls trajectory generation.

    ``training=True`` enables root Dirichlet noise and a non-zero sampling
    temperature; ``training=False`` gives deterministic evaluation collection
    (noise off, temperature 0, greedy argmax over visit counts).
    """

    split: str = "train"
    horizon: int = 64
    num_simulations: int = 16
    discount: float = 0.99
    training: bool = True
    temperature: float | None = None  # None -> 1.0 when training, 0.0 otherwise
    dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    seed: int = 0
    boundary_search: bool = True  # run one extra search for truncated final states

    def __post_init__(self) -> None:
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {self.split!r}; expected {list(SPLIT_NAMES)}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {self.num_simulations}")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {self.discount}")
        if self.temperature is not None and self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0 or None, got {self.temperature}")

    @property
    def effective_temperature(self) -> float:
        if self.temperature is not None:
            return float(self.temperature)
        return 1.0 if self.training else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "horizon": self.horizon,
            "num_simulations": self.num_simulations,
            "discount": self.discount,
            "training": self.training,
            "temperature": self.effective_temperature,
            "dirichlet_alpha": self.dirichlet_alpha,
            "root_exploration_fraction": self.root_exploration_fraction,
            "seed": self.seed,
            "boundary_search": self.boundary_search,
        }


@dataclass(slots=True)
class CollectionStats:
    """Aggregate counters for one or more collection runs."""

    trajectories: int = 0
    env_steps: int = 0
    searches: int = 0
    recurrent_inference_calls: int = 0
    terminated: int = 0
    truncated: int = 0
    reward_sum: float = 0.0
    action_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(MUZERO_NUM_ACTIONS, dtype=np.int64)
    )

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.env_steps if self.env_steps else 0.0

    @property
    def mean_length(self) -> float:
        return self.env_steps / self.trajectories if self.trajectories else 0.0

    def action_frequencies(self) -> np.ndarray:
        total = int(self.action_counts.sum())
        if total == 0:
            return np.zeros(MUZERO_NUM_ACTIONS, dtype=np.float64)
        return self.action_counts.astype(np.float64) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectories": self.trajectories,
            "env_steps": self.env_steps,
            "searches": self.searches,
            "recurrent_inference_calls": self.recurrent_inference_calls,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "mean_length": self.mean_length,
            "reward_sum": self.reward_sum,
            "mean_reward": self.mean_reward,
            "action_counts": self.action_counts.astype(int).tolist(),
            "action_frequencies": self.action_frequencies().tolist(),
        }


class MuZeroCollector:
    """Collects real MuZero trajectories by driving the environment with MCTS."""

    def __init__(
        self,
        dataset: SplitDataset,
        env_config: Any,
        encoder_config: EncoderConfig,
        model: Any,
        config: CollectorConfig | None = None,
        *,
        instruments: tuple[str, ...] | None = None,
    ) -> None:
        self.dataset = dataset
        self.env_config = env_config
        self.encoder_config = encoder_config
        self.model = model
        self.config = config or CollectorConfig()
        self.stats = CollectionStats()
        self.model_version = model_version(model)

        self.encoder = ObservationEncoder(encoder_config)
        self.window_config = WindowConfig(context_length=encoder_config.context_length)
        self.episode_config = EpisodeConfig(
            split=self.config.split,
            horizon=self.config.horizon,
            context_length=encoder_config.context_length,
            seed=self.config.seed,
        )
        self.sampler = EpisodeSampler(dataset, self.episode_config)
        self._instruments = tuple(instruments) if instruments else tuple(dataset.instruments)
        self._envs: dict[str, ForexEnvironment] = {}
        self._builders: dict[str, MarketWindowBuilder] = {}
        self._episode_index = 0

        self.search_config = SearchConfig(
            num_simulations=self.config.num_simulations,
            discount=self.config.discount,
            add_root_noise=self.config.training,
            temperature=self.config.effective_temperature,
            seed=self.config.seed,
            root_dirichlet_alpha=self.config.dirichlet_alpha,
            root_exploration_fraction=self.config.root_exploration_fraction,
        )

    # -- environment plumbing -------------------------------------------------

    def _make_env(self, instrument: str) -> ForexEnvironment:
        if instrument not in self._envs:
            from forexmind.data.dataset import MarketDataset

            market = MarketDataset()
            market.add(self.dataset.load(instrument))
            self._envs[instrument] = ForexEnvironment(
                market, self.env_config, instrument=instrument
            )
        return self._envs[instrument]

    def _make_builder(self, instrument: str) -> MarketWindowBuilder:
        if instrument not in self._builders:
            start, end = self.dataset.split_config.range(self.config.split)
            self._builders[instrument] = MarketWindowBuilder(
                instrument, self.dataset.m5(instrument), start, end, self.window_config
            )
        return self._builders[instrument]

    def _encode(
        self, observation: Any, builder: MarketWindowBuilder, env: ForexEnvironment
    ) -> np.ndarray:
        window = builder.build(env.current_obs_index)
        return np.asarray(self.encoder.encode(observation, window).encoded, dtype=np.float32)

    # -- episode specification ------------------------------------------------

    def episode_seed(self, index: int) -> int:
        return (self.config.seed * 1_000_003 + index * 7_919) % (2**31)

    def search_seed(self, index: int) -> int:
        return (self.config.seed * 1_000_033 + index * 104_729 + 17) % (2**31)

    def sample_spec(self, index: int) -> EpisodeSpec:
        spec = self.sampler.sample(1, seed=self.episode_seed(index), instruments=self._instruments)[
            0
        ]
        if spec.split != self.config.split:
            raise ValueError(
                f"sampler returned split {spec.split!r} but collector is configured for "
                f"{self.config.split!r}"
            )
        return spec

    # -- collection -----------------------------------------------------------

    def collect_trajectory(self, index: int) -> MuZeroTrajectory:
        """Collect exactly one trajectory for episode ``index``."""
        spec = self.sample_spec(index)
        env = self._make_env(spec.instrument)
        builder = self._make_builder(spec.instrument)
        observation, _info = env.reset(
            seed=spec.seed,
            instrument=spec.instrument,
            start_index=spec.start_index,
            horizon=spec.horizon,
        )
        encoded = self._encode(observation, builder, env)

        search_seed = self.search_seed(index)
        search = MuZeroMCTS(self.model, self.search_config, rng=np.random.default_rng(search_seed))

        observations = [encoded]
        actions: list[int] = []
        rewards: list[float] = []
        policies: list[np.ndarray] = []
        values: list[float] = []
        masks: list[np.ndarray] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        # Planning states are read from the live account (ground truth), so the
        # stored metadata can never disagree with the real position.
        planning: list[PlanningState] = [PlanningState.from_env(env)]

        done = False
        while not done:
            mask = project_action_mask(np.asarray(env.action_masks(), dtype=bool))
            state = planning[-1]
            if not np.array_equal(state.action_mask(), mask):
                raise RuntimeError("account planning state disagrees with the environment mask")

            result = search.search(
                encoded,
                mask,
                planning_state=state,
                add_root_noise=self.config.training,
            )
            self.stats.searches += 1
            self.stats.recurrent_inference_calls += result.diagnostics.recurrent_inference_calls
            action = int(result.action)
            if not bool(mask[action]):
                raise RuntimeError(f"search selected an invalid action {action}")

            next_observation, reward, term, trunc, _step_info = env.step(env_action_index(action))
            next_encoded = self._encode(next_observation, builder, env)

            observations.append(next_encoded)
            actions.append(action)
            rewards.append(float(reward))
            policies.append(np.asarray(result.policy, dtype=np.float32))
            values.append(float(result.root_value))
            masks.append(mask.copy())
            terminated.append(bool(term))
            truncated.append(bool(trunc))
            planning.append(PlanningState.from_env(env))

            self.stats.env_steps += 1
            self.stats.action_counts[action] += 1
            self.stats.reward_sum += float(reward)
            done = bool(term or trunc)
            encoded = next_encoded

        if terminated[-1]:
            self.stats.terminated += 1
        else:
            self.stats.truncated += 1

        boundary_value = self._boundary_value(
            search, encoded, env, planning[-1], terminated[-1], truncated[-1]
        )

        trajectory = MuZeroTrajectory(
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            root_policies=np.asarray(policies, dtype=np.float32),
            root_values=np.asarray(values, dtype=np.float32),
            action_masks=np.asarray(masks, dtype=bool),
            terminated=np.asarray(terminated, dtype=bool),
            truncated=np.asarray(truncated, dtype=bool),
            planning_exposure=np.asarray([s.exposure for s in planning], dtype=np.float32),
            planning_is_flat=np.asarray([s.is_flat for s in planning], dtype=bool),
            boundary_value=float(boundary_value),
            metadata=TrajectoryMetadata(
                trajectory_id=index,
                instrument=spec.instrument,
                split=spec.split,
                start_index=int(spec.start_index),
                horizon=int(spec.horizon),
                episode_seed=int(spec.seed),
                search_seed=int(search_seed),
                model_version=self.model_version,
                num_simulations=int(self.config.num_simulations),
                discount=float(self.config.discount),
                temperature=float(self.config.effective_temperature),
                num_steps=len(actions),
                training=bool(self.config.training),
            ),
            extra={"planning_chain_disagreements": self._chain_disagreements(actions, planning)},
        )
        trajectory.validate()
        self.stats.trajectories += 1
        return trajectory

    @staticmethod
    def _chain_disagreements(actions: list[int], planning: list[PlanningState]) -> int:
        """How often the deterministic ``after(action)`` rule differs from reality.

        The rule assumes the account reaches the target exposure exactly; under
        mark-to-market drift it can differ by more than the masking tolerance.
        Recorded as a diagnostic, never treated as corruption.
        """
        return sum(
            1 for t, action in enumerate(actions) if planning[t + 1] != planning[t].after(action)
        )

    def _boundary_value(
        self,
        search: MuZeroMCTS,
        final_observation: np.ndarray,
        env: ForexEnvironment,
        actual_final_state: PlanningState,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Bootstrap value for the final stored observation.

        A true terminal is worth exactly ``0``.  A truncated episode continues,
        so its boundary value is a real MCTS root value at the final
        observation — never an arbitrary substituted neural value.  The mask for
        that search comes from the *actual* account position so the boundary
        state cannot disagree with the environment.
        """
        if terminated or not truncated or not self.config.boundary_search:
            return 0.0
        mask = actual_final_state.action_mask()
        result = search.search(
            final_observation,
            mask,
            planning_state=actual_final_state,
            add_root_noise=False,
        )
        self.stats.searches += 1
        self.stats.recurrent_inference_calls += result.diagnostics.recurrent_inference_calls
        return float(result.root_value)

    def collect(self, count: int) -> list[MuZeroTrajectory]:
        """Collect ``count`` trajectories, increasing the episode counter."""
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        trajectories = []
        for _ in range(count):
            trajectories.append(self.collect_trajectory(self._episode_index))
            self._episode_index += 1
        return trajectories

    def collect_into(
        self,
        replay: Any,
        count: int,
        *,
        require_split: str | None = "train",
    ) -> CollectionStats:
        """Collect ``count`` trajectories straight into a replay buffer."""
        for trajectory in self.collect(count):
            replay.add(trajectory, require_split=require_split)
        return self.stats
