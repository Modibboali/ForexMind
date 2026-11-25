"""MuZero trajectory collection (Stage 4.3; step-driven in Stage 4.6).

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

Stage 4.6 splits collection into two pieces so that *one* episode
implementation serves both the Stage 4.5 single-process loop and the parallel
collectors:

* :class:`EpisodeRun` owns one in-progress episode (its environment, its window
  builder, its recorded arrays, its MCTS object) and exposes
  ``decision_inputs`` / ``record_decision`` / ``advance`` / ``finish``.  A run is
  driven by search *results*, so a caller may produce those results one episode
  at a time or for many episodes at once.
* :class:`MuZeroCollector` drives a single run to completion, exactly as Stage
  4.3 did.

Worker seeding is deterministic and explicit (brief S4)::

    episode_seed(global_seed, worker_rank, episode_index)
    search_seed(global_seed, worker_rank, episode_index)
    decision_seed(search_seed, decision_index)      # when per_decision_seed

``worker_rank = 0`` reproduces the Stage 4.5 expressions bit-for-bit, so the
single-process path is unchanged and a parallel collector can be compared with
it directly (brief S34).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

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
from forexmind.muzero.diagnostics import RootSearchRecord
from forexmind.muzero.inference import apply_action_mask
from forexmind.muzero.profiling import PhaseTimer, phase
from forexmind.muzero.search import MuZeroMCTS
from forexmind.muzero.trajectory import MuZeroTrajectory, TrajectoryMetadata, model_version
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import MarketWindowBuilder, WindowConfig

__all__ = [
    "CollectedTrajectory",
    "CollectionStats",
    "CollectorConfig",
    "EpisodeRun",
    "MuZeroCollector",
    "derive_decision_seed",
    "derive_episode_seed",
    "derive_search_seed",
]

#: Multipliers of the deterministic seed derivation.  The ``worker_rank = 0``
#: case reproduces the Stage 4.3/4.5 expressions exactly.
_EPISODE_SEED_MULT = 7_919
_EPISODE_SEED_WORKER = 51_761
_SEARCH_SEED_MULT = 104_729
_SEARCH_SEED_WORKER = 88_747
_DECISION_SEED_MULT = 15_485_863


def derive_episode_seed(global_seed: int, worker_rank: int, episode_index: int) -> int:
    """Deterministic episode-sampling seed for one collector (brief S4)."""
    return (
        int(global_seed) * 1_000_003
        + int(worker_rank) * _EPISODE_SEED_WORKER
        + int(episode_index) * _EPISODE_SEED_MULT
    ) % (2**31)


def derive_search_seed(global_seed: int, worker_rank: int, episode_index: int) -> int:
    """Deterministic MCTS seed for one collector's episode (brief S4)."""
    return (
        int(global_seed) * 1_000_033
        + int(worker_rank) * _SEARCH_SEED_WORKER
        + int(episode_index) * _SEARCH_SEED_MULT
        + 17
    ) % (2**31)


def derive_decision_seed(search_seed: int, decision_index: int) -> int:
    """Seed one decision's search RNG so results never depend on scheduling."""
    return (
        int(search_seed) * 1_000_003 + int(decision_index) * _DECISION_SEED_MULT + 7
    ) % (2**31)


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
    #: Integrated-loop model version (Stage 4.5) recorded on every trajectory.
    network_version: int = 0
    #: Capture per-root search diagnostics (Stage 4.5) in addition to the
    #: compact replay arrays.  Off by default: the trace is for the training
    #: loop, never for replay storage.
    capture_diagnostics: bool = False
    #: Re-seed the search RNG for every decision (Stage 4.6).  Required for a
    #: parallel collector to be reproducible independently of scheduling, and
    #: used by the single-vs-parallel equivalence test.  ``False`` keeps the
    #: Stage 4.5 behaviour of one RNG stream per episode.
    per_decision_seed: bool = False

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
            "network_version": self.network_version,
            "capture_diagnostics": self.capture_diagnostics,
            "per_decision_seed": self.per_decision_seed,
        }


@dataclass(slots=True)
class CollectedTrajectory:
    """One collected trajectory plus the optional per-root search trace."""

    trajectory: MuZeroTrajectory
    records: list[RootSearchRecord] = field(default_factory=list)


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

    def absorb(self, other: CollectionStats) -> None:
        """Merge another stats record into this one (used by collector pools)."""
        self.trajectories += other.trajectories
        self.env_steps += other.env_steps
        self.searches += other.searches
        self.recurrent_inference_calls += other.recurrent_inference_calls
        self.terminated += other.terminated
        self.truncated += other.truncated
        self.reward_sum += other.reward_sum
        self.action_counts += other.action_counts


class EpisodeRun:
    """One in-progress episode, driven by externally supplied search results.

    The run owns everything that is episode-specific (environment, window
    builder, recorded arrays, MCTS object, RNG stream).  Splitting the episode
    from the search loop is what lets the parallel collectors drive several
    episodes in lock-step through one batched search (brief S14) while sharing
    exactly the same trajectory construction as the single-process collector
    (brief S34).
    """

    def __init__(
        self,
        *,
        spec: EpisodeSpec,
        env: ForexEnvironment,
        builder: MarketWindowBuilder,
        encoder: ObservationEncoder,
        search: MuZeroMCTS,
        config: CollectorConfig,
        episode_index: int,
        search_seed: int,
        model_version_string: str,
        network_version: int | None = None,
        timer: Any | None = None,
    ) -> None:
        self.spec = spec
        self.env = env
        self.builder = builder
        self.encoder = encoder
        self.search = search
        self.config = config
        self.episode_index = int(episode_index)
        self.search_seed = int(search_seed)
        self.model_version = str(model_version_string)
        self.timer = timer

        observation, _info = env.reset(
            seed=spec.seed,
            instrument=spec.instrument,
            start_index=spec.start_index,
            horizon=spec.horizon,
        )
        self.observations: list[np.ndarray] = [self._encode(observation)]
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.policies: list[np.ndarray] = []
        self.values: list[float] = []
        self.masks: list[np.ndarray] = []
        self.terminated: list[bool] = []
        self.truncated: list[bool] = []
        #: Planning states read from the *live* account (ground truth), so the
        #: stored metadata can never disagree with the real position.
        self.planning: list[PlanningState] = [PlanningState.from_env(env)]
        self.records: list[RootSearchRecord] = []
        self.stats = CollectionStats()
        self.network_version = int(
            config.network_version if network_version is None else network_version
        )
        #: Root visit temperature in force for this episode's decisions.
        self.temperature = float(config.effective_temperature)
        self.boundary_value = 0.0
        self.trajectory: MuZeroTrajectory | None = None
        self.done = False
        self.decision_index = 0
        self._pending: tuple[int, np.ndarray, np.ndarray, float] | None = None

    # -- encoding -------------------------------------------------------------

    def _encode(self, observation: Any) -> np.ndarray:
        with phase(self.timer, "observation_encode"):
            window = self.builder.build(self.env.current_obs_index)
            return np.asarray(self.encoder.encode(observation, window).encoded, dtype=np.float32)

    @property
    def observation(self) -> np.ndarray:
        """Observation before the next decision (state ``s_t``)."""
        return self.observations[-1]

    @property
    def planning_state(self) -> PlanningState:
        return self.planning[-1]

    @property
    def mask(self) -> np.ndarray:
        """Current real-environment action mask, projected onto MuZero's six."""
        return project_action_mask(np.asarray(self.env.action_masks(), dtype=bool))

    # -- decision plumbing ----------------------------------------------------

    def decision_inputs(self) -> tuple[np.ndarray, np.ndarray, PlanningState]:
        """``(observation, mask, planning_state)`` for the next decision."""
        obs = self.observation
        mask = self.mask
        state = self.planning_state
        if not np.array_equal(state.action_mask(), mask):
            raise RuntimeError("account planning state disagrees with the environment mask")
        return obs, mask, state

    def reseed_search(self, *, boundary: bool = False) -> None:
        """Give the search a decision-local RNG stream (brief S4).

        With ``per_decision_seed`` the noise of decision ``t`` depends only on
        ``(global_seed, worker_rank, episode_index, t)``, never on how the
        collectors were scheduled.
        """
        if not self.config.per_decision_seed:
            return
        index = self.decision_index if not boundary else self.decision_index + 1_000_003
        self.search.rng = np.random.default_rng(derive_decision_seed(self.search_seed, index))

    def record_decision(
        self,
        result: Any,
        mask: np.ndarray,
        *,
        network_version: int | None = None,
        temperature: float | None = None,
    ) -> int:
        """Store one search result; :meth:`advance` then applies it for real."""
        action = int(result.action)
        if not bool(mask[action]):
            raise RuntimeError(f"search selected an invalid action {action}")
        self._pending = (
            action,
            np.asarray(mask, dtype=bool).copy(),
            np.asarray(result.policy, dtype=np.float32),
            float(result.root_value),
        )
        if self.config.capture_diagnostics:
            self.records.append(self._root_record(result, mask, action))
        if network_version is not None:
            self.network_version = int(network_version)
        if temperature is not None:
            self.temperature = float(temperature)
        return action

    def advance(self) -> bool:
        """Step the real environment with the recorded action; return ``done``."""
        if self._pending is None:
            raise RuntimeError("advance() requires a recorded decision")
        action, mask, policy, value = self._pending
        with phase(self.timer, "environment_step"):
            next_observation, reward, term, trunc, _info = self.env.step(
                env_action_index(action)
            )
        next_encoded = self._encode(next_observation)
        if self.records:
            self.records[-1].real_reward = float(reward)

        self.observations.append(next_encoded)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.policies.append(policy)
        self.values.append(value)
        self.masks.append(mask)
        self.terminated.append(bool(term))
        self.truncated.append(bool(trunc))
        self.planning.append(PlanningState.from_env(self.env))

        self.stats.env_steps += 1
        self.stats.action_counts[action] += 1
        self.stats.reward_sum += float(reward)
        self.decision_index += 1
        self._pending = None
        self.done = bool(term or trunc)
        if self.done:
            if self.terminated[-1]:
                self.stats.terminated += 1
            else:
                self.stats.truncated += 1
        return self.done

    # -- boundary value -------------------------------------------------------

    def needs_boundary_search(self) -> bool:
        """``True`` when the final observation needs a bootstrap MCTS value."""
        return bool(self.truncated and self.truncated[-1]) and bool(self.config.boundary_search)

    def boundary_request(self) -> tuple[np.ndarray, np.ndarray, PlanningState] | None:
        """Inputs for the boundary search, or ``None`` when none is needed."""
        if not self.needs_boundary_search():
            return None
        state = self.planning[-1]
        return self.observations[-1], state.action_mask(), state

    def apply_boundary_result(self, result: Any) -> float:
        """Record a boundary search result (``add_root_noise=False``)."""
        self.boundary_value = float(result.root_value)
        self.stats.searches += 1
        self.stats.recurrent_inference_calls += int(
            result.diagnostics.recurrent_inference_calls
        )
        return self.boundary_value

    def resolve_boundary(self) -> float:
        """Run the boundary search in-process (single-collector path)."""
        request = self.boundary_request()
        if request is None:
            return 0.0
        observation, mask, state = request
        self.reseed_search(boundary=True)
        result = self.search.search(
            observation, mask, planning_state=state, add_root_noise=False
        )
        return self.apply_boundary_result(result)

    # -- completion -----------------------------------------------------------

    def finish(self, boundary_value: float | None = None) -> CollectedTrajectory:
        """Build the validated trajectory for this episode."""
        if not self.done:
            raise RuntimeError("cannot finish an episode that has not terminated")
        value = self.boundary_value if boundary_value is None else float(boundary_value)
        if self.terminated[-1]:
            value = 0.0
        trajectory = MuZeroTrajectory(
            observations=np.asarray(self.observations, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.int64),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            root_policies=np.asarray(self.policies, dtype=np.float32),
            root_values=np.asarray(self.values, dtype=np.float32),
            action_masks=np.asarray(self.masks, dtype=bool),
            terminated=np.asarray(self.terminated, dtype=bool),
            truncated=np.asarray(self.truncated, dtype=bool),
            # float64, not float32: the six-action mask is derived from the
            # exposure with a 1e-3 tolerance, and a float32 round trip can flip
            # a mask for a state that sits within ~1e-7 of the boundary.  The
            # stored planning state must reproduce the masks exactly.
            planning_exposure=np.asarray([s.exposure for s in self.planning], dtype=np.float64),
            planning_is_flat=np.asarray([s.is_flat for s in self.planning], dtype=bool),
            boundary_value=float(value),
            metadata=TrajectoryMetadata(
                trajectory_id=self.episode_index,
                instrument=self.spec.instrument,
                split=self.spec.split,
                start_index=int(self.spec.start_index),
                horizon=int(self.spec.horizon),
                episode_seed=int(self.spec.seed),
                search_seed=int(self.search_seed),
                model_version=self.model_version,
                num_simulations=int(self.config.num_simulations),
                discount=float(self.config.discount),
                temperature=float(self.temperature),
                num_steps=len(self.actions),
                training=bool(self.config.training),
                network_version=int(self.network_version),
            ),
            extra={"planning_chain_disagreements": self.chain_disagreements()},
        )
        trajectory.validate()
        self.stats.trajectories += 1
        self.trajectory = trajectory
        return CollectedTrajectory(trajectory=trajectory, records=list(self.records))

    def chain_disagreements(self) -> int:
        """How often the deterministic ``after(action)`` rule differs from reality."""
        return sum(
            1
            for t, action in enumerate(self.actions)
            if self.planning[t + 1] != self.planning[t].after(action)
        )

    def _root_record(
        self, result: Any, mask: np.ndarray, action: int
    ) -> RootSearchRecord:
        """Noise-free network prior + MCTS outcome for one real decision."""
        with torch.no_grad():
            output = self.search.backend.initial_inference(self.observation, mask)
            prior = torch.softmax(apply_action_mask(output.policy_logits, mask), dim=-1)[0]
        visits = np.asarray(result.visit_counts, dtype=np.float64)
        total = float(visits.sum())
        policy = visits / total if total > 0.0 else np.full_like(visits, 0.0)
        diagnostics = result.diagnostics
        return RootSearchRecord(
            prior=prior.detach().cpu().numpy().astype(np.float64),
            visits=visits,
            policy=policy,
            q_values=np.asarray(diagnostics.q_values, dtype=np.float64),
            predicted_rewards=np.asarray(diagnostics.predicted_rewards, dtype=np.float64),
            mask=np.asarray(mask, dtype=bool).copy(),
            action=int(action),
            network_value=float(diagnostics.root_predicted_value),
            search_value=float(result.root_value),
            tree_depth=int(diagnostics.tree_depth),
        )


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
        model_version_string: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.env_config = env_config
        self.encoder_config = encoder_config
        self.model = model
        self.config = config or CollectorConfig()
        self.stats = CollectionStats()
        #: Fingerprint of the weights used for search.  Collector workers with
        #: remote inference have no local parameters, so the trainer sends the
        #: fingerprint with every weight synchronization (brief S16).
        self.model_version = (
            model_version(model) if model_version_string is None else str(model_version_string)
        )
        #: Optional wall-time instrumentation (Stage 4.6 profiling, brief S2).
        #: Set by the trainer; shared by every search this collector creates.
        self.timer: PhaseTimer | None = None

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
            self._envs[instrument] = self._new_env(instrument)
        return self._envs[instrument]

    def _new_env(self, instrument: str) -> ForexEnvironment:
        """A private environment for one run (never shared between collectors).

        The market data behind it is shared read-only; only the *stateful*
        environment wrapper is new.  Concurrent episodes inside one worker must
        never step the same environment object (brief S3).
        """
        from forexmind.data.dataset import MarketDataset

        market = MarketDataset()
        market.add(self.dataset.load(instrument))
        return ForexEnvironment(market, self.env_config, instrument=instrument)

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
        """Deterministic episode-sampling seed (worker rank 0: Stage 4.5 value)."""
        return derive_episode_seed(self.config.seed, 0, index)

    def search_seed(self, index: int) -> int:
        """Deterministic MCTS seed (worker rank 0: Stage 4.5 value)."""
        return derive_search_seed(self.config.seed, 0, index)

    def make_search(self, search_seed: int, *, backend: Any | None = None) -> MuZeroMCTS:
        """Build the search object for one episode.

        ``backend`` lets a parallel collector route inference to the central
        service; ``None`` keeps the Stage 4.5 in-process behaviour.
        """
        return MuZeroMCTS(
            self.model,
            self.search_config,
            rng=np.random.default_rng(search_seed),
            backend=backend,
            timer=self.timer,
        )

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

    def start_run(
        self,
        index: int,
        *,
        search: MuZeroMCTS | None = None,
        backend: Any | None = None,
        worker_rank: int = 0,
        network_version: int | None = None,
        fresh_env: bool = False,
    ) -> EpisodeRun:
        """Open a new :class:`EpisodeRun` for episode ``index``.

        ``worker_rank`` identifies the collector inside a pool and only shifts
        the derived seeds; rank ``0`` reproduces the Stage 4.5 episode exactly.
        """
        if worker_rank:
            spec = self.sampler.sample(
                1,
                seed=derive_episode_seed(self.config.seed, worker_rank, index),
                instruments=self._instruments,
            )[0]
            if spec.split != self.config.split:
                raise ValueError(
                    f"sampler returned split {spec.split!r} but collector is configured for "
                    f"{self.config.split!r}"
                )
        else:
            spec = self.sample_spec(index)
        search_seed = (
            self.search_seed(index)
            if not worker_rank
            else derive_search_seed(self.config.seed, worker_rank, index)
        )
        return EpisodeRun(
            spec=spec,
            env=(
                self._new_env(spec.instrument)
                if fresh_env
                else self._make_env(spec.instrument)
            ),
            builder=self._make_builder(spec.instrument),
            encoder=self.encoder,
            search=search or self.make_search(search_seed, backend=backend),
            config=self.config,
            episode_index=index,
            search_seed=search_seed,
            model_version_string=self.model_version,
            network_version=network_version,
            timer=self.timer,
        )

    def collect_trajectory(self, index: int) -> MuZeroTrajectory:
        """Collect exactly one trajectory for episode ``index``."""
        return self.collect_with_diagnostics(index).trajectory

    def collect_with_diagnostics(self, index: int) -> CollectedTrajectory:
        """Collect one trajectory plus its per-root search trace.

        The trace (Stage 4.5 S16-S22) is *not* stored in replay: it is returned
        to the caller, aggregated into training diagnostics and discarded, so
        replay stays as compact as Stage 4.3 made it.
        """
        run = self.start_run(index)
        while not run.done:
            observation, mask, state = run.decision_inputs()
            run.reseed_search()
            result = run.search.search(
                observation,
                mask,
                planning_state=state,
                add_root_noise=self.config.training,
            )
            run.stats.searches += 1
            run.stats.recurrent_inference_calls += int(
                result.diagnostics.recurrent_inference_calls
            )
            run.record_decision(result, mask)
            run.advance()
        run.resolve_boundary()
        collected = run.finish()
        self.stats.absorb(run.stats)
        return collected

    def collect(self, count: int) -> list[MuZeroTrajectory]:
        """Collect ``count`` trajectories, increasing the episode counter."""
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        trajectories = []
        for _ in range(count):
            trajectories.append(self.collect_trajectory(self._episode_index))
            self._episode_index += 1
        return trajectories

    def collect_next_with_diagnostics(self) -> CollectedTrajectory:
        """Collect the next episode (advancing the counter) with its search trace."""
        collected = self.collect_with_diagnostics(self._episode_index)
        self._episode_index += 1
        return collected

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
