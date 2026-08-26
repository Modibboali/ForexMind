"""Evaluation runner (Phase 2).

A single reusable evaluator drives every baseline through the identical
protocol::

    reset agent -> reset env -> observe -> act -> step -> record

The evaluator never touches internal environment state except through the
documented ``info`` interface.  It builds causal encoded observations from the
market window + account/time state, so agents only ever see the current
observation.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from forexmind.baselines.base import TradingAgent
from forexmind.config import EnvironmentConfig
from forexmind.data.dataset import MarketDataset
from forexmind.data.splits import SplitDataset
from forexmind.environment import ForexEnvironment
from forexmind.episodes.sampler import EpisodeSpec
from forexmind.episodes.trajectory import Trajectory
from forexmind.evaluation.metrics import compute_metrics
from forexmind.observation.encoder import ObservationEncoder
from forexmind.observation.window import MarketWindowBuilder, WindowConfig


def _to_float(value: object) -> float:
    """Coerce a JSON-like ``info`` value (Decimal/float) to float."""
    return float(value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation-level configuration (serializable)."""

    split: str = "test"
    episodes: int = 100
    seed: int = 42
    horizon: int = 512
    context_length: int = 64
    periods_per_year: float | None = None  # None = "auto" from split data
    save_curves: bool = True  # store equity/timestamps in reports

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "episodes": self.episodes,
            "seed": self.seed,
            "horizon": self.horizon,
            "context_length": self.context_length,
            "periods_per_year": self.periods_per_year,
            "save_curves": self.save_curves,
        }


@dataclass
class AgentEvaluation:
    """Results of one agent over a set of episodes."""

    agent_name: str
    config: object
    trajectories_by_instrument: dict[str, list[Trajectory]] = field(default_factory=dict)
    wall_seconds: float = 0.0
    steps_per_second: float = 0.0
    seed: int | None = None  # agent-level seed (e.g. per-seed random runs)

    def n_episodes(self) -> int:
        return sum(len(v) for v in self.trajectories_by_instrument.values())


class EvaluationRunner:
    """Reusable evaluator.  Environment and window builders are cached per
    instrument so large datasets are loaded once."""

    def __init__(
        self,
        dataset: SplitDataset,
        env_config: EnvironmentConfig,
        encoder: ObservationEncoder,
        window_config: WindowConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.env_config = env_config
        self.encoder = encoder
        self.window_config = window_config or WindowConfig(
            context_length=encoder.config.context_length
        )
        self._envs: dict[str, ForexEnvironment] = {}
        self._builders: dict[tuple[str, str], MarketWindowBuilder] = {}
        self._ppy_cache: dict[str, float] = {}

    # -- factories ------------------------------------------------------------

    def make_env(self, instrument: str) -> ForexEnvironment:
        if instrument not in self._envs:
            data = self.dataset.load(instrument)
            ds = MarketDataset()
            ds.add(data)
            self._envs[instrument] = ForexEnvironment(ds, self.env_config, instrument=instrument)
        return self._envs[instrument]

    def window_builder(self, instrument: str, split: str) -> MarketWindowBuilder:
        key = (instrument, split)
        if key not in self._builders:
            data = self.dataset.load(instrument)
            start, end = self.dataset.split_config.range(split)
            self._builders[key] = MarketWindowBuilder(
                instrument, data.m5, start, end, self.window_config
            )
        return self._builders[key]

    def periods_per_year(self, split: str) -> float:
        if split not in self._ppy_cache:
            self._ppy_cache[split] = self.dataset.periods_per_year(split)
        return self._ppy_cache[split]

    # -- episode execution ----------------------------------------------------

    def run_episode(self, agent: TradingAgent, spec: EpisodeSpec) -> Trajectory:
        env = self.make_env(spec.instrument)
        builder = self.window_builder(spec.instrument, spec.split)
        agent.reset(spec.seed)

        obs, info = env.reset(
            seed=spec.seed,
            instrument=spec.instrument,
            start_index=spec.start_index,
            horizon=spec.horizon,
        )
        initial_equity = _to_float(info["equity"])
        initial_balance = float(self.env_config.margin.initial_balance)

        timestamps: list[np.datetime64] = []
        actions: list[float] = []
        rewards: list[float] = []
        log_returns: list[float] = []
        equity = [initial_equity]
        positions: list[float] = []
        trade_log: list[dict[str, object]] = []
        prev_units = _to_float(info["position_units"])

        for step in range(spec.horizon):
            window = builder.build(env.current_obs_index)
            encoded = self.encoder.encode(obs, window)
            action = agent.act(encoded)
            obs, reward, terminated, truncated, info = env.step(action.target_exposure)

            new_equity = _to_float(info["equity"])
            equity.append(new_equity)
            rewards.append(float(reward))
            timestamps.append(np.datetime64(obs.timestamp))
            actions.append(float(action.target_exposure))
            log_returns.append(math.log(new_equity / equity[-2]) if equity[-2] > 0 else 0.0)
            positions.append(_to_float(info["position_units"]))

            if _to_float(info.get("units_delta", 0.0)) != 0.0:
                trade_log.append(
                    {
                        "timestamp": str(obs.timestamp),
                        "instrument": spec.instrument,
                        "old_position": prev_units,
                        "new_position": _to_float(info["position_units"]),
                        "execution_price": (
                            str(info["execution_price"])
                            if info.get("execution_price") is not None
                            else None
                        ),
                        "units_delta": _to_float(info["units_delta"]),
                        "cost": _to_float(info.get("trade_cost", 0.0)),
                        "realized_pnl": _to_float(info.get("trade_realized_pnl", 0.0)),
                        "equity_after_trade": new_equity,
                        "step": step,
                    }
                )
            prev_units = _to_float(info["position_units"])
            if terminated or truncated:
                break

        traj = Trajectory(
            agent_name=agent.name,
            spec=spec,
            timestamps=np.asarray(timestamps, dtype="datetime64[ns]"),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float64),
            equity=np.asarray(equity, dtype=np.float64),
            log_returns=np.asarray(log_returns, dtype=np.float64),
            position_units=np.asarray(positions, dtype=np.float64),
            trade_log=trade_log,
            info={
                "instrument": spec.instrument,
                "split": spec.split,
                "initial_balance": initial_balance,
                "final_equity": float(equity[-1]),
                "n_steps": len(rewards),
            },
        )
        periods = self.periods_per_year(spec.split)
        traj.metrics = compute_metrics(
            equity=traj.equity,
            log_returns=traj.log_returns,
            position_units=traj.position_units,
            trade_log=traj.trade_log,
            initial_balance=initial_balance,
            periods_per_year=periods,
        )
        return traj

    # -- agent-level evaluation -----------------------------------------------

    def run_agent(
        self,
        agent: TradingAgent,
        specs: list[EpisodeSpec],
    ) -> AgentEvaluation:
        """Run one agent over ``specs``; results grouped by instrument."""
        start = time.perf_counter()
        grouped: dict[str, list[Trajectory]] = defaultdict(list)
        total_steps = 0
        for spec in specs:
            traj = self.run_episode(agent, spec)
            grouped[spec.instrument].append(traj)
            total_steps += traj.n_steps
        wall = time.perf_counter() - start
        return AgentEvaluation(
            agent_name=agent.name,
            config=getattr(agent, "config", None),
            trajectories_by_instrument=dict(grouped),
            wall_seconds=wall,
            steps_per_second=total_steps / wall if wall > 0 else 0.0,
        )


def evaluate(
    agent: TradingAgent,
    runner: EvaluationRunner,
    episode_specs: list[EpisodeSpec],
) -> AgentEvaluation:
    """Convenience wrapper matching the Phase-2 spec's ``evaluate(...)``."""
    return runner.run_agent(agent, episode_specs)
