"""Experiment configuration for ForexMind Phase 3 training (YAML-serializable).

Separates defaults from experiment parameters.  ``ExperimentConfig`` is the
single serializable object for a training run; it is persisted with every
checkpoint, manifest, and report so runs are reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def _filter_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


@dataclass
class AlgorithmConfig:
    name: str = "sac"  # "sac" | "ppo"


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "relu"  # relu | tanh


@dataclass
class TrainingConfig:
    total_env_steps: int = 1_000_000
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    warmup_steps: int = 10_000  # random actions before learning starts (SAC)
    gradient_updates_per_step: int = 1  # learner gradient steps per env step
    collect_batch: int = 2048  # transitions per collection round
    gamma: float = 0.99
    tau: float = 0.005  # SAC target soft-update coefficient
    learning_rate: float = 3e-4
    alpha: float | str = "auto"  # SAC entropy coefficient; "auto" = tuned automatically
    target_entropy: float | None = None  # None -> -action_dim
    # PPO-specific
    ppo_epochs: int = 10
    ppo_clip_epsilon: float = 0.2
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    gae_lambda: float = 0.95
    # PPO numerical stability (see docs/ppo_numerical_stability_audit.md)
    actor_lr: float | None = None  # None -> learning_rate
    critic_lr: float | None = None  # None -> learning_rate
    max_grad_norm: float = 0.5  # 0.0 disables gradient clipping
    log_std_min: float = -5.0  # bounds for the PPO Gaussian log-std
    log_std_max: float = 2.0
    adv_epsilon: float = 1e-8  # advantage-normalization denominator floor
    finite_check: bool = False  # raise FiniteError at the first non-finite value
    ppo_target_kl: float = 0.02  # stop PPO epochs for a rollout when KL exceeds (0 disables)
    # Worker policy sync
    policy_sync_interval: int = 10_000  # env steps between worker policy updates


@dataclass
class TrainingEnvConfig:
    split: str = "train"
    context_length: int = 64
    horizon: int = 512
    initial_balance: str = "10000"
    leverage: float = 50.0
    spread: float = 0.0002
    commission: float = 0.0
    sizing_mode: str = "equity_fraction"
    instruments: tuple[str, ...] = ()  # empty = all available instruments
    # Numerical-stability reward bounds (see RewardConfig): the finite floor
    # replaces -inf on equity collapse so PPO GAE/normalization cannot NaN.
    min_reward: float = -50.0
    max_reward: float | None = None


@dataclass
class ComputeConfig:
    num_workers: int | str = 4  # env worker processes; "auto" = estimate from RAM
    learner_device: str = "cpu"  # "cpu" | "cuda"
    torch_threads: int = 2  # learner BLAS threads (prevent oversubscription)
    torch_interop_threads: int | None = None  # None = leave torch default
    collect_backend: str = "sync"  # "sync" (deterministic, in-process) | "process"
    dataset_backend: str = "auto"  # "auto" | "parquet" | "mmap" (shared store)
    memory_limit_fraction: float | None = None  # warn when est. tree RAM > fraction
    seed: int = 42


@dataclass
class LoggingConfig:
    log_every_env_steps: int = 100_000
    evaluate_every_env_steps: int = 1_000_000
    checkpoint_every_env_steps: int = 1_000_000
    run_dir: str = "runs"


@dataclass
class EvalConfig:
    validation_episodes: int = 100
    test_episodes: int = 100
    eval_horizon: int = 512
    eval_seed: int = 42


@dataclass
class SelectionConfig:
    metric: str = "sharpe_drawdown"  # "sharpe" | "sharpe_drawdown" | "total_return"
    lambda_drawdown: float = 1.0  # Score = Sharpe - lambda * max_drawdown_pct


@dataclass
class ExperimentConfig:
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: TrainingEnvConfig = field(default_factory=TrainingEnvConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    run_id: str = ""
    dataset_version: str = "processed-parquet-2026"

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return _filter_none(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        cfg = cls()
        sections = {
            "algorithm": AlgorithmConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
            "environment": TrainingEnvConfig,
            "compute": ComputeConfig,
            "logging": LoggingConfig,
            "evaluation": EvalConfig,
            "selection": SelectionConfig,
        }
        for key, cls_ in sections.items():
            if data.get(key):
                cfg.__setattr__(key, cls_(**data[key]))
        if "run_id" in data:
            cfg.run_id = str(data["run_id"])
        if "dataset_version" in data:
            cfg.dataset_version = str(data["dataset_version"])
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def smoke(cls, algorithm: str = "sac") -> ExperimentConfig:
        """A tiny deterministic config for CI/smoke tests (not a real run)."""
        return cls(
            algorithm=AlgorithmConfig(name=algorithm),
            model=ModelConfig(hidden_dim=16, num_layers=2),
            training=TrainingConfig(
                total_env_steps=2000,
                batch_size=32,
                replay_capacity=10_000,
                warmup_steps=200,
                gradient_updates_per_step=1,
                collect_batch=64,
                gamma=0.99,
                tau=0.005,
                learning_rate=3e-3,
                alpha=1.0,
                policy_sync_interval=200,
            ),
            environment=TrainingEnvConfig(
                split="train",
                context_length=8,
                horizon=16,
                initial_balance="10000",
                leverage=50.0,
                spread=0.0002,
                sizing_mode="equity_fraction",
                instruments=("EURUSD",),
            ),
            compute=ComputeConfig(
                num_workers=1,
                learner_device="cpu",
                torch_threads=1,
                collect_backend="sync",
                seed=42,
            ),
            logging=LoggingConfig(
                log_every_env_steps=500,
                evaluate_every_env_steps=1000,
                checkpoint_every_env_steps=1000,
                run_dir="runs_smoke",
            ),
            evaluation=EvalConfig(
                validation_episodes=2, test_episodes=2, eval_horizon=16, eval_seed=42
            ),
        )


def default_config(algorithm: str = "sac") -> ExperimentConfig:
    return ExperimentConfig(algorithm=AlgorithmConfig(name=algorithm))
