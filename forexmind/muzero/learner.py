"""MuZero recurrent-unroll learner (Stage 4.4).

Consumes a Stage 4.3 :class:`~forexmind.muzero.targets.MuZeroBatch`, unrolls the
latent dynamics ``K`` steps, computes the masked MuZero objective, and takes one
optimizer step::

    sample replay batch
          -> initial_inference(o_t)                 -> policy/value loss at t
          -> recurrent_inference(latent, a_t)       -> reward/policy/value loss at t+1
          -> ... K times
          -> backprop -> optimizer step

Design points that matter for correctness:

* **One optimizer** over representation + dynamics + prediction (they are one
  model), created by :class:`OptimizerConfig`.
* **Gradient scaling** through the recurrent chain: after every transition
  except the last, ``latent = g * latent + (1 - g) * latent.detach()`` keeps the
  forward value but damps the backward path.  ``g = 1.0`` disables it; the
  default is ``0.5``.  Only the *input to the next step* is scaled, so the head
  losses at the current step still see the true latent.
* **No target network.** Value targets are the stored Stage 4.3 search values.
* **Train-split guard.** Every batch row's ``split`` must be ``"train"``.
* **Finite checks** reuse :mod:`forexmind.training.numerics` and never replace
  invalid values.

Still **not** implemented: distributed actors, reanalysis, prioritized replay,
target refreshing, or Stochastic MuZero.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forexmind.muzero.config import MuZeroConfig
from forexmind.muzero.inference import decode_scalar
from forexmind.muzero.losses import (
    LossConfig,
    MuZeroLossResult,
    MuZeroPrediction,
    muzero_losses,
)
from forexmind.training.numerics import FiniteError, assert_finite

__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "LearnerConfig",
    "MuZeroLearner",
    "OptimizerConfig",
    "check_batch_shapes",
]

CHECKPOINT_FORMAT = "forexmind.muzero.learner"
CHECKPOINT_VERSION = 1

_OPTIMIZERS = ("adamw", "adam")
_SCHEDULES = ("constant", "step")


def check_batch_shapes(batch: Any, *, num_actions: int) -> int:
    """Assert the Stage 4.3 batch contract and return the unroll length ``K``.

    Shapes are asserted *before* any inference or optimization because a
    mis-shaped target can broadcast silently instead of raising: a ``[B, A]``
    policy target multiplies cleanly against ``[B, K+1, A]`` logits, which would
    quietly optimize a different objective.  Every field of
    :class:`~forexmind.muzero.targets.MuZeroBatch` is pinned to its documented
    shape, and the batch size must agree across all of them.
    """
    observation = batch.observation
    actions = batch.actions
    if not torch.is_tensor(observation) or not torch.is_tensor(actions):
        raise ValueError(
            "batch must hold torch tensors; build it with "
            "forexmind.muzero.targets.collate_samples or replay.sample"
        )
    if observation.ndim != 2:
        raise ValueError(f"batch.observation must be [B, obs_dim], got {tuple(observation.shape)}")
    if actions.ndim != 2:
        raise ValueError(f"batch.actions must be [B, K], got {tuple(actions.shape)}")
    if actions.dtype.is_floating_point or actions.dtype.is_complex:
        raise ValueError(f"batch.actions must be an integer tensor, got {actions.dtype}")

    batch_size = int(observation.shape[0])
    if int(actions.shape[0]) != batch_size:
        raise ValueError(
            f"batch.actions has {int(actions.shape[0])} rows but batch.observation has {batch_size}"
        )
    steps = int(actions.shape[1])
    if steps < 1:
        raise ValueError("batch.actions must contain at least one unroll step")

    expected: dict[str, tuple[int, ...]] = {
        "target_rewards": (batch_size, steps),
        "reward_masks": (batch_size, steps),
        "target_values": (batch_size, steps + 1),
        "target_policies": (batch_size, steps + 1, num_actions),
        "policy_masks": (batch_size, steps + 1),
        "value_masks": (batch_size, steps + 1),
        "action_masks": (batch_size, steps + 1, num_actions),
    }
    for name, shape in expected.items():
        array = getattr(batch, name, None)
        if array is None:
            raise ValueError(f"batch is missing the required field {name!r}")
        if tuple(array.shape) != shape:
            raise ValueError(
                f"batch.{name} shape {tuple(array.shape)} != {shape} "
                f"(B={batch_size}, K={steps}, num_actions={num_actions})"
            )
    return steps


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Optimizer and gradient-clipping configuration (nothing is hard-coded)."""

    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float = 5.0
    schedule: str = "constant"
    lr_step_size: int = 1_000
    lr_gamma: float = 0.5

    def __post_init__(self) -> None:
        if self.name not in _OPTIMIZERS:
            raise ValueError(f"name must be one of {_OPTIMIZERS}, got {self.name!r}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if len(self.betas) != 2 or not all(0.0 <= b < 1.0 for b in self.betas):
            raise ValueError(f"betas must be two values in [0, 1), got {self.betas}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.schedule not in _SCHEDULES:
            raise ValueError(f"schedule must be one of {_SCHEDULES}, got {self.schedule!r}")
        if self.schedule == "step":
            if self.lr_step_size < 1:
                raise ValueError("lr_step_size must be >= 1 for a step schedule")
            if not 0.0 < self.lr_gamma < 1.0:
                raise ValueError(f"lr_gamma must be in (0, 1), got {self.lr_gamma}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": list(self.betas),
            "eps": self.eps,
            "max_grad_norm": self.max_grad_norm,
            "schedule": self.schedule,
            "lr_step_size": self.lr_step_size,
            "lr_gamma": self.lr_gamma,
        }


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    """Everything the learner needs beyond the model itself."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    latent_gradient_scale: float = 0.5
    finite_check: bool = True
    require_train_split: bool = True
    device: str = "cpu"
    seed: int = 0
    history_limit: int = 200

    def __post_init__(self) -> None:
        if not 0.0 <= self.latent_gradient_scale <= 1.0:
            raise ValueError(
                "latent_gradient_scale must be in [0, 1] (1.0 disables scaling), got "
                f"{self.latent_gradient_scale}"
            )
        if self.history_limit < 0:
            raise ValueError("history_limit must be >= 0")

    @classmethod
    def for_model(cls, model_config: MuZeroConfig, **overrides: Any) -> LearnerConfig:
        """Default learner config whose scales match ``model_config``."""
        loss_overrides = overrides.pop("loss_overrides", {})
        loss = LossConfig.for_model(model_config, **loss_overrides)
        return cls(loss=loss, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.to_dict(),
            "loss": self.loss.to_dict(),
            "latent_gradient_scale": self.latent_gradient_scale,
            "finite_check": self.finite_check,
            "require_train_split": self.require_train_split,
            "device": self.device,
            "seed": self.seed,
        }


class MuZeroLearner:
    """Joint optimizer for the MuZero representation, dynamics and prediction nets."""

    def __init__(
        self,
        model: Any,
        config: LearnerConfig | None = None,
        *,
        replay: Any | None = None,
    ) -> None:
        self.model = model
        self.config = config or LearnerConfig.for_model(model.config)
        self.replay = replay
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.model.train()

        self.update_count = 0
        self.env_steps = 0
        self._torch_rng = torch.Generator(device="cpu")
        self._torch_rng.manual_seed(self.config.seed)
        self._numpy_rng = np.random.default_rng(self.config.seed)

        self.optimizer = self._build_optimizer()
        self.history: list[dict[str, float]] = []

    # -- model facts ----------------------------------------------------------

    @property
    def model_config(self) -> MuZeroConfig:
        return self.model.config

    @property
    def use_support(self) -> bool:
        return bool(self.model_config.use_support)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config.optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("model has no trainable parameters")
        if cfg.name == "adam":
            return torch.optim.Adam(
                params,
                lr=cfg.learning_rate,
                betas=cfg.betas,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
            )
        return torch.optim.AdamW(
            params,
            lr=cfg.learning_rate,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _apply_schedule(self) -> None:
        cfg = self.config.optimizer
        if cfg.schedule == "constant":
            return
        decay = cfg.lr_gamma ** (self.update_count // cfg.lr_step_size)
        for group in self.optimizer.param_groups:
            group["lr"] = cfg.learning_rate * decay

    # -- unroll ---------------------------------------------------------------

    def unroll(self, batch: Any) -> MuZeroPrediction:
        """Run ``initial_inference`` plus ``K`` recurrent steps, keeping gradients."""
        config = self.model_config
        steps = check_batch_shapes(batch, num_actions=config.num_actions)
        actions = batch.actions.to(self.device)

        root = self.model.initial_inference(batch.observation.to(self.device))
        latents = [root.latent_state]
        policy_logits = [root.policy_logits]
        value_raw = [root.value_logits if root.value_logits is not None else root.value]
        reward_raw: list[torch.Tensor] = []

        gate = float(self.config.latent_gradient_scale)
        for index in range(steps):
            output = self.model.recurrent_inference(latents[-1], actions[:, index])
            latents.append(output.latent_state)
            policy_logits.append(output.policy_logits)
            value_raw.append(
                output.value_logits if output.value_logits is not None else output.value
            )
            reward_raw.append(
                output.reward_logits if output.reward_logits is not None else output.reward
            )
            if gate < 1.0 and index < steps - 1:
                # Keep the forward value, damp the backward path into earlier steps.
                latents[-1] = gate * latents[-1] + (1.0 - gate) * latents[-1].detach()

        stacked_policy = torch.stack(policy_logits, dim=1)  # [B, K+1, A]
        stacked_value_raw = torch.stack(value_raw, dim=1)  # [B, K+1, value_dim]
        stacked_reward_raw = torch.stack(reward_raw, dim=1)  # [B, K, reward_dim]

        value = decode_scalar(
            stacked_value_raw,
            use_support=config.use_support,
            support_size=config.value_support_size,
            scale=config.value_scale,
            epsilon=config.value_epsilon,
        ).squeeze(-1)
        reward = decode_scalar(
            stacked_reward_raw,
            use_support=config.use_support,
            support_size=config.reward_support_size,
            scale=config.reward_scale,
            epsilon=config.reward_epsilon,
        ).squeeze(-1)

        return MuZeroPrediction(
            policy_logits=stacked_policy,
            value_logits=stacked_value_raw,
            reward_logits=stacked_reward_raw,
            value=value,
            reward=reward,
            latents=latents,
        )

    # -- training step --------------------------------------------------------

    def train_step(self, batch: Any, *, splits: Sequence[str] | None = None) -> dict[str, Any]:
        """One unroll, loss, backward pass and optimizer step."""
        self._check_split(batch, splits)
        context: dict[str, object] = {
            "gradient_updates": self.update_count,
            "env_steps": self.env_steps,
        }
        if self.config.finite_check:
            assert_finite("batch.observation", batch.observation, context=context, strict=True)
            assert_finite("batch.target_values", batch.target_values, context=context, strict=True)
            assert_finite(
                "batch.target_rewards", batch.target_rewards, context=context, strict=True
            )
            assert_finite(
                "batch.target_policies", batch.target_policies, context=context, strict=True
            )

        prediction = self.unroll(batch)
        if self.config.finite_check:
            assert_finite(
                "prediction.policy_logits", prediction.policy_logits, context=context, strict=True
            )

        result: MuZeroLossResult = muzero_losses(
            prediction,
            batch,
            self.config.loss,
            use_support=self.use_support,
            value_support_size=self.model_config.value_support_size,
            reward_support_size=self.model_config.reward_support_size,
        )
        if self.config.finite_check:
            assert_finite("loss.total", result.total, context=context, strict=True)

        self.optimizer.zero_grad(set_to_none=True)
        result.total.backward()
        grad_norm = self._clip_gradients()
        if self.config.finite_check:
            self._assert_finite_gradients(context)

        self._apply_schedule()
        self.optimizer.step()
        self.update_count += 1

        metrics = self._metrics(result, batch, grad_norm)
        if self.config.history_limit:
            self.history.append(metrics)
            if len(self.history) > self.config.history_limit:
                self.history.pop(0)
        return metrics

    def _check_split(self, batch: Any, splits: Sequence[str] | None) -> None:
        if not self.config.require_train_split:
            return
        observed = tuple(splits) if splits is not None else tuple(getattr(batch, "splits", ()))
        bad = sorted({s for s in observed if s != "train"})
        if bad:
            raise ValueError(
                f"refusing to train on non-train trajectories {bad}; MuZero replay must "
                "contain TRAIN split episodes only"
            )

    def _clip_gradients(self) -> float:
        cfg = self.config.optimizer
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0
        total = torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm)
        norm = float(total.detach().item()) if torch.is_tensor(total) else float(total)
        return norm

    def _assert_finite_gradients(self, context: dict[str, object]) -> None:
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise FiniteError(
                    f"non-finite gradient in {name}",
                    component=f"grad.{name}",
                    context=context,
                )

    def _metrics(self, result: MuZeroLossResult, batch: Any, grad_norm: float) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "update": self.update_count,
            "total_loss": float(result.total.detach().item()),
            "policy_loss": float(result.policy.detach().item()),
            "value_loss": float(result.value.detach().item()),
            "reward_loss": float(result.reward.detach().item()),
            "gradient_norm": grad_norm,
            "learning_rate": self.learning_rate,
            "batch_size": int(batch.observation.shape[0]),
            "unroll_steps": int(batch.actions.shape[1]),
            "valid_policy_targets": result.valid_policy_targets,
            "valid_value_targets": result.valid_value_targets,
            "valid_reward_targets": result.valid_reward_targets,
            "representation": "support" if self.use_support else "scalar",
        }
        metrics.update(result.diagnostics)
        return metrics

    # -- checkpointing --------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        target_config: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Serialize model, optimizer, counters, config and RNG state."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        model_config = self.model_config
        payload = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "architecture": {
                "num_actions": model_config.num_actions,
                "obs_dim": model_config.obs_dim,
                "latent_dim": model_config.latent_dim,
                "hidden_dim": model_config.hidden_dim,
                "num_layers": model_config.num_layers,
                "action_embedding_dim": model_config.action_embedding_dim,
                "use_support": model_config.use_support,
                "value_support_size": model_config.value_support_size,
                "reward_support_size": model_config.reward_support_size,
                "reward_value_representation": (
                    "categorical_support" if model_config.use_support else "scalar_regression"
                ),
            },
            "model_config": model_config.to_dict(),
            "learner_config": self.config.to_dict(),
            "target_config": target_config.to_dict() if target_config is not None else None,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "env_steps": self.env_steps,
            "replay_metadata": (self.replay.memory_report() if self.replay is not None else None),
            "rng": {
                "torch": self._torch_rng.get_state().tolist(),
                "numpy": _numpy_rng_state_to_json(self._numpy_rng),
            },
            "extra": dict(extra or {}),
        }
        torch.save(payload, destination)
        return destination

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
        """Restore a checkpoint, validating architecture compatibility."""
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a ForexMind MuZero learner checkpoint")
        architecture = payload.get("architecture", {})
        current = self.model_config
        mismatches = {
            key: (architecture.get(key), getattr(current, key))
            for key in (
                "num_actions",
                "obs_dim",
                "latent_dim",
                "hidden_dim",
                "num_layers",
                "action_embedding_dim",
                "use_support",
                "value_support_size",
                "reward_support_size",
            )
            if architecture.get(key) != getattr(current, key)
        }
        if mismatches and strict:
            raise ValueError(f"checkpoint architecture does not match the model: {mismatches}")
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.update_count = int(payload.get("update_count", 0))
        self.env_steps = int(payload.get("env_steps", 0))
        rng = payload.get("rng", {})
        if "torch" in rng:
            self._torch_rng.set_state(torch.tensor(rng["torch"], dtype=torch.uint8))
        if "numpy" in rng:
            self._numpy_rng.bit_generator.state = rng["numpy"]
        return payload

    # -- reporting ------------------------------------------------------------

    def parameter_report(self) -> dict[str, Any]:
        report = self.model.parameter_report()
        report["optimizer"] = self.config.optimizer.name
        report["update_count"] = self.update_count
        return report

    def metrics_summary(self) -> dict[str, Any]:
        """Mean of the trailing history plus the latest step (for logging)."""
        if not self.history:
            return {"updates": 0}
        keys = (
            "total_loss",
            "policy_loss",
            "value_loss",
            "reward_loss",
            "gradient_norm",
            "policy_kl",
            "value_mae",
            "reward_mae",
            "pred_argmax_hold_fraction",
            "target_argmax_hold_fraction",
        )
        summary: dict[str, Any] = {"updates": len(self.history)}
        for key in keys:
            values = [row[key] for row in self.history if key in row]
            if values:
                summary[f"mean_{key}"] = float(np.mean(values))
                summary[f"last_{key}"] = float(values[-1])
        return summary

    def describe(self) -> str:
        report = self.parameter_report()
        return (
            f"MuZeroLearner(updates={self.update_count}, params={report['total']}, "
            f"optimizer={report['optimizer']}, representation="
            f"{'support' if self.use_support else 'scalar'}, "
            f"unroll_gate={self.config.latent_gradient_scale})"
        )


def _numpy_rng_state_to_json(rng: np.random.Generator) -> dict[str, Any]:
    state = rng.bit_generator.state
    return json.loads(json.dumps(state, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")
