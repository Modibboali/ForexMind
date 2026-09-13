"""MuZero losses and prediction diagnostics (Stage 4.4).

The loss is the standard MuZero objective over a ``K``-step latent unroll,
computed against the Stage 4.3 replay targets::

    L = c_p * L_policy + c_v * L_value + c_r * L_reward

with **exact** alignment::

    initial_inference(o_t)  -> policy target pi_t,   value target z_t   (no reward)
    recurrent(a_t)          -> reward target r_{t+1}, pi_{t+1}, z_{t+1}
    recurrent(a_{t+1})      -> reward target r_{t+2}, pi_{t+2}, z_{t+2}
    ...

Every term is normalized by its count of **valid** targets, so padding near the
end of a trajectory cannot dilute the loss.

Representation
--------------
The stage-4.4 brief's preferred design is used: value and reward are
**categorical support** predictions trained with cross-entropy against
``scalar_to_support(target)``.  When the model has ``use_support=False`` the same
API degrades to Huber/MSE regression on scalar heads, and the learner documents
which one is active.  Nothing here silently mixes the two.

Action masking
--------------
Replay targets already carry zero mass on illegal actions, and this module
verifies that (``validate_targets``).  ``mask_policy_logits`` additionally masks
the *prediction* logits with the stored environment mask so the objective matches
how the network is used at inference (where masking is always applied).  Set it
to ``False`` for the classic "learn the zeros from the data" formulation; both
paths are tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from forexmind.muzero.support import scalar_to_support

__all__ = ["LossConfig", "MuZeroLossResult", "MuZeroPrediction", "muzero_losses"]

HOLD_ACTION = 0
FLAT_ACTION = 1
SHORT_ACTIONS = (2, 3)
LONG_ACTIONS = (4, 5)

_SCALAR_LOSSES = ("huber", "mse")


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Weights, scales and switches for the MuZero objective.

    ``value_scale`` / ``reward_scale`` must match the model's decoding config
    (``MuZeroConfig.value_scale`` / ``reward_scale``) or the reported errors and
    the loss would be in different units; :meth:`for_model` wires that up.
    """

    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    value_scale: float = 1.0
    reward_scale: float = 1.0
    support_epsilon: float = 0.0
    mask_policy_logits: bool = True
    validate_targets: bool = True
    target_policy_tolerance: float = 1e-4
    scalar_loss: str = "huber"
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        for name in ("policy_loss_weight", "value_loss_weight", "reward_loss_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("value_scale", "reward_scale"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.support_epsilon < 0.0:
            raise ValueError(f"support_epsilon must be >= 0, got {self.support_epsilon}")
        if self.target_policy_tolerance < 0.0:
            raise ValueError("target_policy_tolerance must be >= 0")
        if self.scalar_loss not in _SCALAR_LOSSES:
            raise ValueError(
                f"scalar_loss must be one of {_SCALAR_LOSSES}, got {self.scalar_loss!r}"
            )
        if self.huber_delta <= 0.0:
            raise ValueError(f"huber_delta must be > 0, got {self.huber_delta}")

    @classmethod
    def for_model(cls, model_config: Any, **overrides: Any) -> LossConfig:
        """Build a config whose scales match ``MuZeroConfig`` decoding."""
        value_epsilon = float(getattr(model_config, "value_epsilon", 0.0))
        reward_epsilon = float(getattr(model_config, "reward_epsilon", 0.0))
        if abs(value_epsilon - reward_epsilon) > 1e-12:
            raise ValueError(
                "the loss assumes one shared support epsilon, but the model has "
                f"value_epsilon={value_epsilon} and reward_epsilon={reward_epsilon}; "
                "set them equal in MuZeroConfig"
            )
        base: dict[str, Any] = {
            "value_scale": float(model_config.value_scale),
            "reward_scale": float(model_config.reward_scale),
            "support_epsilon": value_epsilon,
        }
        base.update(overrides)
        return cls(**base)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_loss_weight": self.policy_loss_weight,
            "value_loss_weight": self.value_loss_weight,
            "reward_loss_weight": self.reward_loss_weight,
            "value_scale": self.value_scale,
            "reward_scale": self.reward_scale,
            "support_epsilon": self.support_epsilon,
            "mask_policy_logits": self.mask_policy_logits,
            "validate_targets": self.validate_targets,
            "scalar_loss": self.scalar_loss,
            "huber_delta": self.huber_delta,
        }


@dataclass(slots=True)
class MuZeroPrediction:
    """Stacked network outputs for one unroll (``D`` = latent dim, ``A`` = actions).

    ``policy_logits`` / ``value_logits`` carry ``K + 1`` entries (initial
    inference plus one per recurrent step); ``reward_logits`` carries ``K``
    because the initial state has no dynamics transition.
    """

    policy_logits: torch.Tensor  # [B, K+1, A]
    value_logits: torch.Tensor  # [B, K+1, value_dim]
    reward_logits: torch.Tensor  # [B, K, reward_dim]
    value: torch.Tensor  # [B, K+1] decoded economic units
    reward: torch.Tensor  # [B, K]    decoded economic units
    latents: list[torch.Tensor] = field(default_factory=list)  # K+1 x [B, D]


@dataclass(frozen=True, slots=True)
class MuZeroLossResult:
    """Masked, weighted loss terms plus detached diagnostics."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    valid_policy_targets: int
    valid_value_targets: int
    valid_reward_targets: int
    diagnostics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_loss": float(self.total.detach().item()),
            "policy_loss": float(self.policy.detach().item()),
            "value_loss": float(self.value.detach().item()),
            "reward_loss": float(self.reward.detach().item()),
            "valid_policy_targets": self.valid_policy_targets,
            "valid_value_targets": self.valid_value_targets,
            "valid_reward_targets": self.valid_reward_targets,
            **self.diagnostics,
        }


def _masked_mean(values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over positions where ``masks`` is 1 (never empty)."""
    weights = masks.reshape(-1).to(values.dtype)
    flat = values.reshape(-1)
    return (flat * weights).sum() / weights.sum().clamp_min(1.0)


def _entropy(probs: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    return -torch.xlogy(probs, probs).sum(dim=dim)


def _kl(target: torch.Tensor, log_pred: torch.Tensor) -> torch.Tensor:
    """``KL(target || pred)`` with the ``0 * log 0 = 0`` convention."""
    return (torch.xlogy(target, target) - target * log_pred).sum(dim=-1)


def _scalar_regression(
    prediction: torch.Tensor, target: torch.Tensor, config: LossConfig
) -> torch.Tensor:
    if config.scalar_loss == "mse":
        return (prediction - target) ** 2
    return F.smooth_l1_loss(prediction, target, beta=config.huber_delta, reduction="none")


def muzero_losses(
    prediction: MuZeroPrediction,
    batch: Any,
    config: LossConfig,
    *,
    use_support: bool = True,
    value_support_size: int = 21,
    reward_support_size: int = 21,
) -> MuZeroLossResult:
    """Compute the masked MuZero objective and its diagnostics.

    Args:
        prediction: stacked network outputs from one unroll.
        batch: a :class:`forexmind.muzero.targets.MuZeroBatch`.
        config: weights, scales and switches.
        use_support: ``True`` for categorical support heads, ``False`` for
            scalar regression.
        value_support_size / reward_support_size: head widths (needed to build
            the target distributions).
    """
    action_masks = batch.action_masks.bool()
    policy_masks = batch.policy_masks
    value_masks = batch.value_masks
    reward_masks = batch.reward_masks
    target_policies = batch.target_policies

    if config.validate_targets:
        invalid_mass = (target_policies * (~action_masks).to(target_policies.dtype)).sum(dim=-1)
        valid_rows = policy_masks > 0.5
        observed = invalid_mass[valid_rows] if valid_rows.any() else invalid_mass.reshape(-1)[:0]
        if observed.numel() and float(observed.max().item()) > config.target_policy_tolerance:
            raise ValueError(
                "policy target places probability mass on an invalid action "
                f"(max {float(observed.max().item()):.3e} > tolerance "
                f"{config.target_policy_tolerance:.1e}); refusing to train on a "
                "corrupted search target"
            )

    logits = prediction.policy_logits
    if config.mask_policy_logits:
        logits = logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min)
    log_probs = torch.log_softmax(logits, dim=-1)

    per_position = -(target_policies * log_probs).sum(dim=-1)
    policy_count = int((policy_masks > 0.5).sum().item())
    policy_loss = _masked_mean(per_position, policy_masks)

    if use_support:
        value_target_dist = scalar_to_support(
            batch.target_values,
            value_support_size,
            scale=config.value_scale,
            epsilon=config.support_epsilon,
        )
        value_per_position = -(
            value_target_dist * torch.log_softmax(prediction.value_logits, dim=-1)
        ).sum(dim=-1)
        reward_target_dist = scalar_to_support(
            batch.target_rewards,
            reward_support_size,
            scale=config.reward_scale,
            epsilon=config.support_epsilon,
        )
        reward_per_position = -(
            reward_target_dist * torch.log_softmax(prediction.reward_logits, dim=-1)
        ).sum(dim=-1)
    else:
        # Scalar heads are compared in economic units (value_scale is already
        # applied by decode_scalar), so the reported MAE is directly comparable
        # with the support-mode diagnostics.
        value_per_position = _scalar_regression(prediction.value, batch.target_values, config)
        reward_per_position = _scalar_regression(prediction.reward, batch.target_rewards, config)

    value_count = int((value_masks > 0.5).sum().item())
    reward_count = int((reward_masks > 0.5).sum().item())
    value_loss = _masked_mean(value_per_position, value_masks)
    reward_loss = _masked_mean(reward_per_position, reward_masks)

    total = (
        config.policy_loss_weight * policy_loss
        + config.value_loss_weight * value_loss
        + config.reward_loss_weight * reward_loss
    )

    diagnostics = _diagnostics(
        prediction=prediction,
        batch=batch,
        log_probs=log_probs,
        action_masks=action_masks,
        policy_masks=policy_masks,
        value_masks=value_masks,
        reward_masks=reward_masks,
    )
    return MuZeroLossResult(
        total=total,
        policy=policy_loss,
        value=value_loss,
        reward=reward_loss,
        valid_policy_targets=policy_count,
        valid_value_targets=value_count,
        valid_reward_targets=reward_count,
        diagnostics=diagnostics,
    )


def _diagnostics(
    *,
    prediction: MuZeroPrediction,
    batch: Any,
    log_probs: torch.Tensor,
    action_masks: torch.Tensor,
    policy_masks: torch.Tensor,
    value_masks: torch.Tensor,
    reward_masks: torch.Tensor,
) -> dict[str, float]:
    """Prediction-quality diagnostics (§21, §22) -- observations only, never losses."""
    with torch.no_grad():
        probs = log_probs.exp()
        target = batch.target_policies
        valid_policy = policy_masks > 0.5

        pred_entropy = _masked_mean(_entropy(probs), policy_masks)
        target_entropy = _masked_mean(_entropy(target), policy_masks)
        kl = _masked_mean(_kl(target, log_probs), policy_masks)

        pred_argmax = probs.argmax(dim=-1)
        target_argmax = target.argmax(dim=-1)
        agreement = (pred_argmax == target_argmax).to(probs.dtype)
        top1 = _masked_mean(agreement, policy_masks)

        pred_valid = probs[valid_policy]  # [N, A] only valid positions
        target_valid = target[valid_policy]

        value_pred = prediction.value
        value_target = batch.target_values
        value_err = (value_pred - value_target).abs()
        value_sq = (value_pred - value_target) ** 2
        reward_err = (prediction.reward - batch.target_rewards).abs()
        reward_sq = (prediction.reward - batch.target_rewards) ** 2

        diagnostics: dict[str, float] = {
            "policy_entropy": float(pred_entropy.item()),
            "target_policy_entropy": float(target_entropy.item()),
            "policy_kl": float(kl.item()),
            "policy_top1_agreement": float(top1.item()),
            "value_pred_mean": float(_masked_mean(value_pred.reshape(-1), value_masks).item()),
            "value_target_mean": float(_masked_mean(value_target, value_masks).item()),
            "value_mae": float(_masked_mean(value_err, value_masks).item()),
            "value_rmse": float(_masked_mean(value_sq, value_masks).sqrt().item()),
            "reward_pred_mean": float(_masked_mean(prediction.reward, reward_masks).item()),
            "reward_target_mean": float(_masked_mean(batch.target_rewards, reward_masks).item()),
            "reward_mae": float(_masked_mean(reward_err, reward_masks).item()),
            "reward_rmse": float(_masked_mean(reward_sq, reward_masks).sqrt().item()),
        }

        if pred_valid.numel():
            diagnostics.update(
                {
                    "pred_hold_prob": float(pred_valid[:, HOLD_ACTION].mean().item()),
                    "target_hold_prob": float(target_valid[:, HOLD_ACTION].mean().item()),
                    "pred_argmax_hold_fraction": float(
                        (pred_argmax.reshape(-1)[valid_policy.reshape(-1)] == HOLD_ACTION)
                        .to(probs.dtype)
                        .mean()
                        .item()
                    ),
                    "target_argmax_hold_fraction": float(
                        (target_argmax.reshape(-1)[valid_policy.reshape(-1)] == HOLD_ACTION)
                        .to(probs.dtype)
                        .mean()
                        .item()
                    ),
                    "pred_group_hold": float(pred_valid[:, HOLD_ACTION].mean().item()),
                    "pred_group_flat": float(pred_valid[:, FLAT_ACTION].mean().item()),
                    "pred_group_short": float(
                        pred_valid[:, list(SHORT_ACTIONS)].sum(dim=-1).mean().item()
                    ),
                    "pred_group_long": float(
                        pred_valid[:, list(LONG_ACTIONS)].sum(dim=-1).mean().item()
                    ),
                    "target_group_hold": float(target_valid[:, HOLD_ACTION].mean().item()),
                    "target_group_flat": float(target_valid[:, FLAT_ACTION].mean().item()),
                    "target_group_short": float(
                        target_valid[:, list(SHORT_ACTIONS)].sum(dim=-1).mean().item()
                    ),
                    "target_group_long": float(
                        target_valid[:, list(LONG_ACTIONS)].sum(dim=-1).mean().item()
                    ),
                }
            )
        if prediction.latents:
            for index, latent in enumerate(prediction.latents):
                detached = latent.detach()
                prefix = f"latent_k{index}_"
                diagnostics[prefix + "mean"] = float(detached.mean().item())
                diagnostics[prefix + "std"] = (
                    float(detached.std().item()) if detached.numel() > 1 else 0.0
                )
                diagnostics[prefix + "min"] = float(detached.min().item())
                diagnostics[prefix + "max"] = float(detached.max().item())
                diagnostics[prefix + "absmax"] = float(detached.abs().max().item())
                diagnostics[prefix + "norm"] = float(detached.norm(dim=-1).mean().item())
                diagnostics[prefix + "finite"] = float(
                    torch.isfinite(detached).to(torch.float32).mean().item()
                )
    return diagnostics
