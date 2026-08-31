"""PPO trainer (Phase 3).

On-policy proximal policy optimization with a Gaussian policy (clipped
objective), a learned value baseline, GAE returns, and an entropy bonus.
The collector performs full on-policy rollouts: each collection round the
learner re-syncs its current policy/value to the workers, collects a batch,
then runs several PPO epochs over that batch.

Numerical stability (see docs/ppo_numerical_stability_audit.md):

* the pipeline is instrumented to find the *first* non-finite tensor in a
  documented order (obs -> reward -> value -> returns -> advantages -> actor
  output -> log-prob -> ratio -> losses -> gradients -> parameters);
* ``finite_check`` mode raises :class:`FiniteError` at the first non-finite
  value (used for debugging and the 1M-step validation run);
* rewards are validated and their min/max/mean/std/max-abs are logged;
* GAE + advantage normalization guard against degenerate std and non-finite
  inputs;
* actor and critic gradients are clipped to ``max_grad_norm`` (0 disables);
* ``actor_lr`` / ``critic_lr`` are independently configurable;
* the stored worker log-prob and the trainer's new log-prob both use the
  density of the RAW pre-clamp Gaussian sample ``u`` (the exact sampling
  distribution), while the environment-facing action stays ``clamp(u)``;
* ``ppo_target_kl`` stops the remaining PPO epochs for a rollout once the
  KL estimate exceeds it (policy cannot run away inside a rollout)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from forexmind.training.collector import Transition
from forexmind.training.config import ExperimentConfig
from forexmind.training.networks import GaussianPolicy, ValueNet
from forexmind.training.numerics import (
    assert_finite,
    grad_norm_stats,
    parameter_stats,
    tensor_stats,
)
from forexmind.training.trainer import BaseTrainer

# Numerically-safe band for ``log(new_prob / old_prob)`` before ``exp``.
# PPO's clip (``ppo_clip_epsilon``) makes ratios beyond ~1 +/- eps equivalent
# in the objective, so clamping the *log* ratio to +/-10 (ratio in
# [4.5e-5, 2.2e4]) only prevents ``exp`` overflow -> inf -> NaN and does not
# change the clipped objective.
LOG_RATIO_CLAMP = 10.0


class PPOTrainer(BaseTrainer):
    def __init__(
        self,
        config: ExperimentConfig,
        run_dir: str | Path,
        *,
        dataset: Any = None,
    ) -> None:
        super().__init__(config, run_dir, dataset=dataset)
        model = config.model
        self.actor = GaussianPolicy(
            self.obs_dim,
            self.action_dim,
            model,
            log_std_min=config.training.log_std_min,
            log_std_max=config.training.log_std_max,
        )
        self._value_net = ValueNet(self.obs_dim, model)
        self.actor.to(self.device)
        self._value_net.to(self.device)
        self.actor.eval()
        self._value_net.eval()

        actor_lr = config.training.actor_lr or config.training.learning_rate
        critic_lr = config.training.critic_lr or config.training.learning_rate
        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.value_opt = Adam(self._value_net.parameters(), lr=critic_lr)

        self._rollout: list[Transition] = []
        self._rng = np.random.default_rng(config.compute.seed)
        self._finite_check = bool(config.training.finite_check)
        self._adv_epsilon = float(config.training.adv_epsilon)
        self._finite_alert_count = 0
        self._nonfinite_total = 0
        self._clip_warn_count = 0

    # -- BaseTrainer interface ------------------------------------------------

    def policy(self) -> nn.Module:
        return self.actor

    def value_net(self) -> nn.Module | None:
        return self._value_net

    def replay_size(self) -> int:
        return len(self._rollout)

    def _consume_transitions(self, transitions: list[Transition]) -> None:
        self._rollout.extend(transitions)
        if len(self._rollout) < self.config.training.collect_batch:
            return
        diag = self.update()
        self._gradient_updates += 1
        self._record_diagnostics(diag)
        self._rollout.clear()

    def _maybe_resync_policy(self) -> None:
        # PPO is on-policy: workers must act with the updated policy every round.
        self._sync_policy_to_workers()

    # -- first-non-finite diagnostics -----------------------------------------

    def _context_for(self, index: int | None) -> dict[str, object]:
        ctx: dict[str, object] = {
            "gradient_updates": self._gradient_updates,
            "env_steps": self._env_steps,
            "instrument": "",
            "timestamp": "",
        }
        if index is not None and 0 <= index < len(self._rollout):
            t = self._rollout[index]
            ctx["instrument"] = t.instrument
            ctx["timestamp"] = str(t.timestamp) if t.timestamp is not None else ""
        return ctx

    def _check_first_nonfinite(self, name: str, value: Any, *, dim: int | None = None) -> bool:
        """Return True when ``value`` contains a non-finite entry.

        In ``finite_check`` mode the first detection raises :class:`FiniteError`
        with instrument/timestamp context.  Otherwise it is logged (a few
        alerts per run) and counted in ``_nonfinite_total``.
        """
        stats = tensor_stats(value)
        if int(stats["finite_count"]) == int(stats["total_count"]):
            return False
        from forexmind.training.numerics import first_nonfinite_index

        flat = first_nonfinite_index(value)
        row = (flat // dim) if (dim and dim is not None and dim > 0 and flat is not None) else flat
        ctx = self._context_for(row)
        if self._finite_check:
            assert_finite(name, value, context=ctx, strict=True)
        elif self._finite_alert_count < 20:
            self._finite_alert_count += 1
            assert_finite(name, value, context=ctx, strict=False)
        self._nonfinite_total += 1
        return True

    def _check_grads_finite(self, module: nn.Module, label: str) -> None:
        for pname, p in module.named_parameters():
            if p.grad is None:
                continue
            if self._check_first_nonfinite(f"{label}.{pname}.grad", p.grad, dim=None):
                break

    def _check_params_finite(self, module: nn.Module, label: str) -> None:
        for pname, p in module.named_parameters():
            if self._check_first_nonfinite(f"{label}.{pname}.param", p, dim=None):
                break

    def _all_params(self) -> list[nn.Parameter]:
        return [p for p in self.actor.parameters()] + [p for p in self._value_net.parameters()]

    def _param_update_magnitude(self, params_before: list[torch.Tensor]) -> tuple[float, float]:
        """Mean/max absolute parameter change over this rollout update."""
        deltas = [
            p.detach().sub(pb) for p, pb in zip(self._all_params(), params_before, strict=True)
        ]
        flat = torch.cat([d.ravel() for d in deltas])
        if flat.numel() == 0:
            return 0.0, 0.0
        return float(flat.abs().mean().item()), float(flat.abs().max().item())

    def _minibatch_actor(
        self,
        dist: Any,
        act_raw: torch.Tensor,
        old_logp: torch.Tensor,
        adv: torch.Tensor,
        *,
        eps: float,
        ent_coef: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float, float, torch.Tensor]:
        """Clipped PPO actor surrogate for one minibatch.

        ``act_raw`` is the pre-clamp Gaussian sample, so ``logp = log pi_theta
        (u)`` is the density of the actual sampling distribution (not the
        clamped boundary).  Returns ``(actor_loss, ratio, approx_kl,
        approx_kl_log, clip_fraction, entropy)`` where

        * ``approx_kl    = E[0.5 (r - 1)^2]``      (kept for continuity)
        * ``approx_kl_log = E[r - 1 - log r]``      (standard KL estimate,
           used for target-KL early stopping)
        """
        logp = dist.log_prob(act_raw).sum(-1, keepdim=True)
        log_ratio = (logp - old_logp).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
        ratio = log_ratio.exp()
        approx_kl = float((0.5 * (ratio - 1.0).pow(2)).mean().item())
        approx_kl_log = float(((ratio - 1.0) - log_ratio).mean().item())
        clip_frac = float(((ratio - 1.0).abs() > eps).float().mean().item())
        entropy = dist.entropy().sum(-1, keepdim=True).mean()
        surr1 = ratio * adv
        surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
        actor_loss = -(torch.min(surr1, surr2).mean() + ent_coef * entropy)
        return actor_loss, ratio, approx_kl, approx_kl_log, clip_frac, entropy

    # -- GAE ------------------------------------------------------------------

    def _compute_gae(
        self,
        rew: np.ndarray,
        done: np.ndarray,
        val: np.ndarray,
        next_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Raw GAE recurrence (pre-normalization), exposed for hand-computed tests.

        ``done`` marks *terminated* episodes only; truncated (horizon-limit)
        episodes bootstrap with ``V(s')`` (``done[i]=False``), which is the
        correct handling for time-limit truncation.
        """
        n = len(rew)
        gamma = self.config.training.gamma
        lam = self.config.training.gae_lambda
        adv = np.zeros(n, dtype=np.float32)
        last = 0.0
        for i in range(n - 1, -1, -1):
            v_next = 0.0 if done[i] else float(next_val[i])
            delta = float(rew[i]) + gamma * v_next - float(val[i])
            last = delta + gamma * lam * (0.0 if done[i] else 1.0) * last
            adv[i] = last
        returns = (adv + np.asarray(val, dtype=np.float32)).astype(np.float32)
        return adv.astype(np.float32), returns

    def _gae(
        self,
        rew: np.ndarray,
        done: np.ndarray,
        val: np.ndarray,
        next_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generalized advantage estimation with finite-std normalization."""
        adv, returns = self._compute_gae(rew, done, val, next_val)
        # Advantage normalization A = (A - mu) / (sigma + eps).
        std = float(adv.std())
        if np.isfinite(std) and std > self._adv_epsilon:
            adv = (adv - adv.mean()) / (std + self._adv_epsilon)
        elif np.isfinite(std):
            # Degenerate batch (all advantages identical): zero signal.
            adv = np.zeros_like(adv)
        # If std is non-finite the caller's finite check reports it.
        return adv.astype(np.float32), returns

    # -- PPO update -----------------------------------------------------------

    def update(self) -> dict[str, float]:
        rollout = self._rollout
        n = len(rollout)
        if n == 0:
            return {}
        obs = np.stack([t.obs for t in rollout]).astype(np.float32)
        next_obs = np.stack([t.next_obs for t in rollout]).astype(np.float32)
        act = np.asarray([t.action for t in rollout], dtype=np.float32).reshape(-1, 1)
        act_raw = np.asarray([t.action_raw for t in rollout], dtype=np.float32).reshape(-1, 1)
        rew = np.asarray([t.reward for t in rollout], dtype=np.float32)
        done = np.asarray([t.terminated for t in rollout], dtype=bool)
        old_logp = np.asarray([t.log_prob for t in rollout], dtype=np.float32).reshape(-1, 1)
        val = np.asarray([t.value for t in rollout], dtype=np.float32)
        next_val = np.asarray([t.next_value for t in rollout], dtype=np.float32)

        # -- stage 1: validate the pipeline in documented order ---------------
        self._check_first_nonfinite("observation", obs, dim=obs.shape[1])
        self._check_first_nonfinite("next_observation", next_obs, dim=next_obs.shape[1])
        self._check_first_nonfinite("action", act, dim=1)
        self._check_first_nonfinite("raw_action", act_raw, dim=1)
        self._check_first_nonfinite("reward", rew, dim=None)
        self._check_first_nonfinite("value_prediction", val, dim=None)
        self._check_first_nonfinite("next_value_prediction", next_val, dim=None)
        self._check_first_nonfinite("old_log_prob", old_logp, dim=1)

        rew_stats = tensor_stats(rew)
        adv, returns = self._gae(rew, done, val, next_val)
        self._check_first_nonfinite("returns", returns, dim=None)
        self._check_first_nonfinite("normalized_advantages", adv, dim=None)
        ret_stats = tensor_stats(returns)
        adv_stats = tensor_stats(adv)

        # -- stage 2: tensors -------------------------------------------------
        obs_t = torch.as_tensor(obs, device=self.device)
        act_raw_t = torch.as_tensor(act_raw, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device).unsqueeze(1)
        ret_t = torch.as_tensor(returns, device=self.device).unsqueeze(1)

        cfg = self.config.training
        eps = cfg.ppo_clip_epsilon
        ent_coef = cfg.ppo_entropy_coef
        val_coef = cfg.ppo_value_coef
        batch = cfg.batch_size
        max_grad_norm = float(cfg.max_grad_norm)
        target_kl = float(cfg.ppo_target_kl or 0.0)

        params_before = [p.detach().clone() for p in self._all_params()]

        total_actor = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_kl_log = 0.0
        total_clip = 0.0
        actor_grad_norm = 0.0
        critic_grad_norm = 0.0
        actor_param_max = 0.0
        critic_param_max = 0.0
        n_batches = 0
        early_stop_kl = False
        kl_stop_epoch = 0
        high_clip = False
        for epoch in range(cfg.ppo_epochs):
            epoch_kl_log = 0.0
            epoch_batches = 0
            perm = self._rng.permutation(n)
            for start in range(0, n, batch):
                idx = perm[start : start + batch]

                # -- actor output stability ------------------------------------
                dist = self.actor.dist(obs_t[idx])
                log_std = torch.log(dist.scale)
                self._check_first_nonfinite("actor_mean", dist.mean, dim=None)
                self._check_first_nonfinite("actor_log_std", log_std, dim=None)
                self._check_first_nonfinite("actor_std", dist.scale, dim=None)
                if torch.any(dist.scale <= 0):
                    self._check_first_nonfinite("actor_std_nonpositive", dist.scale, dim=None)

                actor_loss, _ratio, approx_kl, approx_kl_log, clip_frac, entropy = (
                    self._minibatch_actor(
                        dist,
                        act_raw_t[idx],
                        old_logp_t[idx],
                        adv_t[idx],
                        eps=eps,
                        ent_coef=ent_coef,
                    )
                )

                # -- value loss -------------------------------------------------
                value_pred = self._value_net(obs_t[idx])
                value_loss = F.mse_loss(value_pred, ret_t[idx])
                loss = actor_loss + val_coef * value_loss
                self._check_first_nonfinite("actor_loss", actor_loss, dim=None)
                self._check_first_nonfinite("value_loss", value_loss, dim=None)
                self._check_first_nonfinite("total_loss", loss, dim=None)

                # -- backward + gradient diagnostics + clipping ----------------
                self.actor_opt.zero_grad(set_to_none=True)
                self.value_opt.zero_grad(set_to_none=True)
                loss.backward()
                self._check_grads_finite(self.actor, "actor")
                self._check_grads_finite(self._value_net, "critic")
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self._value_net.parameters(), max_grad_norm)
                self.actor_opt.step()
                self.value_opt.step()

                # -- parameter diagnostics --------------------------------------
                actor_grad_norm = max(actor_grad_norm, grad_norm_stats(self.actor)["grad_norm"])
                critic_grad_norm = max(
                    critic_grad_norm, grad_norm_stats(self._value_net)["grad_norm"]
                )
                a_params = parameter_stats(self.actor)
                c_params = parameter_stats(self._value_net)
                actor_param_max = max(actor_param_max, a_params["max_abs_param"])
                critic_param_max = max(critic_param_max, c_params["max_abs_param"])
                if a_params["nan_params"] or a_params["inf_params"]:
                    self._check_params_finite(self.actor, "actor")
                if c_params["nan_params"] or c_params["inf_params"]:
                    self._check_params_finite(self._value_net, "critic")

                total_actor += float(actor_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_kl += approx_kl
                total_kl_log += approx_kl_log
                total_clip += clip_frac
                epoch_kl_log += approx_kl_log
                epoch_batches += 1
                n_batches += 1
                if clip_frac > 0.95:
                    high_clip = True

            # -- target-KL early stopping for this rollout ---------------------
            if target_kl > 0 and epoch_batches and (epoch_kl_log / epoch_batches) > target_kl:
                early_stop_kl = True
                kl_stop_epoch = epoch + 1
                break

        mean_abs_param_update, max_abs_param_update = self._param_update_magnitude(params_before)

        if high_clip:
            self._clip_warn_count += 1
            if self._clip_warn_count in (1, 5, 10, 20, 50) or self._clip_warn_count % 100 == 0:
                print(
                    "[ppo] WARNING clip_fraction>0.95 in a minibatch "
                    f"(env_steps={self._env_steps:,}, count={self._clip_warn_count})",
                    flush=True,
                )

        # -- stage 3: action-distribution stats for the trading policy ---------
        a = act.ravel()
        mean_action = float(np.mean(a))
        std_action = float(np.std(a))
        mean_abs_action = float(np.mean(np.abs(a)))
        frac_near_minus_one = float(np.mean(a < -0.999))
        frac_near_zero = float(np.mean(np.abs(a) < 0.001))
        frac_near_plus_one = float(np.mean(a > 0.999))

        diag = {
            "actor_loss": total_actor / max(1, n_batches),
            "critic_loss": total_value / max(1, n_batches),
            "entropy": total_entropy / max(1, n_batches),
            "approx_kl": total_kl / max(1, n_batches),
            "approx_kl_log": total_kl_log / max(1, n_batches),
            "clip_fraction": total_clip / max(1, n_batches),
            "early_stop_kl": 1.0 if early_stop_kl else 0.0,
            "kl_stop_epoch": float(kl_stop_epoch),
            "high_clip_fraction": 1.0 if high_clip else 0.0,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "actor_param_max": actor_param_max,
            "critic_param_max": critic_param_max,
            "mean_abs_parameter_update": mean_abs_param_update,
            "max_abs_parameter_update": max_abs_param_update,
            "reward_min": rew_stats.get("min") or 0.0,
            "reward_max": rew_stats.get("max") or 0.0,
            "reward_mean": rew_stats.get("mean") or 0.0,
            "reward_std": rew_stats.get("std") or 0.0,
            "return_min": ret_stats.get("min") or 0.0,
            "return_max": ret_stats.get("max") or 0.0,
            "advantage_std": adv_stats.get("std") or 0.0,
            "mean_action": mean_action,
            "std_action": std_action,
            "mean_abs_action": mean_abs_action,
            "frac_near_minus_one": frac_near_minus_one,
            "frac_near_zero": frac_near_zero,
            "frac_near_plus_one": frac_near_plus_one,
            "alpha": 0.0,
            "alpha_loss": 0.0,
            "q1": float(np.mean(val)),
            "q2": 0.0,
        }
        self._last_diag = diag
        return diag

    def _progress_postfix(self) -> dict[str, object]:
        d = getattr(self, "_last_diag", {})
        return {
            k: f"{d[k]:+.4f}"
            for k in ("entropy", "actor_loss", "approx_kl", "clip_fraction")
            if k in d
        }

    # -- checkpointing --------------------------------------------------------

    def state_dicts(self) -> dict[str, Any]:
        return {
            "critics": {"critic": self._value_net.state_dict()},
            "targets": {},
            "optimizers": {
                "actor_opt": self.actor_opt.state_dict(),
                "value_opt": self.value_opt.state_dict(),
            },
            "log_alpha": None,
            "replay_meta": {},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("policy"):
            self.actor.load_state_dict({k: torch.as_tensor(v) for k, v in state["policy"].items()})
        critics = state.get("critics", {}) or {}
        if critics.get("critic"):
            self._value_net.load_state_dict(
                {k: torch.as_tensor(v) for k, v in critics["critic"].items()}
            )
        opts = state.get("optimizers", {}) or {}
        if opts.get("actor_opt"):
            self.actor_opt.load_state_dict(opts["actor_opt"])
        if opts.get("value_opt"):
            self.value_opt.load_state_dict(opts["value_opt"])
