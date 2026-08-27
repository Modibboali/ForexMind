"""PPO trainer (Phase 3).

On-policy proximal policy optimization with a Gaussian policy (clipped
objective), a learned value baseline, GAE returns, and an entropy bonus.
The collector performs full on-policy rollouts: each collection round the
learner re-syncs its current policy/value to the workers, collects a batch,
then runs several PPO epochs over that batch.
"""

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
from forexmind.training.trainer import BaseTrainer


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
        self.actor = GaussianPolicy(self.obs_dim, self.action_dim, model)
        self._value_net = ValueNet(self.obs_dim, model)
        self.actor.to(self.device)
        self._value_net.to(self.device)
        self.actor.eval()
        self._value_net.eval()

        lr = config.training.learning_rate
        self.actor_opt = Adam(self.actor.parameters(), lr=lr)
        self.value_opt = Adam(self._value_net.parameters(), lr=lr)

        self._rollout: list[Transition] = []
        self._rng = np.random.default_rng(config.compute.seed)

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

    # -- PPO update -----------------------------------------------------------

    def _gae(
        self,
        rew: np.ndarray,
        done: np.ndarray,
        val: np.ndarray,
        next_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        std = adv.std()
        adv = (adv - adv.mean()) / (std + 1e-8)
        return adv.astype(np.float32), returns

    def update(self) -> dict[str, float]:
        rollout = self._rollout
        n = len(rollout)
        if n == 0:
            return {}
        obs = np.stack([t.obs for t in rollout]).astype(np.float32)
        act = np.asarray([t.action for t in rollout], dtype=np.float32).reshape(-1, 1)
        rew = np.asarray([t.reward for t in rollout], dtype=np.float32)
        done = np.asarray([t.terminated for t in rollout], dtype=bool)
        old_logp = np.asarray([t.log_prob for t in rollout], dtype=np.float32).reshape(-1, 1)
        val = np.asarray([t.value for t in rollout], dtype=np.float32)
        next_val = np.asarray([t.next_value for t in rollout], dtype=np.float32)

        adv, returns = self._gae(rew, done, val, next_val)

        obs_t = torch.as_tensor(obs, device=self.device)
        act_t = torch.as_tensor(act, device=self.device)
        old_logp_t = torch.as_tensor(old_logp, device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device).unsqueeze(1)
        ret_t = torch.as_tensor(returns, device=self.device).unsqueeze(1)

        cfg = self.config.training
        eps = cfg.ppo_clip_epsilon
        ent_coef = cfg.ppo_entropy_coef
        val_coef = cfg.ppo_value_coef
        batch = cfg.batch_size

        total_actor = 0.0
        total_value = 0.0
        total_entropy = 0.0
        n_batches = 0
        for _ in range(cfg.ppo_epochs):
            perm = self._rng.permutation(n)
            for start in range(0, n, batch):
                idx = perm[start : start + batch]
                dist = self.actor.dist(obs_t[idx])
                logp = dist.log_prob(act_t[idx]).sum(-1, keepdim=True)
                entropy = dist.entropy().sum(-1, keepdim=True).mean()
                ratio = (logp - old_logp_t[idx]).exp()
                surr1 = ratio * adv_t[idx]
                surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv_t[idx]
                actor_loss = -(torch.min(surr1, surr2).mean() + ent_coef * entropy)
                value_pred = self._value_net(obs_t[idx])
                value_loss = F.mse_loss(value_pred, ret_t[idx])
                loss = actor_loss + val_coef * value_loss

                self.actor_opt.zero_grad(set_to_none=True)
                self.value_opt.zero_grad(set_to_none=True)
                loss.backward()
                self.actor_opt.step()
                self.value_opt.step()

                total_actor += float(actor_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.item())
                n_batches += 1

        diag = {
            "actor_loss": total_actor / max(1, n_batches),
            "critic_loss": total_value / max(1, n_batches),
            "entropy": total_entropy / max(1, n_batches),
            "alpha": 0.0,
            "alpha_loss": 0.0,
            "q1": float(np.mean(val)),
            "q2": 0.0,
        }
        self._last_diag = diag
        return diag

    def _progress_postfix(self) -> dict[str, object]:
        d = getattr(self, "_last_diag", {})
        return {k: f"{d[k]:+.4f}" for k in ("entropy", "actor_loss") if k in d}

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
            self.actor.load_state_dict(
                {k: torch.as_tensor(v) for k, v in state["policy"].items()}
            )
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
