"""SAC trainer (Phase 3).

Off-policy soft actor-critic with continuous target exposure in [-1, 1],
twin critics + target critics, and automatic entropy-temperature tuning.
Only training-split transitions enter the replay buffer (the collector is
configured with ``split="train"``).
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
from forexmind.training.networks import build_sac_networks, soft_update
from forexmind.training.replay import ReplayBuffer
from forexmind.training.trainer import BaseTrainer


class SACTrainer(BaseTrainer):
    def __init__(
        self,
        config: ExperimentConfig,
        run_dir: str | Path,
        *,
        dataset: Any = None,
    ) -> None:
        super().__init__(config, run_dir, dataset=dataset)
        model = config.model
        nets = build_sac_networks(self.obs_dim, self.action_dim, model)
        self.actor = nets.actor
        self.critic = nets.critic
        self.target_critic = nets.target_critic
        for m in (self.actor, self.critic, self.target_critic):
            m.to(self.device)
        self.target_critic.eval()

        lr = config.training.learning_rate
        self.actor_opt = Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)

        auto_alpha = str(config.training.alpha) == "auto"
        self.auto_alpha = auto_alpha
        self.log_alpha = torch.tensor(
            np.log(1.0) if auto_alpha else np.log(float(config.training.alpha)),
            dtype=torch.float32,
            requires_grad=auto_alpha,
            device=self.device,
        )
        self.alpha_opt: Adam | None = Adam([self.log_alpha], lr=lr) if auto_alpha else None
        target_entropy = config.training.target_entropy
        if target_entropy is None:
            target_entropy = -float(self.action_dim)
        self.target_entropy = target_entropy

        self.replay = ReplayBuffer(
            self.obs_dim,
            config.training.replay_capacity,
            self.action_dim,
            seed=config.compute.seed,
        )
        self._last_diag: dict[str, float] = {}

    # -- BaseTrainer interface ------------------------------------------------

    def policy(self) -> nn.Module:
        return self.actor

    def value_net(self) -> nn.Module | None:
        return None

    def replay_size(self) -> int:
        return self.replay.size

    def _consume_transitions(self, transitions: list[Transition]) -> None:
        for t in transitions:
            self.replay.push(t.obs, t.action, t.reward, t.next_obs, t.terminated, t.truncated)
        if self._env_steps < self.config.training.warmup_steps:
            return
        if self.replay.size < self.config.training.batch_size:
            return
        n_updates = max(
            1, int(self.config.training.gradient_updates_per_step * len(transitions))
        )
        for _ in range(n_updates):
            diag = self.update()
            self._gradient_updates += 1
            self._record_diagnostics(diag)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    # -- SAC update -----------------------------------------------------------

    def update(self) -> dict[str, float]:
        batch = self.replay.sample(self.config.training.batch_size)
        obs = torch.as_tensor(batch.obs, device=self.device)
        act = torch.as_tensor(batch.action, device=self.device)
        rew = torch.as_tensor(batch.reward, device=self.device).unsqueeze(1)
        nxt = torch.as_tensor(batch.next_obs, device=self.device)
        done = torch.as_tensor(
            batch.terminated, device=self.device, dtype=torch.float32
        ).unsqueeze(1)

        gamma = self.config.training.gamma
        alpha = self.log_alpha.exp()

        # -- critics ----------------------------------------------------------
        with torch.no_grad():
            next_act, next_logp = self.actor.sample(nxt)
            q1t, q2t = self.target_critic(nxt, next_act)
            min_q = torch.min(q1t, q2t) - alpha * next_logp
            target = rew + gamma * (1.0 - done) * min_q
        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # -- actor ------------------------------------------------------------
        pi_a, pi_logp = self.actor.sample(obs)
        q1p, q2p = self.critic(obs, pi_a)
        actor_loss = (alpha * pi_logp - torch.min(q1p, q2p)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # -- alpha (automatic entropy tuning) ---------------------------------
        alpha_loss = torch.zeros(1, device=self.device)
        if self.auto_alpha and self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (pi_logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

        # -- target soft update ----------------------------------------------
        soft_update(self.target_critic, self.critic, self.config.training.tau)

        with torch.no_grad():
            entropy = self.actor.entropy(obs).mean()
            q1_mean = q1.mean().item()
            q2_mean = q2.mean().item()
        diag = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(alpha.item()),
            "entropy": float(entropy.item()),
            "q1": float(q1_mean),
            "q2": float(q2_mean),
        }
        self._last_diag = diag
        return diag

    def _progress_postfix(self) -> dict[str, object]:
        d = self._last_diag
        return {k: f"{d[k]:+.4f}" for k in ("alpha", "entropy", "q1") if k in d}

    # -- checkpointing --------------------------------------------------------

    def state_dicts(self) -> dict[str, Any]:
        return {
            "critics": {
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
            },
            "targets": {},
            "optimizers": {
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict() if self.alpha_opt else None,
            },
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
            "replay_meta": self.replay.state_meta(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("policy"):
            self.actor.load_state_dict(
                {k: torch.as_tensor(v) for k, v in state["policy"].items()}
            )
        critics = state.get("critics", {}) or {}
        if critics.get("critic"):
            self.critic.load_state_dict(
                {k: torch.as_tensor(v) for k, v in critics["critic"].items()}
            )
        if critics.get("target_critic"):
            self.target_critic.load_state_dict(
                {k: torch.as_tensor(v) for k, v in critics["target_critic"].items()}
            )
        opts = state.get("optimizers", {}) or {}
        if opts.get("actor_opt"):
            self.actor_opt.load_state_dict(opts["actor_opt"])
        if opts.get("critic_opt"):
            self.critic_opt.load_state_dict(opts["critic_opt"])
        if opts.get("alpha_opt") and self.alpha_opt is not None:
            self.alpha_opt.load_state_dict(opts["alpha_opt"])
        if state.get("log_alpha") is not None:
            with torch.no_grad():
                self.log_alpha.copy_(
                    torch.as_tensor(np.asarray(state["log_alpha"], dtype=np.float32))
                )
        replay_meta = (state.get("rng", {}) or {}).get("replay_meta", {}) or {}
        if replay_meta:
            self.replay.load_meta(replay_meta)
