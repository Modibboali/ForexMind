# PPO NaN / Numerical Stability Audit

Status: root cause identified, fixes implemented, local verification green.
Kaggle 1M-step validation run is the remaining step (see §24).

## Observed failure (Kaggle)

```text
Workers: 102, Backend: process, Device: CPU, obs_dim 351, M5 rows 8,883,253
Torch threads 2, interop 112, OMP/MKL/OPENBLAS 2, ~2727 env steps/s
Failed at env_steps ~= 376,832 in torch.distributions.Normal(mean, log_std.exp()):
    ValueError: Expected parameter loc ... to satisfy the constraint Real()
    but found invalid values: tensor([[nan], ...])
```

## First non-finite value (root cause)

```text
COMPONENT:   reward  (transition at the liquidation step of an episode)
VALUE:       -inf
SOURCE:      forexmind/environment/reward.py RewardService._log_return
             -> when curr_equity <= 0 (equity collapse / liquidation wipe-out)
             it returned float("-inf")
```

### The NaN chain (verified in tests/test_training_ppo.py)

```text
liquidation (one bar moves equity <= 0)
  -> reward = -inf                       (concrete bug, was in reward.py)
  -> GAE delta_i = r_i + g*V_{i+1} - V_i = -inf
  -> raw advantages contain -inf
  -> advantage normalization (A - mean) / (std + eps) with mean=-inf, std=nan
  -> normalized advantages = NaN everywhere in the batch
  -> ratio * adv = NaN, actor_loss = NaN
  -> loss.backward() -> NaN gradients -> NaN actor weights
  -> next update: mean = mean_net(obs) = NaN
  -> torch.distributions.Normal(mean=nan, ...) raises
```

A single liquidation event anywhere in the 376k-step window poisons the whole
PPO batch. This is consistent with the failure being in the trainer
(`loc = nan`), with no prior warning.

`KNOWN FROM CODE` vs `REQUIRES KAGGLE CONFIRMATION`:
* Reward `-inf` on collapse: **known from code** (was `float("-inf")`).
* That `-inf` -> NaN advantages: **known** (hand-computed + unit test
  `test_inf_reward_poisons_gae_without_guard`).
* That a liquidation actually occurred by step 376,832 with the training
  leverage/sizing: **not directly observable locally** (the rescue checkpoint
  is on Kaggle). 100 workers x 50x leverage x equity_fraction sizing on real
  data makes it highly plausible; `tools/inspect_checkpoint.py` on the rescue
  checkpoint will confirm whether the saved actor weights are NaN.

## Fixes implemented (all justified; no trading-semantics change)

| Control | Change | Why |
| --- | --- | --- |
| `RewardConfig.min_reward` (default `-50.0`) | equity collapse now returns a finite floor instead of `-inf` | `-inf` is unusable in gradient RL; the floor is far below any real log return |
| `TrainingEnvConfig.min_reward` | plumbed into workers' `EnvironmentConfig` | workers use the same floor |
| `max_grad_norm` (default `0.5`) | actor + critic gradient clipping | prevents param explosion; 0 disables |
| `actor_lr` / `critic_lr` | independent learning rates (default to `learning_rate`) | diagnose actor vs critic divergence |
| `log_std_min` / `log_std_max` (`-5`, `2`) | bounded `log_std` (was hardcoded `LOG_STD_MIN/MAX`) | keeps `exp(log_std)` finite |
| `adv_epsilon` | normalization denominator floor + finite-std guard | degenerate batches give zero signal, not garbage |
| log-prob consistency | worker old-logp and trainer new-logp both use raw Gaussian density on the clamped action; removed the tanh-squash correction from `GaussianPolicy.evaluate` | the action is CLAMPED, not tanh-squashed; the old mismatch made `ratio = exp(new-old)` systematically wrong |
| `ratio` log-clamp | `log_ratio.clamp(-10, 10).exp()` | prevents `exp` overflow -> `inf`; the PPO clip already makes far-out ratios equivalent in the objective |
| `torch_interop_threads` | configurable (default None = torch default ~112 on Kaggle) | resource hygiene, NOT a NaN fix (audit §21) |

## Instrumentation added

`forexmind/training/numerics.py`:

```text
FiniteError, tensor_stats, first_nonfinite_index, assert_finite,
format_finite_message, parameter_stats, grad_norm_stats, first_bad_tensor
```

`PPOTrainer.update()` now validates, in documented order, stopping at the first
non-finite value when `training.finite_check=true` (raises `FiniteError` with
component / gradient_update / env_step / instrument / timestamp):

```text
observation -> next_observation -> action -> reward -> value_prediction ->
next_value_prediction -> old_log_prob -> returns -> normalized_advantages ->
actor_mean -> actor_log_std -> actor_std -> log_prob -> ratio -> actor_loss ->
value_loss -> total_loss -> actor/critic gradients -> actor/critic parameters
```

Per-update diagnostics (always on) tracked in MetricStore:

```text
actor_loss, critic_loss, entropy, approx_kl, clip_fraction,
actor_grad_norm, critic_grad_norm, actor_param_max, critic_param_max,
reward_min/max/mean/std, return_min/max, advantage_std,
mean_action, std_action, mean_abs_action,
frac_near_minus_one, frac_near_zero, frac_near_plus_one
```

`Transition` now carries `instrument` and `timestamp` so the first non-finite
observation maps to a concrete bar. `tools/inspect_checkpoint.py` scans a
checkpoint for NaN/Inf and warns that resuming a poisoned checkpoint
immediately re-diverges.

## GAE audit (verified by hand-computed tests)

```text
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
```

* `done` = `terminated` only. Truncated (horizon-limit) steps bootstrap with
  `V(s')` (correct for time-limit truncation). Verified:
  `test_gae_terminal_at_end`, `test_gae_truncation_bootstraps`,
  `test_gae_terminal_mid_batch_resets`.
* Advantage normalization uses `(A - mean) / (std + adv_epsilon)` with a
  finite/degenerate-std guard (`test_gae_normalization_standard`,
  `test_gae_normalization_degenerate_zero_std`).
* No future values leak: the recurrence walks backwards and `done` masks
  terminal boundaries (`test_gae_terminal_mid_batch_resets`).

## Action squashing audit (§17)

The PPO action is a **clamped** Gaussian sample (`torch.clamp(sample, -1, 1)`),
not a tanh-squash. The old worker code stored `log_prob = dist.log_prob(a) -
log(1 - a^2 + eps)` (a tanh-squash correction) while the trainer's update
computed `logp = dist.log_prob(a)` (no correction). That made the stored
`old_logp` inconsistent with the update's `logp`, biasing `ratio` and risking
overflow. Fixed to use the raw Gaussian density on the clamped action in both
places (see `test_ppo_worker_logprob_consistent_with_trainer` and
`test_gaussian_policy_evaluate_matches_dist_logprob`).

## Thread configuration audit (§21)

Kaggle reports `torch num_interop_threads = 112` (torch default = physical
cores) while `torch num_threads = 2` and `OMP/MKL/OPENBLAS = 2`. This is
oversubscription-prone but is **not** the NaN cause. `torch_interop_threads`
is now configurable; `configs/ppo_stable.yaml` sets it to 2.

## Local verification

* New unit tests in `tests/test_training_ppo.py` (17 tests): reward floor,
  reward `-inf` -> NaN chain, GAE hand-computed, advantage normalization,
  `finite_check` raising with instrument/timestamp context, grad clipping,
  diagnostics keys, log-std bounds, log-prob consistency.
* Full suite: `python -m pytest` green; `ruff check` and `mypy` clean.
* `python -m forexmind.training.train_ppo --config configs/ppo_smoke.yaml`
  completes 2048 steps with the new numerics code and finite checkpoints.
* `python -m tools.inspect_checkpoint <run>` reports all-finite.

## Validation plan (next step on Kaggle)

```bash
# 1M-step numerical-stability validation (finite_check raises on any NaN)
python -m forexmind.training.train_ppo --config configs/ppo_stable.yaml --seeds 1

# if the 1M run is clean, confirm the rescue checkpoint is poisoned:
python -m tools.inspect_checkpoint runs/ppo_cpu_seed42/checkpoints/rescue_step_376832.pt

# binary-search the original failure region (only if needed for the record):
python -m forexmind.training.train_ppo --config configs/ppo_cpu.yaml \
  --max-env-steps 375000 --seeds 1
```

Do NOT resume the 376,832 rescue checkpoint until `inspect_checkpoint`
confirms its weights are finite (they almost certainly are not, because the
run crashed inside an update after the weights went NaN).

## Acceptance criteria checklist (§25)

1. No NaN/Inf during a 1M-step validation run -> `finite_check=true` run
   completes (pending Kaggle).
2. Actor/critic params finite -> tracked (`actor_param_max`/`critic_param_max`).
3. Actor/critic gradients finite -> tracked (`actor_grad_norm`/`critic_grad_norm`).
4. mean/log_std/std finite -> checked every minibatch.
5. rewards/returns/advantages finite -> checked every update.
6. Approx KL monitored (`approx_kl`).
7. Clip fraction monitored (`clip_fraction`).
8. No sudden param explosion -> `actor_param_max`/`critic_param_max`.
9. Throughput close to production -> compare steps/s against the 2727 baseline.
10. Environment/accounting tests still pass -> `python -m pytest` green.
