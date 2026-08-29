# PPO Mathematical Audit & Stability Fix

## Observed failure

A PPO run showed catastrophic divergence (not merely noise):

```text
~102k steps  approx_kl 0.011   clip 0.15   entropy 1.36
~503k steps  approx_kl 0.032   clip 0.40   entropy 1.25
~1.003M      approx_kl 16.4    clip 0.52
1.10M-2.00M  approx_kl 3,058 -> 11,377 -> 69,395 -> 173,807
             -> 8,935,937 -> 50,684,038 -> 90,868,830 -> 93,342,748
clip_fraction -> ~0.99, mean_abs_action -> ~1.0, actor_loss -> thousands
eventual: ValueError Normal(mean, log_std.exp()) loc = nan
```

The validation Sharpe of 19.94 / Sortino 19.71 is treated as UNVERIFIED until a
stable model reproduces a credible result (§ task: the policy saturated at
extreme actions before that evaluation).

## Audit results (per the required sections)

### OLD LOG-PROB IMPLEMENTATION — INCORRECT (primary bug)

The actor is a **clamped Gaussian**: `u ~ N(mu, sigma)` then `a = clamp(u, -1, 1)`
sent to the environment. The old code stored `old_log_prob = log N(a; mu_old,
sigma_old)` — the Gaussian density evaluated at the **clamped boundary a**,
not at the actual sample `u`. The new code also computed `new_log_prob` at the
clamped `a`. For the increasingly common clamped samples (as the policy
saturated, `mean_abs_action -> 1.0`), the boundary density does not correspond
to the density of the sample that was actually drawn, so the importance ratio

```text
r_t = exp( log pi_theta(a_t|s_t) - log pi_theta_old(a_t|s_t) )
```

was not the true density ratio. As `mu` moved, `N(a=+-1; mu)` swung by orders of
magnitude, producing exploding ratios and corrupting the gradient signal.

**Fix:** store the raw pre-clamp sample `u` in the rollout (`Transition.action_raw`)
and compute BOTH `old_log_prob` and `new_log_prob` at `u` — the exact density of
the Gaussian sampling distribution. The environment-facing action stays
`clamp(u)` (trading semantics unchanged). The clamp is now correctly treated as
an execution-time projection, not part of the policy density.

### TANH / ACTION TRANSFORMATION — clamp, not tanh (documented + made consistent)

There is **no tanh**; actions are clamped. A tanh Jacobian correction is
therefore neither applicable nor added (adding one would be wrong). The prior
code applied *no* correction but evaluated at the clamped boundary, which was
the inconsistency. With `action_raw` the density is exact for the Gaussian
sampling distribution. (The previous, pre-audit code had a *different* bug: a
tanh correction in the worker that the trainer did not apply — fixed in an
earlier session; this audit removed the remaining boundary-density issue.)

### PPO RATIO — correct now

`log_ratio = clamp(logp_new - old_logp, -10, 10)`; `ratio = exp(log_ratio)`.
The clamp is only a float-overflow safety net; with the corrected log-probs and
target-KL it never binds (observed clip_fraction ~0 locally). `old_logp` is a
plain (non-grad) tensor; no gradients flow through it.

### CLIPPED OBJECTIVE — correct

```python
surr1 = ratio * A
surr2 = clamp(ratio, 1-eps, 1+eps) * A
actor_loss = -( min(surr1, surr2).mean() + ent_coef * entropy )   # minimization
```
Signs, min/max, clip-on-ratio, and entropy bonus are correct; `A` is a detached
constant. Verified by the hand-computed test.

### GAE — correct

`delta_t = r_t + gamma*(1-d_t)*V_{t+1} - V_t`; `A_t = delta_t + gamma*lambda*
(1-d_t)*A_{t+1}`. `done = terminated` only; truncation bootstraps with V(s')
(correct). Verified by hand-computed tests (terminal / truncation / mid-batch).

### KL CALCULATION — correct, but was reporting only one estimator

`approx_kl = E[0.5 (r-1)^2]` (kept for continuity). Added
`approx_kl_log = E[r - 1 - log r]` (the standard SB3-style estimator) and used
it for target-KL early stopping. The huge observed KL was REAL divergence, not
a metric bug: ratios were exploding.

### ADVANTAGE NORMALIZATION — correct

`A = (A - mean)/(std + eps)` over the whole rollout, detached; finite/degenerate
std guard; `advantage_std ~ 1` confirmed.

### DISTRIBUTION — log_std bounded, mean now stays finite

`log_std` is a single learned parameter bounded to `[log_std_min, log_std_max]`
(defaults -5, 2), so `std` is finite. `mean` is unconstrained but stays finite
once the gradient signal is correct + clipped.

### ACTION SATURATION — a symptom of divergence, not a learned goal

`mean_abs_action -> 1.0` followed the ratio explosion (the policy was being
pushed to the clamp boundary by corrupt gradients). With the fixes, entropy
stays ~1.4 and actions are not pinned at +-1.

### ACTION-TO-ENV ADAPTER — correct

`actor -> u -> clamp(u) -> target exposure -> env.step()`; no extra scaling,
sign flip, or duplicate tanh. `PolicyAgent` uses `clamp(mean)` deterministically.

### REWARD SCALE — fine (~+-0.005); unchanged

### CRITIC — healthy; actor divergence was not caused by the critic targets

### ROLLOUT LIFECYCLE — correct; `old_policy` fixed for all epochs

`old_logp`/`action_raw` are stored once per rollout and never recomputed; the
collector is re-synced before each rollout (on-policy).

## Fixes applied

1. `Transition.action_raw` — the raw pre-clamp Gaussian sample (new field).
2. Worker PPO branch stores `log_prob = log N(u)` at the raw sample; env action
   stays `clamp(u)`.
3. Trainer uses `action_raw` for `new_log_prob`; diagnostics still use the
   env-facing `action`.
4. `ppo_target_kl` (default 0.02, 0 disables): stop the remaining PPO epochs
   for a rollout when the KL estimate exceeds it; records `early_stop_kl` and
   `kl_stop_epoch`.
5. `max_grad_norm` gradient clipping (already present, kept at 0.5).
6. `mean_abs_parameter_update` / `max_abs_parameter_update` diagnostics.
7. `clip_fraction > 0.95` repeated-update warning.
8. `approx_kl_log` second KL estimator for early stopping + reporting.
9. Stable config: `ppo_epochs 3`, `ppo_target_kl 0.02`, `finite_check true`,
   `actor_lr/critic_lr 3e-4`, `max_grad_norm 0.5`, log_std bounds [-5, 2].

## Local stability evidence (20k steps, sync, EURUSD, finite_check=true)

```text
approx_kl       ~0.0000-0.0006   (was 0.01 -> 93M)
clip_fraction   ~0.000-0.0013    (was -> 0.99)
entropy         ~1.40            (healthy)
actor_loss      ~-0.015          (stable)
NaN/Inf         none (finite_check completed)
```

Process-backend smoke (2 workers, 2k steps): KL ~0.006-0.012, clip ~0.07-0.11,
no NaN, clean shutdown.

## Tests

* `tests/test_training_ppo.py::test_ppo_minibatch_math_hand_computed` — manual
  old/new log-prob, ratio, clipped objective, both KL estimators, entropy vs
  the implementation (mandatory §18).
* `test_ppo_worker_logprob_consistent_with_trainer` — stored log-prob equals
  N(u) at the raw sample; env action is clamp(u).
* `test_ppo_target_kl_early_stop` — tiny target-KL stops remaining epochs.
* Existing GAE, advantage-normalization, grad-clip, finite-check, log-std tests
  all pass.

Full suite (`pytest`), `ruff`, and `mypy` are green.

## Remaining validation (Kaggle)

```bash
python -m forexmind.training.train_ppo --config configs/ppo_stable.yaml --seeds 1 \
    --max-env-steps 100000     # 100k: finite, sane KL/clip, no explosion
# then (fresh run dir/seed):
python -m forexmind.training.train_ppo --config configs/ppo_stable.yaml --seeds 1
                              # 2M-step stable-config run (ppo_stable.yaml)
```

Do NOT resume the divergent checkpoint. Start a fresh run; the previous
19.94-Sharpe checkpoint is retained for forensics only.
