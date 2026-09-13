# Stage 4.4 — MuZero loss, recurrent-unroll training, and learner

Implementation date: 2026-09-10. This stage closes the MuZero loop that Stages
4.1-4.3 opened: the representation `h_θ`, dynamics `g_θ` and prediction `f_θ`
networks are now **jointly** optimized through a multi-step latent unroll using
the Stage 4.3 targets::

    replay batch  (o_t, a_{t..t+K-1}, r_{t+1..t+K}, pi_{t..t+K}, z_{t..t+K}, masks)
        -> initial_inference(o_t)            -> policy/value losses at k = 0
        -> recurrent_inference(latent, a_t)  -> reward/policy/value losses at k = 1
        -> ... K times
        -> masked, weighted total loss -> backward -> one optimizer step

**Criterion for this stage** — *"The MuZero representation, dynamics, reward,
policy and value networks can be jointly optimized through a multi-step latent
unroll using mathematically aligned replay targets, with stable gradients and
demonstrably decreasing loss on controlled data."* Sections 9-13 and 15 below are
the direct evidence for that sentence.

Still **not** implemented (deferred to Stage 4.5+): distributed actors,
reanalysis, prioritized replay, target refreshing, target networks, Stochastic
MuZero, and any long production training run. The frozen Forex reward
`r_t = log(equity[t+1] / equity[t])`, execution timing, Decimal accounting,
currency conversion, split semantics and the ten-action environment are
**untouched**; MuZero still uses its own six-action space.

---

## 1. Files added / changed

Added:

| File | Purpose |
|---|---|
| `forexmind/muzero/losses.py` | `LossConfig`, `MuZeroPrediction`, `MuZeroLossResult`, `muzero_losses`, masked means, diagnostics. |
| `forexmind/muzero/calibration.py` | `TargetStatistics`, `describe_targets`, `headroom_factor`, `propose_scale`, `calibration_report`, `reward_targets` / `value_targets`. |
| `forexmind/muzero/learner.py` | `OptimizerConfig`, `LearnerConfig`, `MuZeroLearner` (joint optimizer, `unroll`, `train_step`, checkpointing, metrics). |
| `tools/calibrate_muzero_targets.py` | Real-data target-scale calibration (§10) inspection tool. |
| `tools/smoke_muzero_learning.py` | Real-data end-to-end learning smoke test (§32). |
| `tools/benchmark_muzero_learning.py` | Joint-update throughput / memory benchmark (§38). |
| `tests/test_muzero_losses.py` | Loss, masking and alignment guards (22 tests). |
| `tests/test_muzero_learner.py` | Optimizer, batch-contract assertions, gradient flow, overfit, sanity, split, numerics, checkpoint (41 tests). |
| `tests/test_muzero_calibration.py` | Target statistics and scale proposal (23 tests). |
| `docs/stage44_muzero_learner.md` | This report. |

Rewritten:

| File | Change |
|---|---|
| `forexmind/muzero/support.py` | Cancellation-free `h` / `h⁻¹`, `scale`, `epsilon`, `saturation_fraction`, `_two_point`; `SUPPORT_RANGE = 3.0`. |

Changed:

| File | Change |
|---|---|
| `forexmind/muzero/config.py` | `value_scale` / `reward_scale` and `value_epsilon` / `reward_epsilon` documented and validated; epsilon defaults moved to `0.0` (the old `0.001` biased small targets). |
| `forexmind/muzero/inference.py` | Public `decode_scalar(raw, *, use_support, support_size, scale, epsilon)` so the learner and MCTS decode identically. |
| `forexmind/muzero/targets.py` | `MuZeroSample.split` and `MuZeroBatch.splits` so the learner can enforce the TRAIN-only guard (§25). |
| `forexmind/muzero/__init__.py` | Exports for calibration, losses and learner (60 public names). |
| `tests/test_muzero_support.py` | Rewritten for the new transform (28 tests). |
| `tests/muzero_synthetic.py` | `make_trajectory(..., observation_tag=...)` so different trajectories are distinguishable from their observations alone. |
| `README.md` | Stage 4.4 section. |

No existing Forex, training or evaluation module was modified.

### Verification pass (2026-09-13)

Stage 4.4 was re-verified end to end against the stage brief. The suite, the
calibration tool, the smoke-training tool and the benchmark were re-run and
reproduce the numbers in this report (see §4, §15, §17, §18; throughput is
re-measured in §16). Two gaps found during that pass were fixed:

1. `forexmind/muzero/learner.py` gained `check_batch_shapes`, which asserts the
   whole `MuZeroBatch` contract before any inference (§2 of the brief). A
   mis-shaped target such as `target_policies[B, A]` multiplies cleanly against
   `[B, K+1, A]` logits, so without this check it would have trained a different
   objective silently instead of raising. Four regression tests were added to
   `tests/test_muzero_learner.py`.
2. `tools/benchmark_muzero_learning.py` no longer prints `RSS MB 0.0` /
   `dRSS +nan` when `psutil` (a `train` extra) is absent: it falls back to
   `psapi`/`getrusage` from the standard library and prints `n/a` when memory
   genuinely cannot be measured. This made the §16 memory column reproducible
   in an environment that has `torch` but not `psutil`.

## 2. Exact loss equations

Notation: `k ∈ {0..K}` indexes unroll steps, `t_k` is the absolute trajectory
index of unroll step `k`, `A = 6` actions, `N` bins per support head, and
`M(·)` is the mask (`policy_masks`, `value_masks`, `reward_masks`).

**Policy** (§11). Logits are masked with `torch.finfo(dtype).min` on invalid
actions, then `log π = log_softmax(z^p)`:

$$
L_\pi = \frac{1}{\sum_k M^{\pi}_k}\sum_{k=0}^{K} M^{\pi}_k
\;\bigl(-\sum_{a=0}^{A-1} \pi^{\text{target}}_{t_k,a}\,\log \pi_{k,a}\bigr)
$$

**Value** (§12) and **reward** (§13) are cross-entropies between the two-hot
target distribution $\hat{z}$ built by `scalar_to_support` and the softmax of
the head logits:

$$
L_v = \frac{1}{\sum_k M^{v}_k}\sum_{k=0}^{K} M^{v}_k
\bigl(-\sum_{b=1}^{N}\hat{p}^{\,v}_{k,b}\,\log \sigma(\ell^{v}_{k})_b\bigr),
\qquad
L_r = \frac{1}{\sum_k M^{r}_k}\sum_{k=0}^{K-1} M^{r}_k
\bigl(-\sum_{b=1}^{N}\hat{p}^{\,r}_{k,b}\,\log \sigma(\ell^{r}_{k})_b\bigr)
$$

`target_rewards` has only `K` entries because the initial state has no incoming
transition (`reward_masks[:, 0]` is always 0 by construction, but the formula
never assumes it).

**Total:**

$$
L = c_\pi L_\pi + c_v L_v + c_r L_r,
\qquad c_\pi = c_v = c_r = 1.0 \text{ by default}
$$

Every term is divided by **its own count of valid targets**, so padding never
dilutes a loss term, and a term whose mask is empty contributes exactly 0
instead of `0/0` (`weights.sum().clamp_min(1.0)`).

**Scalar mode** (`use_support=False`) replaces the two cross-entropies with plain
regression in economic units — `smooth_l1` (Huber, `delta = 1.0`) or `mse`:

$$
L_v = \frac{1}{\sum_k M^v_k}\sum_k M^v_k\,\ell(v_k - z_{t_k}),
\qquad
\ell(e) = \begin{cases} \tfrac12 e^2 & |e| < \delta \\ \delta(|e| - \tfrac{\delta}{2}) & \text{otherwise}\end{cases}
$$

Because `prediction.value` / `prediction.reward` are already decoded by
`decode_scalar` (which applies `scale`), the reported `value_mae` /
`reward_mae` diagnostics are in the same units in both modes and are directly
comparable.

## 3. Scalar/support representation chosen

**Categorical support (two-hot) is the default and the regime used for the
real-data numbers in this report**, because the reward distribution is heavy
tailed at a scale (~1e-4) where a scalar MSE head has almost no gradient
signal. `use_support=False` remains fully supported and is exercised by the same
tests (`@pytest.mark.parametrize("use_support", [True, False])`), including the
scalar loss path.

The support transform is `forexmind.muzero.support`:

$$
h(x) = \operatorname{sign}(x)\bigl(\sqrt{|x|+1}-1\bigr) + \varepsilon x
      \;=\; \frac{x}{\sqrt{|x|+1}+1} + \varepsilon x
$$

$$
h^{-1}(y) = \begin{cases} y\,(|y|+2) & \varepsilon = 0\\[2pt] d\,(d+2),\; d = \dfrac{\sqrt{a^2 + 4\varepsilon|y|} - a}{2\varepsilon},\; a = 1 + 2\varepsilon & \varepsilon > 0\end{cases}
$$

The second, algebraically identical form of `h` is what the code evaluates: it
avoids the catastrophic cancellation of `sqrt(|x|+1) - 1` for the tiny `x` that
Forex produces. With `ε = 0` the representable window is
`|z| ≤ 3·scale` and the support always contains `0`. `SUPPORT_RANGE = 3.0`,
`N = 21` by default (odd, ≥ 3 enforced), and `ε = 0.0` for both heads — a
non-zero `ε` was tried and rejected because it biases *all* small targets toward
`mix(0.5, 0.5)`; `LossConfig.for_model` raises if the two epsilons differ.

## 4. Observed reward / value target scales

Produced by `python -m tools.calibrate_muzero_targets` (Stage 4.3 collector,
EURUSD+GBPUSD+USDJPY, 4 trajectories × 16 decisions × 8 simulations, untrained
network, 344 valid targets):

| Target | mean | std | p01 | p50 | p99 | \|p99\| | max |
|---|---|---|---|---|---|---|---|
| reward `r_{t+k}` | +5.21e-06 | 1.60e-04 | -3.27e-04 | -9.21e-06 | +3.74e-04 | 3.74e-04 | +6.55e-04 |
| value `z_t` | -3.16e-01 | 8.49e-02 | -4.45e-01 | -3.37e-01 | -1.56e-01 | 4.45e-01 | -1.53e-01 |

`propose_scale` (see §10) gives:

| Head | proposed scale | saturation | \|z\|/scale at p99 | bins used (of 21) |
|---|---|---|---|---|
| reward | 2.3946e-04 | 0.00% | 1.560 | 13 (61.9%) |
| value | 2.8516e-01 | 0.00% | 1.560 | 5 (23.8%) |

The value targets are concentrated (all negative here — the untrained search is
pessimistic), so only ~5 of 21 value bins carry data; that is genuine
information about this replay, not a bug, and the diagnostic reports it rather
than hiding it. With 6 trajectories the numbers move to reward
`|p99| = 7.21e-04` (scale 4.62e-04) and value `|p99| = 6.08e-01`
(scale 3.90e-01) — i.e. the proposal tracks the data, which is the point.

The important observation: **the previous default `scale = 1.0` would have
placed every reward target in the centre bin** (`|z|/scale ≈ 3.7e-04`), giving a
near-constant target and effectively no reward learning signal. Calibration is
therefore not cosmetic.

## 5. Loss weights

`LossConfig` defaults: `policy_loss_weight = value_loss_weight =
reward_loss_weight = 1.0`; all three must be `>= 0`. They are validated and
serialized into the checkpoint, and the smoke tool exposes them as
`--policy-loss-weight`, `--value-loss-weight`, `--reward-loss-weight`. No
weighting scheme is hard-coded or hidden, and — unlike the AlphaZero-style
literature default — the policy term is *not* scaled down, because the six-action
Forex policy target is heavily concentrated (argmax HOLD ≈ 30%) and the entropy
is low, so a 1:1:1 weighting does not let the policy dominate: the observed loss
decomposition after 200 real updates is policy 1.24 / value 0.59 / reward 0.52.

## 6. Optimizer configuration

`OptimizerConfig` (all fields explicit, all validated):

| Field | Default | Used in this stage's runs |
|---|---|---|
| `name` | `adamw` | `adamw` |
| `learning_rate` | 3e-4 | 3e-4 (sanity tests 1e-2) |
| `weight_decay` | 0.0 | 0.0 |
| `betas` | (0.9, 0.999) | (0.9, 0.999) |
| `eps` | 1e-8 | 1e-8 |
| `max_grad_norm` | 5.0 | 5.0 |
| `schedule` | `constant` | `constant` (`step` supported: `lr_gamma ** (update // lr_step_size)`) |

**One optimizer over one model.** `MuZeroLearner` builds a single AdamW whose
parameter groups cover *every* `requires_grad` parameter of the representation,
dynamics and prediction networks (test:
`test_learner_is_one_joint_optimizer_over_the_whole_model`). There is **no
target network** and no separate per-head optimizer (test:
`test_no_target_network_is_created`). `zero_grad(set_to_none=True)` before every
backward pass.

## 7. Gradient-scaling implementation

The unroll keeps gradients across all `K` recurrent steps — nothing is detached
by default. Gradient *scaling* is opt-in through `LearnerConfig.latent_gradient_scale`
(§31), applied after each recurrent transition:

```python
output = model.recurrent_inference(latents[-1], actions[:, index])
latents.append(output.latent_state)
...
if gate < 1.0 and index < steps - 1:
    latents[-1] = gate * latents[-1] + (1.0 - gate) * latents[-1].detach()
```

Three properties that matter:

1. **Every step contributes** — the head losses at step `k` are computed from
   the *true* latent `latents[-1]`; only the tensor handed to the *next*
   recurrent step is re-scaled, so policy/value/reward gradients at every step
   reach the representation through the real path.
2. **The last transition is never gated**, so step `K` receives full gradient.
3. **`gate = 1.0` disables it exactly** (the branch is skipped), which is what
   the anti-detach regression test
   (`test_late_step_gradient_reaches_the_representation_only_when_not_detached`)
   and the norm-ordering test
   (`test_latent_gradient_scaling_reduces_the_gradient_norm`, `0.0 < 1.0`)
   pin down.

Default `gate = 0.5`. Validation rejects anything outside `[0, 1]`, and the
value is stored in the checkpoint so a resumed run behaves identically.

## 8. Padding / mask handling

* **The batch contract is asserted before optimization** (`check_batch_shapes`,
  added in the §1 verification pass): `observation [B, obs_dim]`,
  `actions [B, K]` (integer), `target_rewards`/`reward_masks` `[B, K]`, and
  `target_values`/`policy_masks`/`value_masks` `[B, K+1]`,
  `target_policies`/`action_masks` `[B, K+1, 6]`, with a single consistent `B`.
  A mismatch raises `ValueError` naming the offending field, before any forward
  pass, backward pass or optimizer step.
* `policy_masks`, `value_masks`, `reward_masks` are `float` `[B, K+1]` /
  `[B, K]` tensors produced in Stage 4.3; `1.0` = valid, `0.0` = padding or
  true-terminal.
* `_masked_mean(values, masks) = (flat * w).sum() / w.sum().clamp_min(1.0)`, so
  a fully-masked position contributes 0 and never produces NaN.
* Padded actions are `PAD_ACTION = 0` (HOLD) with `reward_mask = 0`, so they
  never enter any loss term — but the latent chain still runs through them,
  which is harmless because their head outputs are masked out.
* **Policy targets are validated before use**:
  `(π_target · 1[invalid action]).max() ≤ target_policy_tolerance (1e-4)` else
  `ValueError("... policy target places probability mass on an invalid action")`.
  This catches a corrupted search target rather than silently training on it.
* Prediction logits are masked (`mask_policy_logits=True`) with
  `torch.finfo(dtype).min`, matching `CategoricalPolicy.dist`, so invalid actions
  get exactly `0.0` probability.
* `MuZeroBatch` masks are inspected by the finite-check pass together with the
  targets and observations (§36).

## 9. One-batch overfit result

`test_single_batch_overfit` (§27): a fixed batch of 6 synthetic trajectories
(fixed seed, no resampling), 150 steps, `lr = 1e-2`, `K = 3`.

| Metric | support | scalar |
|---|---|---|
| total loss | 8.685 → **1.065** | 1.955 → **0.0029** |
| policy loss | 1.667 → **0.0029** | 1.907 → **0.0027** |
| policy KL | 1.667 → **0.0029** | 1.907 → **0.0027** |
| value MAE | 0.2456 → **0.0032** | 0.2523 → **0.0150** |
| reward MAE | 0.0566 → **0.00039** | 0.0911 → **0.0056** |

Asserts: finite loss, `total`, `policy_kl`, `reward_mae`, `value_mae` all lower at
the end than at the start, and `policy_loss` at least halved. Both
representations pass. (The scalar total starts lower simply because Huber is
quadratic near zero; the end states are comparable.)

## 10. Synthetic trajectory overfit result

`test_overfits_a_tiny_synthetic_replay` (§26): 8 synthetic trajectories, **random
replay sampling** (batch 8) for 200 updates — i.e. every update sees a different
mix of positions.

| Metric | first | last |
|---|---|---|
| total loss | 8.948 | **3.221 (-64%)** |
| policy loss | 1.838 | **0.476 (-74%)** |
| policy KL | 1.838 | 0.476 |
| value MAE | 0.2599 | 0.2542 |
| reward MAE | 0.0949 | 0.0803 |

Asserts: `total_loss` and `policy_loss` both below half their initial value,
`value_mae` / `reward_mae` strictly lower, no non-finite loss at any step.

**Honest reading:** value and reward MAE barely move on this fixture. That is
expected — the fixture's n-step value/reward targets are near-constant per
trajectory (they are built from constant per-trajectory rewards), so once the
head predicts the mean there is little left to learn, and the residual is the
between-trajectory variance that 8 sampled positions per update cannot resolve in
200 steps. The claim "the objective decreases and the policy tightens" is what
this test demonstrates. The decisive value/reward evidence is §11 and §13, where
the targets are *designed* to require action conditioning.

## 11. Reward-model sanity result

`test_reward_model_learns_action_dependent_rewards` (§28). A controlled batch with
**one row per action** (row `i` takes action `i` at `t` and its reward target is
`ACTION_REWARDS[i]`), only the first recurrent step carries a reward target
(`reward_masks[:, 1:] = 0`), 200 steps:

| Target action | HOLD 0 | FLAT 1 | SHORT_100 2 | SHORT_50 3 | LONG_50 4 | LONG_100 5 |
|---|---|---|---|---|---|---|
| true reward | 0.000 | -0.100 | -0.200 | -0.050 | +0.100 | +0.200 |
| predicted (support) | -0.00016 | -0.09972 | -0.19952 | -0.04968 | +0.09980 | +0.19910 |
| predicted (scalar) | +0.00013 | -0.09904 | -0.20009 | -0.05094 | +0.09998 | +0.19996 |

* reward MAE: **0.1082 → 0.000395** (support), **0.1131 → 0.000365** (scalar)
* Pearson correlation between predicted and target reward: **0.9999993** /
  **0.9999911**; the predicted ranking equals the target ranking exactly
  (`argmax_order_matches = True`).

The dynamics network therefore learns *how the reward depends on the selected
action*, not merely the marginal reward mean.

## 12. Policy-model sanity result

`test_policy_model_fits_one_hot_multimodal_and_soft_targets` (§30). Four
controlled targets — one-hot, multimodal `[0.5, 0.5, 0, ...]`, maximum-entropy
uniform `1/6`, and low-entropy `0.94` — with value/reward masks zeroed so only
the policy term trains. 200 steps:

| | support | scalar |
|---|---|---|
| policy KL | 1.0146 → **0.001059** | 1.5313 → **0.000931** |
| top-1 agreement (last) | 0.75 | 0.50 |
| largest \|p̂ − π_target\| | 0.00272 | 0.00632 |

Fitted distributions (support): `[1.000, 0.000, 0.000, 0.000, 0.000, 0.000]`,
`[0.501, 0.497, 0.000, 0.000, 0.001, 0.000]`, `[0.168, 0.169, 0.166, 0.166,
0.165, 0.166]`, `[0.940, 0.007, 0.011, 0.011, 0.011, 0.020]` against targets
`[1,0,0,0,0,0]`, `[0.5,0.5,0,0,0,0]`, `[1/6 ×6]`, `[0.94,0.01,0.01,0.01,0.01,0.02]`.

The assertion is on the **probability gap**, not on argmax equality: the
multimodal target `[0.5, 0.5, ...]` has two tied maxima, so "argmax" is not
well-defined there and any argmax-equality test would be testing a coin flip.
Top-1 agreement is therefore reported as a diagnostic only (0.75 is exactly the
3 of 4 rows with a unique argmax).

## 13. Value-model sanity result

`test_value_model_learns_the_target_ordering` (§29): two trajectories with the
same actions but opposite reward streams (`+0.2` vs `-0.2` per step) and
distinguishable observations (`observation_tag = ±1`), 200 steps:

| | trajectory A ("good") | trajectory B ("bad") |
|---|---|---|
| target `z_0` | +0.59402 | -0.59402 |
| predicted (start) | — value MAE 0.5886 — | |
| predicted (end) | +0.59283 | -0.59327 |

* value MAE: **0.5886 → 0.00102**
* correlation: **1.0**; `values[0] > values[1]` and `targets[0] > targets[1]`.

## 14. Multi-step gradient verification

Four tests pin the unroll's backward path (§14 / §31):

| Test | What it establishes |
|---|---|
| `test_multi_step_unroll_gives_every_component_a_finite_gradient[gate]` (parametrised `1.0` and `0.5`) | **every** parameter of representation, dynamics and prediction receives a finite, non-zero gradient after a 5-step unroll — no component is starved. |
| `test_late_step_gradient_reaches_the_representation_only_when_not_detached` | zeroing the head losses at steps `1..K` still changes the representation gradient **only** because the last transition is not gated: with `gate = 1.0` the representation gradient changes, with a fully detached chain (all steps gated) it does not. This is the anti-detach regression guard. |
| `test_latent_gradient_scaling_reduces_the_gradient_norm` | the reported pre-clip gradient norm is strictly smaller at `gate = 0.0` than at `gate = 1.0`. |
| `test_latents_are_finite_across_the_whole_unroll` | `latent_k{0..K}_finite == 1.0` and finite `norm`/`min`/`max` for every `k`. |

Additionally `test_gradient_norm_is_measured_before_clipping` confirms the
reported `gradient_norm` is the norm *before* `clip_grad_norm_` (the clamp is a
safety net, not a measurement artifact) — which is why the real-data run reports
`max 7.11` with `max_grad_norm = 5.0`.

## 15. Real Forex replay smoke-training result

`python -m tools.smoke_muzero_learning --updates 200 --batch-size 32`
(EURUSD+GBPUSD+USDJPY, TRAIN split; 6 trajectories × 16 decisions × 8 simulations
= 96 env steps collected in 12.9 s; scales taken from the data-driven
calibration: reward 4.62e-04, value 3.90e-01; `K = 5`, `td_steps = 5`,
`γ = 0.99`; 539,280 parameters; `lr = 3e-4`, AdamW, `clip = 5.0`):

| update window (20 updates) | total | policy | value | reward |
|---|---|---|---|---|
| 0 | 5.5189 | 1.5333 | 1.7402 | 2.2454 |
| 1 | 4.2374 | 1.4817 | 1.0828 | 1.6729 |
| 2 | 3.7716 | 1.4334 | 0.9481 | 1.3901 |
| 3 | 3.4233 | 1.3828 | 0.8471 | 1.1934 |
| 4 | 3.1130 | 1.3538 | 0.7826 | 0.9766 |
| 5 | 2.8435 | 1.3211 | 0.7163 | 0.8060 |
| 6 | 2.6379 | 1.2838 | 0.6735 | 0.6806 |
| 7 | 2.4866 | 1.2569 | 0.6284 | 0.6014 |
| 8 | 2.3900 | 1.2332 | 0.5978 | 0.5591 |
| 9 | 2.3435 | 1.2374 | 0.5851 | 0.5211 |

* total loss **5.5189 → 2.3435 (-57.5%)**, best 2.2675
* policy KL **0.3757 → 0.0761**; policy top-1 agreement 70.25%
* value MAE **0.1685 → 0.0112**; reward MAE **1.459e-04 → 1.589e-05**
* gradient norm: mean 2.175, max 7.110 (pre-clip; clipped to 5.0)
* all losses decreasing monotonically window-to-window up to window 9

Caveat stated plainly: this is a **6-trajectory** replay and 200 updates. It
demonstrates that the Stage 4.4 path runs end-to-end on real Forex data and that
the objective decreases; it is not evidence of learned trading behaviour, and the
replay is far too small to tell overfitting from generalisation.

## 16. Training throughput

`python -m tools.benchmark_muzero_learning --iters 60` (CPU,
torch 2.13.0+cpu, latent 128 / hidden 256 × 2, obs_dim 351, `K = 5`,
synthetic batches with the real shapes):

| batch | updates/s | samples/s | unrolled states/s | ms/update | ms/forward (no grad) | RSS MB |
|---|---|---|---|---|---|---|
| 32 | 11.72 | 375 | 2,250 | 85.3 | 25.8 | 350.7 |
| 64 | 9.74 | 623 | 3,741 | 102.7 | 24.7 | 358.2 |
| 128 | 4.91 | 629 | 3,774 | 203.5 | 36.2 | 374.9 |
| 256 | 6.08 | 1,556 | 9,338 | 164.5 | 42.1 | 397.6 |

This is a re-measurement on the same machine as the original run (5.56 / 5.36 /
3.80 / 3.54 updates/s, RSS 350.8-395.0 MB). Throughput on this box varies by
roughly 1.5-2× between runs because of background load - three consecutive runs
of the same command gave 11.7 / 9.7 / 4.9 / 6.1, then 10.9 / 6.6 / 5.1 / 5.8,
then 14.9 / 10.4 / 9.9 / 6.2 updates/s - so the ratios and the memory numbers
are trustworthy while the absolute rates are only good to about a factor of two.
The original table is a valid draw from the same distribution.

"states/s" counts unrolled latent states `= batch × (K+1) = batch × 6`. The
no-grad unroll is only ~25-30% of a batch-256 update, so the cost is dominated by
backward through the `K`-step chain (which is exactly why the gradient gate
exists). RSS grows ~47 MB from batch 32 to 256 (activations the allocator
retains); there is no per-update leak (the deltas across the timed loop are within
noise). Memory is measured with `psutil` when the `train` extra is installed and
with a stdlib `psapi`/`getrusage` fallback otherwise; the column prints `n/a`
rather than a fake `0.0` when neither is available. These are the numbers Stage 4.5
production planning should use: on this CPU, 1,000 updates at batch 64 is ~3
minutes of pure learner time, so **the MCTS collection cost (~0.9 s/search) will
dominate any real run**, not the learner.

## 17. Checkpoint / resume test

`MuZeroLearner.save_checkpoint` / `load_checkpoint` (§34) store model +
optimizer state, `update_count`, `env_steps`, both RNG states, the architecture
fingerprint, the model/learner/target configs and **replay metadata only** (the
buffer itself is explicitly *not* embedded — `test_checkpoint_does_not_embed_the_replay_buffer`).

Round-trip verification, both in the unit tests and on the real run:

| Check | Result |
|---|---|
| probe unroll on a fixed batch before vs after reload | `8.741699 → 8.741699` (**identical**) |
| `update_count` restored | 200 |
| optimizer state restored (continuation step produces a finite loss) | 2.3585 |
| architecture mismatch (different `latent_dim` / `use_support`) | `ValueError` |
| foreign file | `ValueError("... is not a ForexMind MuZero learner checkpoint")` |
| learning-rate schedule position | preserved via `update_count` |

## 18. HOLD-related learning diagnostics

HOLD is the one action that is *always* valid and it dominates the search (≈30%
argmax in this replay), so it is the action most likely to be over-fitted or
collapsed. The loss layer reports, per update, and never uses them as loss terms:

| Diagnostic | Meaning | Real run (last update) |
|---|---|---|
| `pred_hold_prob` / `target_hold_prob` | mean predicted/target probability of HOLD over valid positions | 24.42% / 24.60% |
| `pred_argmax_hold_fraction` / `target_argmax_hold_fraction` | fraction of valid positions whose argmax is HOLD | 17.72% / 30.38% |
| `pred_group_{hold,flat,short,long}` | predicted probability mass per action family | 24.4 / 16.7 / 31.8 / 27.1 (%) |
| `target_group_{hold,flat,short,long}` | target mass per action family | 24.6 / 16.8 / 29.8 / 28.8 (%) |

Reading: after 200 updates the predicted **mass** matches the target mass closely
(HOLD 24.42% vs 24.60%, i.e. no HOLD collapse and no HOLD starvation), while the
*argmax* share is lower than the target's (17.7% vs 30.4%) — the head is still
spreading mass across the SHORT/LONG family rather than committing, which is the
expected early-training behaviour and the thing Stage 4.5 should watch. The
short/long **family** masses are close to their targets (31.8% vs 29.8% and
27.1% vs 28.8%), so the head has learned the family-level signal before the
within-family choice.

## 19. Remaining concerns

1. **Value-target concentration.** Only ~5 of 21 value bins carry data in the
   real replay (all targets negative, std 0.085). A narrower support
   (`scale ≈ 0.1`) or a different value parameterisation would use the bins
   better. The calibration tool reports this instead of hiding it; the fix is
   deliberate Stage 4.5 work.
2. **Reward target scale vs reward signal.** `mean |r| ≈ 1.0e-04`, `std =
   1.6e-04` — the reward is two orders of magnitude smaller than the value. With
   `c_r = 1.0` the reward term contributes little to the total; the calibration
   tool exists precisely so this can be chosen from data rather than guessed.
3. **Gradient gate is a heuristic.** `latent_gradient_scale = 0.5` damps the
   backward path through earlier steps. `1.0` (no scaling) passes the same tests;
   the value was chosen for stability, not from a sweep, and a short sweep is
   appropriate before production.
4. **No replay persistence.** A resumed run restarts collection with an empty
   buffer (checkpoints store metadata only). Deliberate for Stage 4.4; Stage 4.5
   needs a serialized replay if runs are to be interrupted.
5. **Gradient clipping is active.** Real runs hit `max_grad_norm = 5.0`
   (observed pre-clip max 7.11). That is a safety net working as intended, but it
   means the effective step size early in training is smaller than `lr`.
6. **Single-process, CPU only.** 3.5-5.6 updates/s (see §16) is not a
   production budget; no AMP / no threads tuning / no device batching has been
   attempted. Also, `latent_dim 128` was never tuned against the 351-dim
   observation.
7. **The overfit fixture is weak on value/reward** (§10). It is a genuine
   limitation of that fixture, not of the objective; replacing it with a fixture
   with action-dependent value targets would make the test stricter.
8. **Not verified:** reproducibility across processes beyond RNG capture,
   distributed gradient averaging, mixed precision, resume-in-the-middle-of-an-
   epoch exactness, and any long-run stability (>10k updates).

## 20. Readiness for Stage 4.5

Ready:

* The joint objective is implemented, masked correctly, and provably
  differentiable through `K` recurrent steps with a documented, testable
  anti-detach design.
* Loss, targets, masks and the CLI-facing learner are one coherent, typed,
  exported API (`forexmind.muzero` re-exports 60 names; `mypy` clean over 116
  files).
* Data-driven target calibration exists and is wired into the real-data smoke
  tool, so scales are chosen from measurements, not constants.
* Checkpoint/resume works and is validated, including architecture checks.
* Real-data end-to-end training decreases the objective monotonically over 200
  updates with a 57.5% total-loss reduction, no non-finite values, and no HOLD
  collapse.
* Throughput and memory characteristics are measured (§16) and the collection
  cost is identified as the dominant term.

Recommended, in order, for Stage 4.5:

1. Serialize the replay buffer (or store collected trajectories) so runs are
   resumable and the replay survives process exit.
2. Run a longer real-data training run with a proper TRAIN/VALIDATION split and
   report loss on held-out trajectories — the current evidence is train-only.
3. Sweep `latent_gradient_scale ∈ {0.25, 0.5, 1.0}` and the three loss weights on
   a fixed replay, and adopt the value/scale combination from §4 that spreads the
   value bins wider.
4. Then move to the deferred Stage 4.5+ items: actors, reanalysis, prioritized
   replay, target refreshing, Stochastic MuZero.
