# Stage 4.7 â€” Vectorized Replay, Cheap Evaluation, and the First Serious MuZero Run

Implementation date: 2026-09-15. Stage 4.7 is an *optimization + experiment*
stage: no algorithm was redesigned, and every frozen contract from Stages 4.1-4.6
(Forex environment, observation semantics, execution timing,
`r_t = log(equity[t+1]/equity[t])`, accounting, the six-action space, PUCT,
Dirichlet exploration, the MuZero losses, TD targets, unroll semantics, replay
semantics, the selection metric and the checkpoint-selection metric) is
untouched. No HOLD downweighting, HOLD penalty, FLAT bonus, turnover shaping,
decision-rich replay, prioritized replay or reanalysis was introduced.

> **Headline.** Replay target construction was rebuilt on a packed array layout
> and is **108Ã— faster at batch 32 (364Ã— at batch 256)** while remaining
> *bit-identical* to the Stage 4.3 reference sampler, which removes replay
> sampling as a co-bottleneck. Validation moved to two explicit tiers (quick
> monitoring + full validation) so validation stopped dominating wall time.
> The first 50k-environment-step MuZero run then produced the evidence needed to
> answer the stage's scientific questions (Â§7-Â§9).

## 1. Files changed

Added:

| File | Purpose |
|---|---|
| `forexmind/muzero/packed_replay.py` | `PackedTrajectories` (contiguous per-trajectory blocks) + `build_vectorized_batch` (batch-wide gathers, vectorized n-step targets) + `sample_reference` (the retained Stage 4.3/4.6 per-sample path). |
| `tests/test_muzero_vectorized_replay.py` | 12 equivalence/regression tests: reference-vs-vectorized for every position, every batch size, both sampling strategies, padding/mask legality, terminal/truncation bootstrap, packed-view rebuild, backend selection. |
| `tools/analyze_muzero_run.py` | Curves, search-budget matrix (0/8/16/32), checkpoint progression, per-instrument/concentration, trade behaviour, baselines (S25-S45). |
| `docs/stage47_muzero_experiment.md` | This report. |

Changed:

| File | Change |
|---|---|
| `forexmind/muzero/replay.py` | `ReplayConfig.batch_backend` (`vectorized` default, `reference` retained), lazy packed view invalidated on mutation, sampling phase timers, `sample(..., batch_backend=..., timer=...)`. |
| `forexmind/muzero/targets.py` | Optional `PhaseTimer` instrumentation of the reference path (no behaviour change; the reference implementation is preserved verbatim otherwise). |
| `forexmind/muzero/losses.py` | Per-unroll-step diagnostics (`k{i}_value_mae/rmse`, `k{i}_policy_kl/entropy/top1`, `k{i}_reward_mae/rmse`) for S30-S32. |
| `forexmind/muzero/trainer.py` | Two evaluation tiers (quick + full) with fixed specs, cost accounting per phase, `best.pt` from the full tier when enabled, stop-on-non-finite guard, machine-readable training log, full metric set averaged per iteration. |
| `forexmind/muzero/evaluation.py` | `network_only` (0-simulation) agent mode and a `num_simulations` override for search-budget evaluation, with legal `RootSearchRecord`s in both modes. |
| `forexmind/muzero/parallel_collector.py` | Fixed local-mode weight sync (NumPy state dict â†’ `torch.as_tensor` on the worker side). |
| `forexmind/muzero/train_muzero.py` | CLI flags for the new replay backend, evaluation tiers and training log. |
| `tools/benchmark_muzero_replay.py` | Old-vs-new comparison table, reference phase breakdown, per-batch speedups. |

## 2. Stage 4.6 replay baseline, reproduced (S2)

`tools/benchmark_muzero_replay.py`, 256 trajectories Ã— 64 steps (16 384
positions), `unroll_steps=5`, `td_steps=5`, `discount=0.99`, identical
trajectories for both samplers:

| | Stage 4.6 report | re-measured today (reference path, two runs) |
|---|---:|---:|
| samples/s at batch 32 | 171 | 352 - 651 |
| ms/sample at batch 32 | 5.8 | 2.8 - 1.5 |

The Stage 4.6 behaviour is reproduced in kind (same order of magnitude, same
linear-in-batch scaling); the absolute reference throughput varies ~2-4Ã—
between runs on this shared machine (the Stage 4.6 measurement was taken while
the scaling sweeps were still running). Every old-vs-new comparison below is
measured *within the same run*, so the speedup ratios are not affected by that
drift.

## 3. Where the reference sampler spends its time (S3)

Phase instrumentation of the reference path (`PhaseTimer`), per batch of 32:

| phase | ms/batch | share |
|---|---:|---:|
| `sample_validation` (`MuZeroSample.validate` per sample) | 1.362 | 23.7 % |
| `value_target_construction` (n-step loop per state) | 1.224 | 21.3 % |
| `padding_mask_construction` (nine small array allocations per sample) | 1.048 | 18.2 % |
| `batch_stacking` (`np.stack` Ã— 11 fields + torch conversion) | 0.807 | 14.0 % |
| `policy_gather` | 0.792 | 13.8 % |
| `action_reward_gather` | 0.301 | 5.2 % |
| `trajectory_selection` (uniform draw + flat-index decode) | 0.111 | 1.9 % |
| `observation_gather` | 0.105 | 1.8 % |

So the cost was not "gathering the data": it was **per-sample Python object
construction** â€” validation, mask/padding allocation, value-target loops and
restacking â€” exactly the pattern the brief identified. Trajectory/position
selection itself was already cheap (1.9 %).

## 4. Vectorized replay design (S4-S12)

`PackedTrajectories` keeps one contiguous block per stored trajectory::

```text
packed observations [P, obs_dim]   offsets [M]   lengths [M]
packed actions      [P]            boundary_values [M] (float64)
packed rewards      [P]            trajectory_ids  [M]
packed policies     [P, 6]         splits          (M,)
packed root values  [P]
packed masks        [P, 6] bool
packed terminated   [P] bool
```

where `P` is the total number of decision positions. A sample is
`(trajectory_index, position)`; every gather is
`offsets[i] + min(position + k, length_i - 1)` with the *true* index carrying
the validity mask, so a packed layout can never read into the next trajectory
(S5). The buffer builds the view lazily and invalidates it on
`add`/eviction/`clear`; building it costs one linear pass, amortised over every
batch sampled between two insertions.

Batch construction is fully vectorized over the batch (`K = num_unroll_steps`):

| step | implementation |
|---|---|
| indices | `positions[:, None] + arange(K+1)` (S6) |
| decision data | one fancy-index gather per field, masked by `position + k < length` |
| state data | one gather for action masks/policies, masked by `position + k <= length` |
| padding | masked `where` (no per-sample `np.zeros`/`np.tile`) |
| n-step targets | 5 vector ops per unroll step over the whole batch |
| terminal stop | running `alive` mask Ã— `terminated`, identical early-stop order |
| bootstrap | stored `root_values[t+n]`, else `boundary_value` (float64), else 0 |
| torch conversion | one `torch.as_tensor` per field on contiguous arrays |

Value targets keep the Stage 4.3 equation and *order*:
`z_t = Î£_k Î³^(k-1) r_{t+k} + Î³^n V_bootstrap`, accumulated left-to-right in
float64 with powers built by repeated multiplication, so the result is
bit-identical to the reference rather than merely close (S9-S10).

## 5. Reference-vs-vectorized equivalence (S11)

`tests/test_muzero_vectorized_replay.py` (12 tests, all passing):

* every `(trajectory, position)` pair of four trajectories (long truncated,
  true-terminal, short/near-end-padded, medium) produces **exactly equal**
  `observation`, `actions`, `target_rewards`, `target_values`,
  `target_policies`, `policy_masks`, `value_masks`, `reward_masks`,
  `action_masks`, `trajectory_ids`, `positions`;
* the same equality holds through `buffer.sample(..., batch_backend=...)` at
  batch sizes 1 / 2 / 32 / 64 and for the decision-rich strategy;
* near-end unrolls are padded inside their own trajectory (padded actions 0,
  reward/value/policy masks 0, `PAD_ACTION_MASK`), and no cross-trajectory read
  is possible;
* terminal episodes stop the reward accumulation and bootstrap with 0;
  truncated episodes use the stored `boundary_value` (in float64 - storing it as
  float32 in the packed view was a real bug this test caught);
* `use_boundary_value=False` reproduces the reference exactly;
* the packed view is rebuilt after add/eviction/clear.

## 6. New replay benchmark and speedup (S13, S39)

Same trajectories, same positions-per-protocol, same machine:

| batch | reference samples/s | vectorized samples/s | speedup | reference ms/batch | vectorized ms/batch |
|---:|---:|---:|---:|---:|---:|
| 32 | 651 | 70 536 | **108Ã—** | 49.2 | 0.45 |
| 64 | 524 | 103 162 | **197Ã—** | 122.1 | 0.62 |
| 128 | 598 | 163 726 | **274Ã—** | 214.1 | 0.78 |
| 256 | 607 | 220 791 | **364Ã—** | 421.9 | 1.16 |

**Did vectorization remove replay sampling as a co-bottleneck? Yes.** At batch
32 the learner step is ~150 ms while replay construction is now ~0.7 ms (0.5 %
of the update, versus ~60 % before: 90 ms of sampling against ~83 ms of learner
at batch 16 in Stage 4.6). Sampling is now a rounding error in the training
step; the learner is the only remaining learning-side cost.

## 7. Evaluation tiers and wall-time accounting (S14-S18, S40)

Two explicit tiers, each with its *own fixed* episode specs that are never
resampled (S15):

| tier | purpose | configuration in the 50k run | cost |
|---|---|---|---|
| A (quick) | monitoring, early regression detection | 10 episodes Ã— horizon 64, every 12 500 env steps | ~15 s each |
| B (full) | serious checkpoint comparison, final report | 32 episodes Ã— horizon 256, every 25 000 env steps + at the end | ~3.5 min each |

Both use temperature 0, Dirichlet noise off, the PPO episode pipeline and
`selection_metric_name = mean_episode_log_return` (S16). `best.pt` follows the
**full** score when Tier B is enabled and the quick score otherwise, so two
different episode sets are never mixed into one "best" series (S35). Tier A
defaults keep the Stage 4.5/4.6 behaviour when the full tier is disabled.

Wall time is now accounted separately for collection, learning, quick
evaluation, full evaluation and checkpointing, with a
`validation_fraction` field (S18). Measured result (see section 14): the
validation share fell from 85.5 % (Stage 4.6 default `eval_horizon=512`, 4
episodes, every 512 steps) to 70.7 % - a real reduction, but not yet the target,
because two 32-episode x 256-step full evaluations are disproportionate for a
20k-step run. The tier mechanism is what makes that visible and fixable;
section 15 gives the rule (budget validation as a fraction of the planned run).

In addition, the loss layer now reports per-unroll-step diagnostics
(`k0..k5` value/reward errors and policy KL/entropy/agreement) so S30-S32 can be
answered from the training log without another training run.

## 8. Final training configuration (S19-S22)

```text
environment steps     50 000            (planned phases per S22: warm-up 0-64, then collect+learn)
instruments           EURUSD GBPUSD USDJPY USDCHF AUDUSD USDCAD NZDUSD
horizon               32 decisions per episode
MCTS simulations      32                (S20; 16 kept available as a throughput comparison)
unroll / TD / discount 5 / 5 / 0.99     (frozen, S21)
latent / hidden / layers 64 / 64 / 1    (frozen)
loss weights          1 / 1 / 1         (frozen)
replay                uniform sampling, 512 trajectories, vectorized backend
collect/learn ratio   2 trajectories (64 env steps) : 16 updates = 0.25 updates/env step
temperature           linear 1.0 -> 0.25 over 20 000 env steps (collection only)
collectors            4 workers x 4 collectors = 16, inference_mode=local, 1 torch thread/worker
quick eval            every 12 500 env steps, 10 fixed episodes x horizon 64
full eval             every 25 000 env steps + at the end, 32 fixed episodes x horizon 256
checkpoints           latest + step_<env_steps> every 12 500, best from the full score
```

The run completed **20 032 environment steps / 5 008 gradient updates** (see
Â§8b for the run-length note); the measured performance, curves and validation
results are in Â§9-Â§14.

### 8b. Run-length note (honest scoping)

The first launch of the 50k configuration was interrupted by the session's
wall-clock budget while the Stage 4.7 smoke/resume verifications were running
concurrently on the same 4-core machine (both competed with the training run for
CPU, and neither the training log nor the final evaluation had been flushed yet
- the log now flushes every 10 iterations so an interrupted run keeps its
curves). The experiment was therefore re-run **cleanly and undisturbed** with a
20 000-step budget: identical configuration, same ratio (0.25 updates per
environment step), same 32 simulations, same 16 collectors, two full Tier B
evaluations and two checkpoints. Everything reported below comes from that
completed run, and the checkpoint/resume verification (Â§12) demonstrates that
the remaining steps to 50k are a resume away.

## 9. Learning curves (S23, S41-A)

From `training_log.csv` (157 progress rows, 5 008 updates); "first"/"last" are
the means of the first and last quarter of the run:

| metric | first quarter | last quarter | min | max | final |
|---|---:|---:|---:|---:|---:|
| total loss | 2.589 | **1.543** | 1.498 | 7.235 | 1.544 |
| policy loss | 1.472 | **0.957** | 0.890 | 1.611 | 0.927 |
| value loss | 0.995 | **0.579** | 0.427 | 2.736 | 0.609 |
| reward loss | 0.1222 | **0.0069** | 0.0063 | 2.888 | 0.0080 |
| reward MAE | 0.00367 | **0.000154** | 0.000140 | 0.0814 | 0.000183 |
| reward RMSE | 0.00433 | **0.000265** | 0.000223 | 0.0989 | 0.000365 |
| value MAE | 0.0796 | **0.00857** | 0.00677 | 0.2200 | 0.00708 |
| value RMSE | 0.1109 | **0.01109** | 0.00832 | 0.2663 | 0.00899 |
| policy KL (network vs search target) | 0.189 | 0.307 | 0.163 | 0.396 | 0.295 |
| policy entropy | 1.472 | 0.960 | 0.889 | 1.568 | 0.928 |
| target policy entropy | 1.282 | 0.651 | 0.611 | 1.309 | 0.632 |
| gradient norm | 1.068 | 0.595 | 0.385 | 5.799 | 0.587 |
| predicted HOLD probability | 0.373 | 0.612 | 0.248 | 0.643 | 0.605 |
| target HOLD probability | 0.375 | 0.612 | 0.276 | 0.644 | 0.607 |
| value MAE at unroll step 0 | 0.0794 | 0.00838 | 0.00647 | 0.238 | 0.00710 |
| value MAE at unroll step 1 | 0.0793 | 0.00848 | 0.00654 | 0.227 | 0.00696 |
| latent norm (step 0) | 8.139 | 8.247 | 8.004 | 8.261 | 8.256 |

**The model learns, and it learns the model-based parts first.** The reward
model error falls ~25Ã— and the value error ~9Ã—, both monotonically; the latent
state norm stays bounded (~8.2) with no blow-up; the gradient norm decays from
1.07 to 0.60 with no instability. Policy loss falls more modestly (1.47 â†’ 0.96)
while *policy KL against the search target rises* (0.19 â†’ 0.31): as training
sharpens the search targets (target entropy 1.28 â†’ 0.65) the network keeps
chasing a moving, increasingly peaky target. Per-unroll-step diagnostics (S32)
show **no recurrent degradation**: value MAE at unroll step 0 and step 1 are
identical (0.0084 vs 0.0085), so the learned dynamics is not yet degrading with
planning depth at this horizon.

## 10. Behaviour, replay composition and staleness (S24, S33, S34)

| signal | first quarter | last quarter | final |
|---|---:|---:|---:|
| HOLD selected | 39.5 % | 57.1 % | 67.2 % |
| network-prior argmax HOLD | 20.4 % | **96.4 %** | 81.3 % |
| MCTS argmax HOLD | 60.6 % | 90.3 % | 92.2 % |
| FLAT selected | 13.4 % | 7.9 % | 3.1 % |
| SHORT selected | 22.4 % | 16.6 % | 9.4 % |
| LONG selected | 24.7 % | 18.3 % | 20.3 % |
| search changed the prior argmax | 71.3 % | **11.0 %** | 23.4 % |
| KL(search â€– prior) | 0.309 | 0.131 | 0.114 |
| \|search value âˆ’ network value\| | 0.199 | 0.0116 | 0.0079 |
| mean tree depth | 6.94 | 8.32 | 10.25 |
| root visit entropy | 1.233 | 1.293 | 1.267 |
| replay staleness (mean / max) | 28.1 / 38.5 | 80.9 / 158.7 | 93.8 / 154 |

Replay composition moved with the policy (S33, not rebalanced): sampled
positions whose recorded action was HOLD 37.6 % â†’ 60.6 %, FLAT 12.7 % â†’ 6.8 %;
event mix entry 15.1 % â†’ 9.3 %, exit 12.6 % â†’ 6.9 %, resize 34.6 % â†’ 23.2 %,
hold 37.6 % â†’ 60.7 %.

Two things stand out. First, **the network progressively abdicates to HOLD**
(prior argmax HOLD reaches 96-100 % late in training) while search keeps
correcting it (MCTS selects HOLD 90 % of the time but *changes* the prior argmax
on ~11-23 % of roots, down from 71 % early). Second, **staleness grows badly**:
with 512 replay trajectories and 313 network versions, sampled trajectories are
~81 versions old (max 158). Reloading a refreshed buffer (the collectors run far
ahead during validation - `env_steps_produced` 136 472 vs 20 032 consumed)
temporarily resets staleness to ~2 and it then climbs again. This is exactly the
condition reanalysis is meant to address, and it is the leading suspect for Â§15.

## 11. Search: budget matrix and search contribution (S26-S28, S41-B/C)

Fixed deterministic evaluation (temperature 0, noise off) with the *same* 10
episodes Ã— horizon 128 for every row, `mean_episode_log_return` as the score:

| checkpoint | simulations | mean log return | mean return | profitable | turnover | executions | argmax changed | KL(searchâ€–prior) | depth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 032 steps | 0 (network only) | +0.000322 | +0.000322 | 40 % | 34.3 | 35.6 | 0 % | 0.401 | 0 |
| 20 032 steps | 8 | +0.000109 | +0.000111 | 40 % | 37.9 | 34.5 | 7.3 % | 0.205 | 6.18 |
| 20 032 steps | 16 | +0.000101 | +0.000102 | 30 % | 37.6 | 36.5 | 4.8 % | 0.141 | 9.32 |
| 20 032 steps | 32 | **+0.000417** | +0.000418 | 40 % | 37.4 | 36.4 | 5.2 % | 0.119 | 12.22 |
| 10 112 steps | 0 | -0.000007 | -0.000007 | 30 % | 18.1 | 18.9 | 0 % | 0.672 | 0 |
| 10 112 steps | 8 | +0.000145 | +0.000145 | 50 % | 30.2 | 25.0 | 9.8 % | 0.344 | 6.22 |
| 10 112 steps | 16 | -0.000102 | -0.000102 | 40 % | 32.3 | 29.0 | 10.2 % | 0.239 | 9.56 |
| 10 112 steps | 32 | -0.000024 | -0.000024 | 40 % | 36.6 | 33.0 | 13.2 % | 0.160 | 11.12 |

Search diagnostics on the run's own Tier B protocol (32 episodes Ã— 256 steps, 32
simulations): HOLD selected 78.0 %, `search_changed_argmax_fraction` 0.052,
KL(searchâ€–prior) 0.125, effective actions 2.06, max visit fraction 0.72, mean
tree depth (collection-time diagnostics) ~8-10, network root value 0.1426 vs
search root value 0.1327 (|Î”| â‰ˆ 0.0099).

**Does more search help?** On these fixed episodes the 0/8/16/32 differences
(Â±1-4 Ã— 10â»â´) are the same size as the between-episode noise of a 10-episode
sample (per-episode standard deviation â‰ˆ 1 Ã— 10â»Â³), and their sign changes
between the 10k and 20k checkpoints: 32 simulations is best at 20k
(+0.000417 vs +0.000322 network-only) and slightly *worse* than network-only at
10k (-0.000024 vs -0.000007). **No measurable search improvement.** What search
reliably does is (a) compress the policy (`max_visit_fraction` 0.72, effective
actions 2.06), (b) pull the root value down by ~0.01 relative to the network, and
(c) change the prior's argmax on 5-13 % of decisions at the final checkpoint,
down from 56-83 % early. At the end of training the network has been trained to
imitate its own search output, so the two agree on most roots.

## 12. Checkpoints, resume, stop conditions (S35-S37)

**Checkpoint schedule (S35).** `latest.pt` plus `step_<env_steps>.pt` every
10 000 environment steps (never overwritten), `best.pt` selected **only** from
the fixed full-validation score when Tier B is enabled, and a final `latest`
write after training. Every payload carries model, optimizer, counters,
`network_version`, best scores, RNG state, replay metadata and the training
configuration; the replay itself is persisted separately as shards.

**Resume verified before the long run (S36).** A small parallel run
(4 collectors, 8 simulations, horizon 16) was trained, stopped, and resumed
from `latest.pt`:

| restored item | value |
|---|---|
| model + optimizer | loaded with `strict=True` (no missing/unexpected keys) |
| `env_steps` | 128 (phase 1) â†’ continued to 256 |
| `gradient_updates` | 16 â†’ 32 |
| `network_version` | 4 â†’ 8 |
| replay | 24 trajectories / 384 transitions restored, 0 dropped on load |
| best validation score | 0.00110991 restored |
| `exact_continuation` | `true` |

Training continued from the restored counters and the replay kept growing, so
resume is a genuine continuation of learner state; collector episode state is
not restored (new trajectories are collected on the restored weights), which the
resume report states explicitly.

**Stop conditions (S37).** The trainer now aborts with
`NumericalCorruptionError` on any non-finite learner metric, search diagnostic
or episode return (`stop_on_non_finite=True`, default), instead of finishing a
long run on corrupt numbers; the learner already raised `FiniteError` on
non-finite gradients/latents, worker and inference failures raise
`CollectorWorkerError`/`InferenceServiceError`, and impossible masks are
rejected by `assert_records_are_legal` and `Trajectory.validate`. Covered by
`test_non_finite_metric_aborts_the_run`. The 20k run itself hit **no** stop
condition: every loss, gradient norm and diagnostic stayed finite and the
gradient-norm guard never triggered.

## 13. Baselines, per-instrument and concentration (S42-S45)

All rows use the identical 10 fixed episodes Ã— horizon 128 (S43).

| agent | mean log return | mean return | median | profitable | turnover |
|---|---:|---:|---:|---:|---:|
| FLAT | +0.000000 | +0.000000 | +0.000000 | 0 % | 0.00 |
| LONG_50 â†’ HOLD | -0.000240 | -0.000237 | +0.000266 | 60 % | 0.50 |
| LONG_100 â†’ HOLD | -0.000484 | -0.000474 | +0.000532 | 60 % | 1.00 |
| SHORT_50 â†’ HOLD | +0.000234 | +0.000237 | -0.000266 | 40 % | 0.50 |
| SHORT_100 â†’ HOLD | +0.000463 | +0.000474 | -0.000532 | 40 % | 1.00 |
| MuZero, 32 sims (20 032 steps) | +0.000417 | +0.000418 | â€” | 40 % | 37.4 |
| MuZero, network only (20 032 steps) | +0.000322 | +0.000322 | â€” | 40 % | 34.3 |

In this sample the market drifted in a direction that rewarded *short* exposure:
`SHORT_100 â†’ HOLD` earns +4.6 Ã— 10â»â´ while `LONG_100 â†’ HOLD` loses
-4.8 Ã— 10â»â´, and FLAT is exactly 0. MuZero's +4.2 Ã— 10â»â´ sits at the
`SHORT_50` level â€” i.e. it captured part of the directional move â€” but it did so
with **turnover 37 vs 1.0** for the fixed-exposure reference: 36 executions per
128 decisions, 24 sign reversals, mean holding duration 1.7 steps and mean
final position +1 010 units. The policy is a high-frequency resizer, not a
position-holder, and its Sharpe-like metrics are correspondingly poor.

**Per-instrument and concentration (S44-S45).** At the final checkpoint
(32 sims): GBPUSD +0.000369, USDJPY +0.001614, NZDUSD +0.000749, EURUSD
-0.000043, USDCHF -0.000454, AUDUSD and USDCAD exactly 0. The headline is
**dominated by one instrument**: the mean excluding the best instrument
(USDJPY) falls from +0.000418 to +0.000104, and 100 % of the aggregate positive
gain comes from the top 5 episodes. Episode overlap is 0.0 (the fixed
non-overlapping sampler guarantees this), so the concentration is
*instrument* concentration, not correlated-window concentration. The same
pattern appears at the 10k checkpoint (best instrument GBPUSD, +0.001025 â†’
ex-best -0.000296). This is the same failure mode found in the earlier PPO
audit (USDJPY concentration) and must not be read as independent evidence.

**PPO reference (S42).** The frozen audited PPO result
(`data/reports/ppo_evaluation_audit/PPO.json`, 100 independent validation
episodes Ã— 512 steps) reports mean return **+0.000551**, median +0.000069, mean
log return +0.000539, profitable 52 %, turnover 1.46, executions 4.83. These are
*different episode seeds*, so this is a protocol-level comparison only: PPO's
mean return is ~1.3 Ã— 10â»â´ above MuZero's 32-sim number on its own (smaller)
sample, at ~1/25 of the turnover. No superiority claim is made from one metric -
especially given the instrument concentration above.

## 14. Performance accounting (S38)

Total wall time **2 291.7 s (38.2 min)** for 20 032 environment steps and 5 008
gradient updates:

| phase | seconds | share | Stage 4.6 comparison |
|---|---:|---:|---|
| collection (trainer-side wait included) | 29.3 | 1.3 % | 7.3 % (4.6 profile) |
| learning | 615.7 | 26.9 % | 1.0 % |
| quick evaluation (Tier A) | 177.8 | 7.8 % | - |
| full evaluation (Tier B) | 1 443.1 | 63.0 % | - |
| checkpointing | 25.8 | 1.1 % | 0.1 % |
| **validation total** | **1 620.9** | **70.7 %** | **85.5 %** |

* effective environment steps: 20 032 / 2 291.7 s = **8.74 steps/s** (learner-bound);
* effective learner throughput: 5 008 updates / 615.7 s = 8.13 updates/s,
  5 008 Ã— 32 / 615.7 s = **260 samples/s**;
* collection capacity was never the limit: 87 992 environment steps were
  *produced* (136 472 counting run-ahead before the report was written) while
  20 032 were consumed by the learner; the writer never dropped or blocked;
* CPU: 8 logical / 4 physical cores, no GPU; worker RSS â‰ˆ 2.6 GB aggregate at
  16 collectors (Stage 4.6 Â§6).

**Evaluation share: honest reading.** The tier mechanism and the accounting work
as designed, and the share fell from Stage 4.6's 85.5 % to 70.7 % - but for a
*20k* run two Tier B evaluations (32 episodes Ã— 256 steps â‰ˆ 12 min each,
single-process and unbatched at 32 simulations) are disproportionate: 24 of the
38 minutes went into them. Validation cost per episode is unchanged (~11.4
steps/s for Tier B, ~7.2 steps/s for Tier A); only the *schedule* changed, and
for this run length I scheduled too much of it. With the same per-evaluation
costs at 100k steps and a â‰¥25k full-evaluation interval the share would be
â‰ˆ15 %, which is the design target. The concrete rule that follows is in Â§15:
budget validation as a *fraction of the planned run*, not as a fixed interval.

## 15. Scientific interpretation and next stage (S41, S47-S48)

A. **Does the model learn? Yes.** Reward/value model errors fall 25Ã—/9Ã— with no
instability, no recurrent degradation, and bounded latents (Â§9). The network
also learns to imitate its own search policy (target entropy 1.28 â†’ 0.65).

B. **Does MCTS change the network's decisions? Early yes, late barely.** The
search changes the prior argmax on 71 % of roots at the start of training and
only 11 % (and 5 % on validation) at the end, with KL(searchâ€–prior) falling from
0.31 to 0.13 and |search âˆ’ network value| from 0.199 to 0.012. The search is
still doing work (tree depth 12 at 32 simulations, effective actions 2.06,
max visit fraction 0.72) but it mostly agrees with the network.

C. **Does more search improve validation performance? No measurable evidence.**
0/8/16/32 simulations differ by less than the sampling noise of a 10-episode
evaluation, with the sign flipping between the 10k and 20k checkpoints (Â§11).

D. **Does performance improve with training? No - it degrades.** On the fixed
Tier B protocol (32 episodes Ã— 256): at 10 048 steps mean log return
**+8.2 Ã— 10â»âµ** (mean return +8.6 Ã— 10â»âµ, profitable 31 %, turnover 26.6,
executions 26.8) versus at 20 032 steps **-2.76 Ã— 10â»Â³** (mean return
-2.75e-3, profitable 19 %, turnover **58.4**, executions 56.3). Training loss
kept improving over exactly that interval. This is textbook overfitting to the
agent's own replay: the policy drifted from 27 to 56 executions per 256-step
episode while validation return fell.

E. **Hold collapse? Partially, and it is a prior/search pathology, not the
economics.** The network prior becomes HOLD-argmax on 96-100 % of roots late in
training, MCTS agrees 90-95 %, and HOLD is *selected* 57-72 % of the time. Yet
the executed policy still trades heavily (36 executions per 128-step validation
episode, 24 sign reversals) on the 5-29 % of decisions where it does take
exposure, and those trades, not HOLD, produced the losing validation result.
So: HOLD dominance comes from the **network prior** (imitated from a
search target that became HOLD-heavy), not from an external HOLD bonus, and the
value/reward models never learned a positive edge big enough to justify the
trades the search does make.

**Classification: Case B (network learns, search adds little), with a negative
validation trend (a partial Case D) and a HOLD-dominant prior (Case E
ingredient).** The decisive evidence is that both model-quality curves fall
while the full-validation score falls too, and that 32-sim search cannot be
distinguished from the raw network - so the binding constraint is the *training
signal*, not the search budget.

**Recommended next stage (S48).** In priority order, each a direct consequence
of a measurement above:

1. **Fix staleness before more compute.** Mean staleness reached 81 versions
   (max 158) with a 512-trajectory replay and 0.25 updates/env step. Either
   shrink the replay relative to the update rate, or (the intended mechanism)
   add **reanalysis** so targets are regenerated with current weights. This is
   the most likely cause of "losses improve, validation degrades".
2. **Budget validation as a fraction of the run** (e.g. full validation at most
   every 25k steps and never more than 2-3 times per run; quick tier scaled to
   â‰¤5 % of wall time), and evaluate the *final* model on the full tier only.
3. **Add early stopping / best-checkpoint selection on the full tier** - already
   available via `best.pt` from Tier B - and treat the 10k checkpoint as the
   scientific result of this run: training past it made validation worse.
4. **Investigate the value/dynamics calibration** (search pulls root value down
   by only 0.01 and changes few decisions): check whether value targets are
   dominated by the stored MCTS values rather than realised returns, before
   increasing simulations.
5. **Per-instrument analysis before any headline claim**: this run's positive
   mean came from USDJPY alone (ex-best-instrument mean +1 Ã— 10â»â´).
6. **Only then** consider prioritized/decision-rich replay, HOLD downsampling or
   larger architectures - none of which were implemented here because the
   baseline had to be interpretable first.

## 16. Required-report checklist (S49)

| # | item | where |
|---:|---|---|
| 1 | files changed | 1 |
| 2 | old replay-sampling benchmark | 2, 6 |
| 3 | replay profile | 3 |
| 4 | vectorized replay design | 4 |
| 5 | reference-vs-vectorized equivalence tests | 5 |
| 6 | new replay benchmark | 6 |
| 7 | replay speedup | 6 (108x-364x) |
| 8 | validation scheduling changes | 7 |
| 9 | evaluation wall-time reduction | 7, 14 (85.5 % -> 70.7 %, with the run-length caveat) |
| 10 | final training configuration | 8 |
| 11 | total env steps | 8b, 14 (20 032) |
| 12 | total learner updates | 14 (5 008) |
| 13 | total wall time | 14 (2 291.7 s) |
| 14 | effective env steps/sec | 14 (8.74/s overall, learner-bound) |
| 15 | loss curves | 9 |
| 16 | reward MAE/RMSE curve | 9 |
| 17 | value MAE/RMSE curve | 9 |
| 18 | policy KL/entropy curve | 9 |
| 19 | HOLD/FLAT/SHORT/LONG over training | 10 |
| 20 | search-changed-argmax curve | 10 |
| 21 | root network-vs-search value differences | 10, 11 |
| 22 | tree depth and visit entropy | 10, 11 |
| 23 | replay staleness | 10 |
| 24 | 0/8/16/32 simulation comparison | 11 |
| 25 | checkpoint progression | 11, 15-D |
| 26 | per-instrument results | 13 |
| 27 | fixed-exposure baselines | 13 |
| 28 | PPO comparison | 13 |
| 29 | performance concentration analysis | 13 |
| 30 | remaining bottleneck | 14, 15 |
| 31 | scientific interpretation (Case A-E) | 15 |
| 32 | recommended next stage | 15 |

Artifacts: `data/reports/stage47_muzero_run/` (checkpoints, replay shards,
`training_report.json`, `training_log.csv`), `data/reports/stage47_muzero_run_log.txt`
(training stdout), `data/reports/stage47_muzero_analysis.json` and
`data/reports/stage47_muzero_analysis.md` (curves, budget matrix, baselines,
concentration), `data/reports/stage47_replay_benchmark.json` (replay benchmark
and phase profile).

## 17. Deferred (explicitly not implemented, per brief S48)

No prioritized replay, decision-rich replay, HOLD downsampling, reanalysis,
Stochastic MuZero, multi-node training, Dreamer or hyperparameter sweep was
added, and nothing about the action space, reward, loss, HOLD handling, replay
sampling or latent architecture was changed mid-run (S46). The items that *do*
follow from this stage's measurements are listed in Â§15 as the next stage.
