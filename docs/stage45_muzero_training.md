# Stage 4.5 — the first complete MuZero training loop

Implementation date: 2026-09-13. This stage wires Stages 4.1-4.4 into one
synchronous loop: MCTS with the current weights collects real TRAIN
trajectories, the trajectory replay feeds the Stage 4.4 learner, the learner
writes new weights, and the **next** MCTS searches use them. Validation is the
same corrected independent-episode pipeline PPO uses, and the selection metric
is `mean_episode_log_return`.

Still **not** implemented (deferred to Stage 4.6+): actor multiprocessing,
batched/parallel MCTS, reanalysis, prioritized or decision-rich replay, HOLD
downsampling, target refreshing, distributed learners, Stochastic MuZero, and
any production-scale run. The frozen Forex reward `r_t = log(equity[t+1] /
equity[t])`, execution timing, accounting, currency conversion, split semantics
and the ten-action environment are untouched; MuZero still projects onto its own
six actions with no HOLD bonus, penalty or frequency shaping.

## 1. Files added / changed

Added:

| File | Purpose |
|---|---|
| `forexmind/muzero/trainer.py` | `MuZeroTrainingConfig`, `TrainingProgress`, `MuZeroTrainer` (lifecycle, warm-up, ratio, network version, staleness, checkpoints, resume, progress block, timing/throughput). |
| `forexmind/muzero/diagnostics.py` | `RootSearchRecord`, `search_summary`, `action_frequency_diagnostics`, `staleness_summary`, `assert_records_are_legal` (S16-S22, S33-S35, S39). |
| `forexmind/muzero/evaluation.py` | `MuZeroAgent`, `MuZeroEvaluator`, `MuZeroEvaluation` - deterministic MCTS evaluation on the PPO pipeline (S23-S26). |
| `forexmind/muzero/replay_store.py` | `save_replay` / `load_replay` / `replay_store_report`, one compressed shard per trajectory plus a JSON index (S30). |
| `forexmind/muzero/train_muzero.py` | CLI launcher, `python -m forexmind.muzero.train_muzero`. |
| `tests/test_muzero_trainer.py` | 12 tests: lifecycle, warm-up gating, ratio, diagnostics, staleness, checkpoints, resume, replay round-trip, mask violations, selection metric. |
| `docs/stage45_muzero_training.md` | This report. |

Changed:

| File | Change |
|---|---|
| `forexmind/muzero/trajectory.py` | `TrajectoryMetadata.network_version` (default 0) so replay staleness is measurable. |
| `forexmind/muzero/collector.py` | `CollectorConfig.network_version` / `.capture_diagnostics`, `CollectedTrajectory`, `collect_with_diagnostics`, `collect_next_with_diagnostics`; the per-root trace is returned to the trainer and never stored in replay. |
| `forexmind/muzero/__init__.py` | Exports for the trainer, evaluator, diagnostics and replay store. |
| `README.md` | Stage 4.5 section. |

No Forex, environment, accounting or PPO module was modified.

## 2. Integrated loop architecture

```
MuZeroTrainer.train()
  while env_steps < max_env_steps:
      collect_phase()      MuZeroCollector.collect_next_with_diagnostics()  x trajectories_per_iteration
                           -> TRAIN assertion -> replay.add(trajectory)
                           -> per-root records -> diagnostics.search_summary()
      learn_phase()        if warm-up met: replay.sample() -> learner.train_step()  x learner_updates_per_iteration
                           -> network_version += 1 per update group
      _maybe_evaluate()    MuZeroEvaluator.evaluate() on fixed VALIDATION specs
      _maybe_checkpoint()  latest / step_<env_steps> / best
      progress block       counters, losses, search diagnostics, action mix, rates
```

The trainer owns **one** model object shared by the collector and the learner, so
collection and learning cannot diverge. `collector_model_version` and
`learner_model_version` (Stage 4.4 parameter fingerprints) are recomputed at
checkpoint time and stored; because collection happens before learning inside an
iteration, the pair differs exactly by the updates of that iteration and they
agree again at the start of the next collection phase.

## 3. Collection / learning ratio, warm-up, model versioning

* `trajectories_per_iteration = 2` real TRAIN episodes (32 environment steps at
  horizon 16) and `learner_updates_per_iteration = 8` in the experiment below:
  a conservative **0.25 learner updates per environment step**, logged every
  iteration. The brief's warning about SAC's actor-idle over-updating is
  respected: the ratio is explicit configuration, not implicit.
* `min_replay_transitions_before_training = 64`: iterations 1-2 collect only;
  learning starts in iteration 3 and the code prints the warm-up state in every
  progress block.
* `network_version` starts at 0, increments once per update group, and is written
  into every collected trajectory's metadata (S7). `collector_model_version` /
  `learner_model_version` are hashed parameter fingerprints (S8).

## 4. Replay behaviour

Stage 4.3 `TrajectoryReplayBuffer` with `max_trajectories = 32`, uniform
position sampling, no prioritization and no HOLD downsampling. In the experiment
the buffer filled to capacity (32 trajectories / 512 transitions / 0.44 MB) and
FIFO eviction began; `replay.memory_report()` and `sampling_diagnostics()` are
re-emitted in every report. Replay composition over the run:
HOLD 41.2 %, FLAT 17.4 %, SHORT 21.3 %, LONG 20.1 %.

## 5. MCTS training / validation configuration

Training search: `num_simulations = 16`, PUCT with `discount = 0.99`, Dirichlet
root noise **enabled**, root visit temperature from a configurable schedule
(`constant` here at 1.0; `linear` decays to `temperature_end` over
`temperature_decay_steps`). Evaluation search: identical budget with noise
**disabled** and `temperature = 0` (visit argmax). Exploration therefore comes
only from PUCT, Dirichlet noise and temperature.

## 6. Validation, selection metric, baselines

Validation uses `MuZeroEvaluator` on fixed, balanced, **non-overlapping**
`VALIDATION` episode specifications (`EpisodeSampler.sample_non_overlapping`),
the same `EvaluationRunner` and the same `sampled_report` as PPO;
`selection_metric_name = mean_episode_log_return`. Reported per evaluation:
mean/median episode return, mean log return, profitable fraction, p10/p90,
mean turnover, mean executions. No portfolio Sharpe is used, and the matched
FLAT / enter-and-hold references are available through
`MuZeroEvaluator.evaluate_matched_baselines` as diagnostic-only comparisons.
Validation trajectories are returned to the caller and never enter replay.

## 7. Checkpoints, resume and replay persistence

`latest.pt`, `best.pt` (best corrected validation score) and
`step_<env_steps>.pt` are written separately. Each payload carries model +
optimizer state, `env_steps`, `gradient_updates`, `trajectories_collected`,
`network_version`, `best_validation_score`, `best_checkpoint_step`, config, RNG
state, replay metadata, `selection_metric_name`, `num_actions = 6`,
`num_simulations`, `unroll_steps`, `td_steps`, `discount` and the
reward/value representation type.

Replay is persisted **outside** the checkpoint as shards
(`<output_dir>/replay/replay_shard_XXXXX.npz` + `replay_index.json`).
`trainer.resume(path)` restores model/optimizer/counters/RNG and, when the store
exists, reloads the replay in insertion order; the returned report states
`replay_restored`, `replay_trajectories`, `replay_dropped_on_load` and
`exact_continuation`, so a resume with an empty replay can never be mistaken for
exact continuation.

## 8. End-to-end smoke test (S31)

`python -m forexmind.muzero.train_muzero --instruments EURUSD GBPUSD --horizon 6
--num-simulations 4 --max-env-steps 12 --eval-episodes 2 --eval-horizon 6 ...`
verified the full lifecycle on real processed data: 2 trajectories / 12 env
steps collected, replay filled, warm-up released the learner at 6 transitions,
4 gradient updates over 2 network versions, search diagnostics present,
validation ran, `latest.pt` / `best.pt` / `step_12.pt` written, and the report
JSON emitted.

## 9. Small integrated experiment (S32-S37)

`python -m forexmind.muzero.train_muzero --instruments EURUSD GBPUSD USDJPY
--horizon 16 --num-simulations 16 --trajectories-per-iteration 2
--min-replay-transitions-before-training 64 --learner-updates-per-iteration 8
--batch-size 16 --unroll-steps 5 --td-steps 5 --latent-dim 64 --hidden-dim 64
--num-layers 1 --max-env-steps 512 --eval-every-env-steps 256 --eval-episodes 4
--eval-horizon 32 --checkpoint-every-env-steps 256 --max-trajectories 32`

Counters: **512 environment steps**, 32 trajectories, 544 MCTS searches,
**120 gradient updates**, 15 network versions, 0.44 MB replay.

Learner (mean over the first vs last five update groups - it is fitting its own
replay):

| metric | first (env 64) | last (env 512) | change |
|---|---:|---:|---:|
| total loss | 7.197 | **2.914** | −59.5 % |
| value loss | 2.688 | **1.349** | −49.8 % |
| reward loss | 2.973 | **0.201** | −93.2 % |
| policy loss | 1.536 | 1.364 | −11.2 % |
| policy KL | 0.639 | 0.540 | −15.5 % |
| value MAE | 0.248 | **0.114** | −54.0 % |
| reward MAE | 0.0910 | **0.0099** | −89.2 % |

Search behaviour (collection-time, S16-S22, S33-S35): MCTS changes the network
prior's argmax on **50 %** of roots (78 % in the first evaluation, so search is
not a copy of the prior); mean root visit entropy 0.85-0.94 nats; mean tree depth
5.9-7.0 with max depth above 7; MCTS‖prior KL 0.41-0.55; reward-model MAE falls
from 0.0204 to 0.0087; `|search value − network value|` 0.070 → 0.050. The prior
is HOLD-argmax on 84-100 % of roots while MCTS selects HOLD on 37-50 %, i.e.
search is actively moving off the prior's HOLD default. Invalid actions received
zero visits and were never selected (asserted every iteration).

Validation (4 fixed episodes, horizon 32, temperature 0): mean episode log
return −9.96e-05 at 32 env steps (untrained) versus −2.22e-04 at 288 env steps.
Mean executions fell 10.75 → 0.25 and turnover 10.13 → 0.125, i.e. the policy
drifted toward inaction.

Timing / throughput: 23.9 env steps/s, 25.4 searches/s, 406 recurrent
inferences/s, 16.2 learner updates/s, 260 samples/s. Wall-time split:
**collection 48 %**, **evaluation 35 %**, learning 17 % of 44.5 s.

## 10. Remaining concerns

1. **No evidence of validation improvement yet.** 512 environment steps and 4
   validation episodes cannot distinguish learning from noise, and the observed
   validation score got *worse* while the learner was clearly fitting its replay.
   This is the honest state of "the loop works", not "MuZero is learning Forex".
2. **Early HOLD drift.** Both the untrained prior and the emerging policy lean
   HOLD-heavy; turnover collapsed during the run. Since validation is HOLD-like
   in return terms at this scale, checkpoint selection cannot yet penalise it.
3. **Best-checkpoint noise.** With 4 episodes the best score was captured at the
   very first evaluation (32 steps). Selection needs more episodes before it
   means anything.
4. **Replay staleness grows** (mean 6.6-8.2, max 14 network versions) because
   trajectories are reused while the weights keep moving - expected without
   reanalysis, and now measurable.
5. **Replay persistence is a shard directory**, not a memory-mapped store; a very
   large replay would be slow to save. FIFO order is preserved and capacity
   drops are reported.
6. **CPU only, single process.** Evaluation, not learning, was the second-largest
   cost; a larger experiment needs more validation episodes, which the current
   synchronous loop pays for serially.
7. Not run: the brief's 50k-200k step experiment, held-out TEST evaluation, any
   GPU path, and multi-seed reproducibility of the loop.

## 11. Readiness for Stage 4.6 scaling

Ready, with the caveats above:

* the complete lifecycle runs end to end on real data and is covered by 12
  focused tests plus 341 passing MuZero tests overall;
* collection and learning are explicit, measured and ratio-controlled;
* every collected trajectory records the network version that produced it, so
  staleness is observable before any parallel collection is introduced;
* search diagnostics (prior vs MCTS, depth, visit concentration, reward model,
  masks) are separated from learner diagnostics as the brief requires;
* checkpoints, best/latest separation and shard-based resume all work.

Recommended order for Stage 4.6: (1) raise validation episodes and use them to
decide whether the objective's improvements transfer at all; (2) run the
50k-200k step experiment with the same configuration; (3) only then add
parallel actors / batched MCTS, reanalysis and prioritized replay.
