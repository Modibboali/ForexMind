# Stage 4.6 — Scaling MuZero Collection, Batched MCTS Inference and Replay Throughput

Implementation date: 2026-09-14. Stage 4.6 turns the Stage 4.5 synchronous loop
into a *scalable* collector/learner system without touching the MuZero
objective, the reward, the action space, the validation protocol or the search
mathematics. Everything below is measured on this machine, and every table
states the configuration it was measured with.

> **Headline.** Collection throughput rises from **15.3 env steps/s** (single
> process, Stage 4.5 path) to **78.0 env steps/s** (8 workers × 4 collectors,
> local batched inference) — **5.1×** on a 4-physical-core CPU-only box, with
> search results unchanged (verified numerically, S34). The central inference
> server also scales (**63.5 env steps/s**, 4.2×) and is the right design when
> the model must live on one GPU, but on CPU-only hardware it is IPC-bound.
> Two real bugs were found and fixed by this work: an intermittent float32
> mask-precision failure in stored trajectories, and a replay-sampling path that
> is a co-bottleneck with the learner (~170 samples/s).

## 0. What was added

| File | Purpose |
|---|---|
| `forexmind/muzero/inference_service.py` | Batched inference service: `InferenceRequest`/`InferenceResponse`, `BatchedInferenceServer` (atomic active/spare weight swap), `LocalInferenceBackend`, `RemoteInferenceBackend`, `InferenceStats` (S9-S13, S17, S30). |
| `forexmind/muzero/parallel_collector.py` | `MuZeroCollectorPool`, worker processes, bounded trajectory queue, single `ReplayWriter` thread, `LockedReplay`, `WorkerDatasetSpec`, trajectory wire format, worker-failure detection (S3-S8, S19-S21, S36-S38). |
| `forexmind/muzero/profiling.py` | `PhaseTimer` + `merge_phase_reports` for the S2 phase audit (trainer- and worker-side). |
| `forexmind/muzero/search.py` | `MuZeroMCTS.search_batch(...)`: `B` independent roots driven in lock-step with one batched inference call per simulation round; per-root RNG streams; `search()` delegates to it, so single- and multi-root search cannot drift apart (S9, S14-S15). |
| `forexmind/muzero/collector.py` | `EpisodeRun` (step-driven episode shared by the sequential collector and the workers), deterministic `derive_episode_seed` / `derive_search_seed` / `derive_decision_seed`, `per_decision_seed`, `CollectionStats.absorb`, float64 planning exposure. |
| `forexmind/muzero/trainer.py` | Optional parallel mode (`num_collectors > 1`), inference-weight synchronization schedule, `target_updates_per_env_step` ratio guard, profiling, pool diagnostics and clean shutdown. |
| `forexmind/muzero/train_muzero.py` | CLI flags for all of the above. |
| `tools/benchmark_muzero_scaling.py` | Worker-scaling and simulation-budget benchmark (S28-S30, S44). |
| `tools/profile_muzero_phases.py` | End-to-end phase profile of the integrated loop (S2, S45). |
| `tools/benchmark_muzero_replay.py` | Replay insertion / sampling / store benchmark (S20-S22). |
| `tools/smoke_muzero_parallel.py` | Parallel-collection smoke test with worker / queue / inference diagnostics (S31). |
| `tests/test_muzero_parallel.py` | 19 tests: batched-inference equivalence, batched-search equivalence and invariants, seeding, wire format, pool lifecycle, backpressure, worker failure, inference failure/timeout, single-vs-parallel equivalence, mask-precision regression, trainer integration. |

Frozen contracts: the Forex environment, the reward
`r_t = log(equity[t+1]/equity[t])`, execution timing, accounting, the ten-action
environment, MuZero's six-action projection, the PUCT equation, the MuZero loss,
target construction, the selection metric (`mean_episode_log_return`) and the
validation pipeline are unchanged. With `num_collectors <= 1` every code path is
the Stage 4.5 one.

## 1. Baseline: what Stage 4.5 actually costs (S2, S44)

Measured with `tools/profile_muzero_phases.py` (single process, mmap dataset,
latent 64 / hidden 64 × 1 layer, 3 instruments, horizon 16, 16 simulations,
256 environment steps, 30 progress iterations). Phase timings are exclusive
wall time; `collection_phase` is the enclosing wrapper and is therefore larger
than the sum of its parts.

| phase | seconds | share | calls | ms/call |
|---|---:|---:|---:|---:|
| validation | 396.0 | 85.5 % | 2 | 198 002 |
| collection_phase (wrapper) | 34.0 | 7.3 % | 8 | 4 250 |
| recurrent inference | 16.6 | 3.6 % | 4 352 | 3.83 |
| MCTS expand + backup | 5.3 | 1.2 % | 4 352 | 1.23 |
| learner forward+backward | 4.7 | 1.0 % | 56 | 83.7 |
| replay sampling | 2.0 | 0.4 % | 56 | 36.6 |
| environment step | 1.8 | 0.4 % | 256 | 6.95 |
| initial inference | 0.9 | 0.2 % | 272 | 3.49 |
| MCTS tree logic (PUCT + descent) | 0.8 | 0.2 % | 4 352 | 0.18 |
| checkpointing | 0.4 | 0.1 % | 4 | 108 |
| observation encode | 0.3 | 0.1 % | 272 | 0.98 |

Conclusions, measured rather than assumed:

* **The network is not FLOP-bound; per-call dispatch is.** A 16-simulation
  search costs 77.4 ms on this machine (`benchmark_muzero_search`, latent 64 /
  hidden 64 × 2), i.e. **~4.5 ms per single-row inference call**. That is the
  dominant term in collection (initial + 16 recurrent calls ≈ 82 ms per
  environment step ≈ 12 steps/s at best) and it is Python/`torch` dispatch
  overhead for a tiny MLP.
* **Validation dwarfs everything** with the default `eval_horizon=512`:
  2 evaluations × 4 episodes × 512 steps = 4 096 environment steps at ~10
  steps/s ≈ 396 s versus 34 s of collection. Large runs must evaluate less
  often, use a shorter evaluation horizon, or evaluate out-of-band.
* **Replay sampling is a co-bottleneck with the learner**: 36.6 ms per
  16-sample batch versus 83.7 ms for the learner step (see §10).
* Environment stepping and observation encoding are ~8 % of collection: the
  environment is *not* the bottleneck, contrary to the SAC-era assumption.
* The Stage 4.5 report's published 23.9 env steps/s is **not reproducible on
  this machine today**; the same configuration re-measured here gives
  **9.2 env steps/s** with 4 BLAS threads and **15.3 env steps/s** with
  `torch.set_num_threads(1)` (see §5). All comparisons below use the
  re-measured values.

Learner, replay and staleness in the same run (batch 16, unroll 5, 56 updates):

| metric | value |
|---|---:|
| learner update | 83.7 ms (8.3 updates/s, 133 samples/s) |
| replay sampling per update | 36.6 ms (219 samples/s at batch 16) |
| replay staleness (mean / max) | 3.34 / 6 network versions at 256 env steps |
| checkpoint (model + shard replay) | 108 ms per write |
| CPU / GPU | 8 logical CPUs (4 physical), **no CUDA device** — every measurement here is CPU-only |

## 2. Parallel collector architecture (S3, S6, S19)

```text
collector worker 0  (environments, RNG streams, MCTS trees — never shared)
collector worker 1  ...
collector worker N-1
        |  trajectory payload (NumPy arrays + metadata + optional search records)
        v
  trajectory queue  (bounded, default 8)      <- backpressure lives here
        v
  ReplayWriter thread (single writer, owns insertion and capacity enforcement)
        v
  TrajectoryReplayBuffer  <- the trainer samples under the same lock
```

* One **collector** = (environment, RNG stream, episode state, MCTS tree).
  `num_collectors = num_workers × collectors_per_worker`; the collectors inside
  one worker are driven in lock-step (§3) and each owns a *private* environment
  object (the market data behind it is shared read-only).
* Workers are `spawn` processes. The parent never step-blocks on a fixed worker
  order: `collect_trajectories(n)` waits on the writer, not on a worker.
* Collectors never touch replay. `LockedReplay` gives the writer thread and the
  learner's sampling the same lock, so sampling can never observe a
  half-inserted trajectory.
* The trainer accounts for **learner-visible** environment steps (`env_steps`)
  separately from **produced** steps (`env_steps_produced`), because collectors
  deliberately run ahead.

## 3. Batched MCTS inference (S9-S15, S31)

`MuZeroMCTS.search_batch(observations, masks, planning_states=None, rngs=None)`
drives `B` roots through the same `num_simulations` rounds. Each round performs
the Stage 4.2 descent/expansion/backup for every root and merges *only* the leaf
transitions into one `recurrent_inference` call:

```text
round k:  root A leaf -+
          root B leaf -+- > recurrent_inference([B, latent], [B]) -> B rewards/values/priors
          root C leaf -+
```

Not shared between roots: visit counts, Q values, children, planning state,
`MinMaxStats` and RNG streams (S15). With `B = 1` the call shapes are exactly the
Stage 4.2 ones, and `search()` is literally `search_batch(...)[0]`, so the
single-root and multi-root paths cannot drift apart.

Evidence (tests, not claims):

* `test_batched_search_with_one_root_is_identical_to_search` — batch-1 search is
  *bit-identical* to the Stage 4.2 search (visit counts, policy, action, root
  value, priors, recurrent-call count).
* `test_batched_search_matches_independent_per_root_searches` — each root of a
  3-root batch gets exactly the visits/policy/priors/action it would get alone;
  root values agree to 1e-5 relative (batch-shape reduction order only, S13).
* `test_batched_search_preserves_tree_invariants` — visit counts sum to
  `num_simulations`, invalid actions receive zero visits, the root policy sums
  to 1, HOLD stays valid (S31).
* Search records are still produced per real decision (`capture_diagnostics`),
  so the Stage 4.5 prior/MCTS/reward/value diagnostics are unchanged.

Measured batch sizes (S30): the batching unit is the *worker*, not the pool.
`mean_batch_size` equals `collectors_per_worker` (2.00 and 4.00 in the tables
below) and is never pulled up by other workers' requests, because a worker is
latency-bound: it has one request in flight at a time, so the server usually
finds at most one worker's batch queued. Raising `max_batch_wait_ms` merges more
rows per call but adds that latency to every request, and with a dispatch-bound
model the trade is not free; the knob that actually increases the batch is
`collectors_per_worker`.

## 4. Inference service, synchronization and failures (S11-S13, S16-S18, S38)

`BatchedInferenceServer` runs one thread in the trainer process and owns **two**
model copies (`active`/`spare`). `sync_weights` loads the learner's state dict
into the spare copy and then swaps references under the same lock that guards a
running batch, so inference can never observe a half-loaded parameter set
(S17). Requests carry `initial`/`recurrent` kinds; the server groups them, runs
one model call per group and answers per request
(`test_service_batching_preserves_values` checks priors/values/latents against
the local backend to 1e-6).

Version bookkeeping (S16, S18): `learner_network_version` is the checkpointed
counter, `inference_network_version` is stamped on every response, and every
trajectory records the inference version that produced its decisions
(`metadata.network_version`). Staleness is `current - trajectory_version` and is
reported per iteration as mean/median/p90/max.

Failures are loud: a worker death raises `CollectorWorkerError` with `worker_id`,
`pid`, `exitcode`, last trajectory id and the worker's own error (S37); an
inference exception marks the server fatal and every pending request receives an
error response, so clients raise `InferenceServiceError` instead of deadlocking
(S38); a slow service raises `InferenceTimeoutError`. Measured batch wait and
latency are reported for every run (§9).

## 5. Torch threading and CPU affinity (S26-S27)

* Machine: 8 logical CPUs (4 physical cores, SMT), affinity spans all 8, torch
  default 4 threads, no `OMP/MKL/OPENBLAS` overrides.
* Worker processes run `torch.set_num_threads(1)` (configurable). The effect is
  large on this workload: the *same* single-process collector measures
  **9.2 env steps/s** with 4 threads and **15.3 env steps/s** with 1 thread
  (1.66×). Per-call BLAS threading overhead dominates a 64-dimensional MLP.
* The benchmark's sequential baseline uses the worker threading setting so the
  comparison is apples-to-apples.

## 6. Dataset memory behaviour (S5)

Workers rebuild the dataset from a description and prefer the shared
memory-mapped store (`resolve_dataset(backend="auto")`): every worker maps the
same files, so the OS shares the physical pages.

| mode | workers | aggregate RSS | aggregate USS (private) | RSS/worker |
|---|---:|---:|---:|---:|
| 1 worker × 4 collectors | 1 | 665 MB | 402 MB | 665 MB |
| 2 × 4 | 2 | 1 301 MB | 778 MB | 651 MB |
| 4 × 4 | 4 | 2 591 MB | 1 498 MB | 648 MB |
| 8 × 4 | 8 | 5 106 MB | 3 000 MB | 638 MB |

RSS/worker is flat across worker counts, the expected signature of shared pages
(the same code with per-worker parquet copies showed ~1.6 GB for a single
process, §7). USS is *not* flat: each worker still holds ~400 MB of private
memory (pandas frames plus the per-instrument environment and window builder).
The shared store fixes dataset duplication, not the frames a worker
materialises; reducing that ~400 MB is the main remaining memory task for
16-32 worker runs.

Speed cost of the shared store: with 4 torch threads the same single-process
benchmark gives 8.4 env steps/s (mmap) versus 9.2 env steps/s (parquet), i.e.
**≈9 % slower hot-path access** in exchange for the shared pages. That is a good
trade at scale and a small one at small scale.

## 7. Worker scaling table (S28, S44)

3 instruments shared across all workers, horizon 16, 16 simulations, latent 64 /
hidden 64 × 1 layer, `torch.set_num_threads(1)`, mmap dataset, 24-36 measured
trajectories per configuration after 4-8 untimed warm-up trajectories.
Throughput is computed on **production timestamps**, which makes it immune to
worker run-ahead and to how quickly the trainer drains the queue (an early
version of the benchmark measured `take()` wall time and reported an
impossible 141 env steps/s for one configuration; the production window is the
honest measure). No worker exceeded its queue: blocked time 0.0 s, 0 dropped
trajectories in every run.

### 7a. Single process baseline (Stage 4.5 path, `--sequential`)

| collectors | torch threads | env steps/s | searches/s | simulations/s | CPU cores | process RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | **15.3** | 16.3 | 260 | 1.02 | 1.61 GB |
| 1 | 4 | 9.2 | 9.8 | 157 | 0.94 | 1.61 GB |

### 7b. Parallel collectors, central batched inference (`--inference-mode server`)

2 collectors per worker:

| workers | collectors | env steps/s | searches/s | sims/s | mean batch | CPU cores | worker RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 5.8 | 6.2 | 99 | 2.00 | 2.78 | 537 MB |
| 2 | 4 | 12.2 | 13.0 | 208 | 2.00 | 3.44 | 1 184 MB |
| 4 | 8 | 24.9 | 28.2 | 452 | 2.00 | 4.42 | 2 315 MB |
| 8 | 16 | **63.5** | 55.5 | 888 | 2.00 | 5.81 | 4 457 MB |

### 7c. Parallel collectors, local batched inference (`--inference-mode local`)

4 collectors per worker:

| workers | collectors | env steps/s | searches/s | sims/s | mean batch | CPU cores | worker RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 13.3 | 14.1 | 226 | 4.00 | 1.00 | 665 MB |
| 2 | 8 | 22.3 | 26.3 | 420 | 4.00 | 1.97 | 1 301 MB |
| 4 | 16 | 49.5 | 52.2 | 836 | 4.00 | 3.87 | 2 591 MB |
| 8 | 32 | **78.0** | 89.9 | 1 438 | 4.00 | 6.27 | 5 106 MB |

Reading the tables:

* Local batched inference beats the central service at every worker count on
  this machine (78.0 vs 63.5 env steps/s at 8 workers; 49.5 vs 24.9 at 4): with
  CPU-only inference and a tiny model, an IPC round trip (5-10 ms measured) is
  far more expensive than the inference it carries.
* **One worker is not enough to pay for the machinery**: 1 worker × 4 collectors
  gives 13.3 env steps/s, slightly *below* the 15.3 env steps/s single-process
  baseline, because driving four trees in lock-step adds Python bookkeeping and a
  4-row batch is not cheaper per row for this model. Scaling starts at 2 workers
  (1.46×) and is strongly superlinear up to 4 (3.24×) as in-flight inference
  calls multiply.
* Saturation: 4 → 8 workers adds 1.58× (local) at 6.27 of 8 logical cores (4
  physical). **16 collectors (4 workers × 4) is the knee; 32 collectors is the
  practical ceiling on this box.** The brief's warning not to set
  `num_workers = logical_cpu_count` is confirmed: 8 workers ≥ 32 collectors buy
  little over 4 workers × 4, and the central server degrades relative to local
  mode as workers multiply.
* Throughput per collector: ~3.3 steps/s at 1 worker × 4 collectors versus
  ~15.3 steps/s for the single-process baseline. A wider batch is *not* cheaper
  per row for this model; the parallel win comes from using more cores, not from
  cheaper model math. On a GPU (S23-S25) the balance inverts and the batched
  service is the design that matters.

## 8. MCTS simulation-budget table (S29)

4 workers × 2 collectors, 48 measured trajectories, local inference, otherwise
identical to §7.

| simulations | env steps/s | searches/s | simulations/s | mean search latency | CPU cores |
|---:|---:|---:|---:|---:|---:|
| 8 | 47.1 | 52.1 | 417 | 19 ms | 3.72 |
| 16 | 38.2 | 40.6 | 650 | 25 ms | 3.63 |
| 32 | 27.9 | 29.9 | 955 | 33 ms | 3.80 |
| 64 | 18.0 | 19.3 | 1 235 | 52 ms | 3.79 |

Cost/quality frontier: 16 → 32 simulations costs 27 % of environment throughput
for 1.47× the search work per decision; 32 → 64 costs a further 35 % for 1.29×.
Since Stage 4.5 could show no validation improvement at 16 simulations, the
recommended next experiment is **32 simulations** (§14), with the throughput
penalty now known (27.9 vs 38.2 env steps/s at 8 collectors — still 1.8× the
single-process baseline).

## 9. Inference batch statistics (S30)

| mode | collectors/worker | batch mean | batch p90 | batch max | mean batch wait | mean inference latency | calls/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 4 | 4.00 | 4.00 | 4.00 | none (no queue) | 3.1 ms (4 rows) | ~230 |
| server, 8 workers | 2 | 2.00 | 2.00 | 2.00 | 12.9 ms | 7.1 ms | ~110 |

A batching layer that runs at batch size 1 would be pointless; here the batch
always equals the number of collectors in a worker, and the server's *wait*
(12.9 ms mean, mostly Windows queue transit) is what keeps cross-worker merging
rare. These diagnostics are emitted in every trainer report, so a future
`collectors_per_worker` or `max_batch_wait_ms` change is directly measurable.

## 10. Replay behaviour and throughput (S19-S22)

`tools/benchmark_muzero_replay.py --episodes 256 --horizon 64` (16 384
transitions, 24.2 MB in RAM):

| measurement | value |
|---|---:|
| insertion (with validation) | 6.4 ms/trajectory |
| sampling, batch 32 | 171 samples/s (187 ms/batch, 5.8 ms/sample) |
| sampling, batch 64 | 169 samples/s (379 ms/batch) |
| sampling, batch 128 | 172 samples/s (745 ms/batch) |
| sampling, batch 256 | 164 samples/s (1 559 ms/batch) |
| trajectory payload (horizon 64) | 92.4 KB arrays / 93.4 KB pickled |
| shard store write / read | 7.3 s / 5.5 s for 22.8 MB |

Findings:

* Sampling cost is **linear in batch size with no amortisation** (~5.8 ms per
  sample), because each sample builds a full unroll target set in Python. At
  batch 32 that is 187 ms per learner batch — the same order as the learner step
  itself (83 ms at batch 16). Replay sampling must be vectorised before large
  runs; it is *not* something parallel collectors can fix.
* Insertion is cheap (6.4 ms per 64-step trajectory including validation) and
  the writer thread kept up in every run (0.0 s blocked time, 0 dropped
  trajectories), so the single-writer design is sufficient at this scale.
* The shard store is slow (≈7.9 MB/s write): fine for checkpoints of a small
  replay, not for a multi-GB replay. This is the concrete argument for a
  memory-mapped replay backend later; the common `TrajectoryReplayBuffer`
  interface and the writer are already backend-agnostic.
* Serialization is not a bottleneck: a horizon-16 payload is ~24 KB and the
  measured handoff (production timestamp → insertion) averaged 4-8 ms.

## 11. Correctness: numerical equivalence and single-vs-parallel (S13, S31-S34)

* **Batched vs individual inference** (`test_compute_batch_matches_individual_calls`,
  `test_service_batching_preserves_values`): identical to 1e-6 for latents,
  priors and values, over initial and recurrent calls, through the local
  backend, the wire format and the service.
* **Batched vs sequential search**: bit-identical at batch 1; per-root
  equivalence to 1e-5 (root value) with identical visits/policy/priors/action.
* **Single-process vs parallel collection (mandatory, S34)** —
  `test_single_process_and_parallel_collectors_agree`: with identical weights,
  identical episode specification and `per_decision_seed=True`, a pooled worker
  reproduces the sequential collector **action for action**, with the same
  masks and exact rewards, and policies/observations/root values to 1e-6/1e-5.
  The same seeding scheme makes a parallel run reproducible under a fixed
  configuration regardless of scheduling, and `worker_rank = 0` reproduces the
  Stage 4.5 seed expressions bit-for-bit.
* **Bug found by this work**: a stored `planning_exposure` rounded to float32
  could flip a six-action mask for states within ~1e-7 of the 1e-3 exposure
  tolerance, raising
  `action_masks[t] disagrees with the deterministic planning state` mid-run
  (observed with 8 workers × 4 collectors; not reproducible on demand because
  model initialisation is random). Fixed by storing the exposure in float64 and
  guarded by
  `test_collected_trajectories_store_planning_exposure_in_double_precision`.
  This was a latent Stage 4.3/4.5 defect that parallel runs exposed.
* Determinism of worker streams: `derive_episode_seed` / `derive_search_seed` /
  `derive_decision_seed` give distinct, reproducible streams per
  `(global_seed, worker_rank, episode, decision)`
  (`test_worker_seed_derivation_is_deterministic_and_distinct`,
  `test_worker_rank_changes_episode_choices`).

## 12. Lifecycle: queues, checkpoints, resume, shutdown, failures (S7, S35-S40)

* **Bounded queue / backpressure (S7)**: the trajectory queue is bounded
  (`trajectory_queue_size`, default 8). Workers block on a full queue and report
  `blocked_seconds`; trajectories are dropped only when `drop_when_full` is
  explicitly enabled, and drops are counted and reported. In every benchmark run
  blocked time was 0.0 s with 0 drops.
* **Clean shutdown (S36)**: `pool.close()` signals every worker, joins with a
  timeout, terminates only as a last resort (reported in
  `shutdown.workers_terminated`), flushes the queue into replay, stops the
  inference server and the writer, and reports
  `alive_workers / queue_drained / queue_leftover / trajectories_written`.
  `MuZeroTrainer.train()` closes the pool in a `finally`, and the trainer report
  is generated *after* shutdown so it carries the real numbers. Tests assert
  `alive_workers == 0` and no live processes after close.
* **Worker failure (S37)**: `test_worker_failure_is_reported` kills a worker and
  asserts `CollectorWorkerError` with worker id and exit code; the trainer aborts
  instead of continuing with missing collectors.
* **Inference failure (S38)**: `test_inference_service_failure_propagates`
  (a raising model → error response → `InferenceServiceError`, server marked
  fatal, second request fails fast) and `test_inference_timeout_propagates`.
* **Checkpoint / resume (S39-S40)**: checkpoints keep the Stage 4.5 payload
  (model, optimizer, counters, `network_version`, RNG, replay metadata) and add
  the parallel collection configuration. Resume restores the replay store and
  shifts the pool's episode-id space past the restored ids, so new trajectories
  cannot collide. Resume does **not** restore in-flight collector episode
  state: it starts new trajectories on the restored weights, which the resume
  report states explicitly through `exact_continuation`.
* **Longer stress runs (S35)**: the benchmark configurations exercised queue
  fill/drain (32 collectors against a queue of 8), replay capacity eviction,
  checkpoint and resume (`tests/test_muzero_trainer.py`), and collector
  start/stop. No deadlocks, no orphan processes, no lost or duplicated
  trajectory ids were observed; ids are partitioned per worker
  (`worker_id * stride + episode`) and asserted unique in the pool tests. A
  multi-hour soak was **not** run.

## 13. End-to-end smoke tests

* `python -m tools.smoke_muzero_parallel --workers 2 --collectors-per-worker 2
  --simulations 4 --horizon 4 --trajectories 4` — 4 real trajectories, distinct
  seed streams, replay written, inference batches reported, clean shutdown,
  worker RSS/USS printed before and after collection.
* `tests/test_muzero_parallel.py::test_trainer_parallel_mode_collects_learns_and_shuts_down`
  — the full loop with `num_collectors=2`: collection → replay → learner →
  network-version bump → weight synchronization → validation → checkpoint →
  shutdown diagnostics, plus the merged trainer/worker phase profile and the
  ratio guard.

## 14. Dominant remaining bottleneck (S45) and readiness (S48)

Ranked by measured evidence:

1. **Per-call neural dispatch** (~4.5 ms for one row, 3.1 ms for four rows) — the
   reason a single search costs 77 ms, and why batching plus more cores, not
   fewer FLOPs, is the lever. On GPU this term shrinks; the batched service is
   ready for it (`learner_device` / `inference_device` already exist;
   non-blocking host→device transfer is the remaining wiring).
2. **Replay sampling** (~5.8 ms per sample, linear in batch size) — now the
   co-bottleneck of learning and independent of collection.
3. **Python-side MCTS bookkeeping** (1.2 ms per simulation for expand + backup,
   0.18 ms for descent) — the next largest collection term after inference.
4. **Validation** when configured with long horizons (85 % of wall time at
   `eval_horizon=512`).
5. **Environment stepping** (7 ms per step, ~7 % of collection) and observation
   encoding (~1 ms per step).
6. IPC/serialization and replay insertion are *not* bottlenecks at this scale
   (0.0 s blocked, 6.4 ms insertion). They would matter for a central GPU
   service with many CPU workers, where the measured 5-10 ms round trip is the
   limiting factor.

Recommendations:

* **Worker count**: `num_collectors = 16` (4 workers × 4 collectors) with
  `inference_mode="local"` on this CPU-only box (3.24× the single-process
  baseline at 3.9 of 4 physical cores); 32 collectors (5.1×) only when the
  machine is otherwise idle. On a single-GPU machine use
  `inference_mode="server"` with 4-8 workers × 2-4 collectors and expect the GPU
  to remove the inference term.
* **Simulation count**: 32 for the next learning experiment; 16 remains the
  throughput-oriented choice (38.2 vs 27.9 env steps/s at 8 collectors).
* **Ready for a larger run?** The collection/learning pipeline, seeding,
  versioning, checkpointing, resume and shutdown paths are implemented, tested
  and numerically verified, and the client/server equivalences hold. Before a
  50k-200k step run, fix **replay sampling** (vectorised batch construction) and
  cut **validation** cost (shorter or less frequent evaluation); otherwise those
  two dominate wall time no matter how many collectors are added.

## 15. Deferred (explicitly not done, per brief S46-S47)

No reanalysis, prioritized replay, decision-rich sampling, HOLD downsampling,
Stochastic MuZero, multi-node training, population-based tuning or large
hyperparameter sweeps. Also deferred, with the measured justification above: a
memory-mapped replay backend (replay is not yet the bottleneck; the *store* is),
vectorised target construction, GPU non-blocking transfers, and reducing the
worker's ~400 MB private footprint. Throughput comparisons never changed
`num_simulations`, model dimensions, unroll, horizon or reward unless that
parameter was itself the axis of the table (§8).

Raw data for every table:
`data/reports/stage46_muzero_scaling_sequential.json`,
`data/reports/stage46_muzero_scaling_server.json`,
`data/reports/stage46_muzero_scaling_local.json`,
`data/reports/stage46_muzero_simulation_sweep.json`,
`data/reports/stage46_muzero_phase_profile_single.json`,
`data/reports/stage46_muzero_replay_benchmark.json`.
