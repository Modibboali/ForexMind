# SAC Multiprocessing Architecture Audit

## Known From Code

Execution path:

```text
python -m forexmind.training.train_sac
-> forexmind.training.cli.run_multiseed()
-> forexmind.training.cli.load_config()
-> ExperimentConfig.compute.num_workers
-> SACTrainer
-> BaseTrainer._build_collector()
-> ProcessCollector(..., num_workers=cfg.compute.num_workers)
-> multiprocessing.get_context("spawn").Process(...)
-> _worker_process_main()
-> EnvWorker(...)
```

`--workers` is read by the shared training CLI and applied to
`config.compute.num_workers`. `ProcessCollector.__init__()` then stores
`self.num_workers = max(1, num_workers)` and starts one
`multiprocessing.Process` for each `wid in range(self.num_workers)`.

This only creates multiple processes when `compute.collect_backend == "process"`.
The `"sync"` backend intentionally uses one in-process `EnvWorker`.

Process backend mechanism:

```text
backend: multiprocessing with spawn
creation site: forexmind.training.collector.ProcessCollector.__init__
entry point: forexmind.training.collector._worker_process_main
shutdown: q_in.put(None) -> worker break -> parent join(timeout=30)
```

Each worker independently rebuilds the split config, loads processed parquet,
creates its policy network, creates an `EnvWorker`, receives commands on its
input queue, and sends transition batches on its output queue.

`env.step()` execution:

```text
ProcessCollector.collect()
-> parent sends ("collect", n, random_action)
-> _worker_process_main() receives it in the child process
-> [worker.step(...) for _ in range(n)]
-> EnvWorker.step()
-> ForexEnvironment.step(action_f)
```

So for the process backend, training `env.step()` occurs inside the worker
process. For the sync backend, it occurs in the parent by design. Other
`env.step()` calls exist in tests, smoke tools, and evaluation runner; they are
not the SAC process collector.

Synchronization model:

```text
for each worker:
    q_in.put(("collect", per_worker_steps, random_action))

for each worker output queue in fixed order:
    batch = q_out.get()
    transitions.extend(batch)
```

This is a synchronous collection-round barrier. The parent waits for every
worker before replay insertion or learner updates. Because output queues are
read in fixed order, the parent can block on an earlier slow worker while later
workers already have results ready.

No `ProcessPoolExecutor`, `multiprocessing.Pool`, or `executor.map()` path was
found in the SAC collector.

Queue/IPC mechanism:

```text
trainer q_in[i] -> worker i
worker i q_out[i] -> trainer
```

The worker sends one Python list of `Transition` dataclass objects per collect
round, not one `Queue.put()` per environment step. Each transition still carries
Python fields plus NumPy `obs` and `next_obs` arrays, so pickling that batch may
be material at high worker counts.

Replay insertion:

```text
SACTrainer._consume_transitions()
-> for each Transition:
       replay.push(...)
-> if warmup and replay size allow:
       run learner updates
```

Replay insertion is one transition at a time in the parent process.

CPU/thread configuration:

```text
trainer: torch.set_num_threads(config.compute.torch_threads)
trainer env defaults: OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS
workers: torch.set_num_threads(1)
```

No source-level CPU affinity restriction was found. No `taskset`,
`os.sched_setaffinity`, or equivalent repo launcher restriction exists.

Current `201%` CPU figure:

The repository originally contained no code that computed/printed CPU percent, so
the `201%` observed on Kaggle was not explained by the repo. The trainer now
samples the process tree itself (see below). Local evidence strongly indicates
`201%` is the **trainer (SAC learner) process** running `torch_threads=2` (two
fully-busy threads), with workers largely idle behind the sync barrier.

## Instrumentation Added

`ProcessCollector` now exposes and logs:

```text
configured worker count
worker PIDs
alive worker count
per-worker start/stop log lines
```

`Transition` now carries diagnostic-only provenance:

```text
worker_id
worker_pid
```

`forexmind.training.runtime_diagnostics` reports:

```text
logical CPUs
process PID
worker PIDs
alive workers
trainer CPU
worker CPU aggregate
process-tree CPU
CPU affinity
PyTorch threads
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
```

`BaseTrainer` now:

```text
tracks producer worker ids per transition (workers_producing_transitions)
starts a ProcessTreeCpuSampler at train() start
prints an interval-average CPU line at each logging interval:
  [cpu] interval_avg trainer=..% worker_agg=..% tree=..% live_workers=.. producers=..
prints a FINAL PROCESS TREE block before closing workers in finalize()
persists trainer_cpu_percent / worker_cpu_percent / process_tree_cpu_percent /
  effective_cores_utilized / workers_producing_transitions in training_summary.json
```

CPU values use psutil window semantics: 100.0 means one fully-utilized core, so
`201.0` means about two effective cores. psutil is a `train` extra dependency.

## Local Findings (workers = 1, 2 on an 8-logical-core machine)

Tiny local runs (real processed data, random actions, horizon 512) verify the
tooling works end to end and already expose the likely scaling pattern. These
are **not** throughput predictions for Kaggle; they identify *where* the
bottleneck sits.

```text
A_env_only   workers=1 -> ~144 steps/s   (worker CPU ~95%, trainer ~0%)
             workers=2 -> ~338 steps/s   (worker CPU ~94% each)
B_collection workers=2 -> ~145 steps/s   (IPC + replay.push() added)
C_full_sac   workers=2 -> ~14  steps/s   (SAC learner added)
```

Full-SAC process tree during the measured window:

```text
trainer CPU: 175%    worker CPU aggregate: 26%   tree: 202%
```

And a complete smoke `train()` run (2048 steps, 2 workers) showed, over the
whole run:

```text
trainer_cpu_percent: 97.5   worker_cpu_percent: 0.0   tree: 97.5
workers_producing_transitions: 2   workers_alive_at_finalize: 2
```

The pattern: the SAC learner (with `gradient_updates_per_step=1`, i.e. one
update per env step) saturates roughly `torch_threads` cores, while workers are
starved by the synchronous collection barrier and show near-zero aggregate CPU
over a full run. This reproduces the Kaggle signature (trainer ~201%, flat
throughput regardless of worker count) on a small machine and suggests the
learner, not worker parallelism, is the throughput cap.

## Suspected Bottlenecks

Ranked after local evidence (still to be confirmed by the Kaggle sweep):

1. **SAC learner is the wall-time bottleneck.** `gradient_updates_per_step=1`
   means one full SAC update per env step; with `collect_batch=2048` and 204
   workers that is 2048 updates per round while each worker only produced ~10
   steps. Trainer CPU saturates ~2 cores while workers idle.
2. Collection-round barrier + fixed output-queue order: parent waits for every
   worker (in fixed order) before replay insertion / learner updates.
3. Tiny per-worker round size at high worker counts (2048/204 = 10 steps per
   worker per round) makes queue round-trip latency dominate.
4. Python pickle overhead for `Transition` batches containing NumPy arrays.
5. Parent-side one-by-one replay insertion.
6. Dataset duplication and memory pressure with hundreds of spawned workers.
7. External CPU affinity or notebook process accounting hiding worker CPU (must
   be verified on Kaggle).

## Requires Kaggle Benchmark

Only Kaggle can determine the practical worker count on the 224-vCPU machine.

Runtime sanity check:

```bash
python -m tools.inspect_runtime --config configs/sac_cpu.yaml --workers 204 \
  --backend process --probe-steps 204
```

Separated throughput benchmarks:

```bash
python -m tools.benchmark_env_workers \
  --workers 1,2,4,8,16,32,64,128,192,204

python -m tools.benchmark_training \
  --workers 1,2,4,8,16,32,64,128,192,204 \
  --mode all \
  --measured-steps 100000 \
  --warmup-steps 10000 \
  --horizon 512
```

Interpretation:

```text
A_env_only    raw EnvWorker.step() scalability
B_collection  EnvWorker.step() + IPC + replay.push()
C_full_sac    EnvWorker.step() + IPC + replay.push() + SAC updates
```

Each mode prints the PROCESS TREE block (trainer CPU, worker CPU aggregate,
process-tree CPU, live workers, producers) and writes JSON rows. Do not start
another long SAC run until these rows show actual worker count, process-tree
CPU, worker CPU aggregate, worker producers, and throughput by mode.

## Recommended Next Changes

If A scales but B does not, focus on transition serialization, queue design, and
batch replay insertion.

If B scales but C does not (the locally observed case), focus on the learner:

1. Benchmark `gradient_updates_per_step` below 1 (e.g. 0.25/0.5) — a
   hyperparameter, not an SAC semantic change.
2. Benchmark larger `collect_batch` so each worker does more steps per round
   (amortize the sync barrier) — e.g. `collect_batch = 2048 * workers`.
3. Measure learner cost with `--torch-threads` 2 vs 4 on Kaggle; more threads
   raise trainer CPU but may raise per-update throughput.
4. If worker CPU aggregate stays near zero, an asynchronous collector
   (workers continuously fill a shared experience queue while the learner
   drains it) is the architectural fix — implement only after the Kaggle
   modes confirm the learner is the cap.
5. Consider a GPU learner on Kaggle (learner_device=cuda) — removes the learner
   from the CPU pool entirely.

If A does not scale, focus on environment cost, Decimal profiling, dataset
loading/memory pressure, worker affinity, and process startup/runtime limits.

If worker CPU aggregate stays near zero *in env-only mode*, debug process
creation, Kaggle resource limits, affinity, and worker crashes before changing
SAC or environment semantics.

Do not modify reward, action space, observation, environment accounting,
execution model, split policy, or SAC objective unless a concrete implementation
bug is found (none found in this audit).
