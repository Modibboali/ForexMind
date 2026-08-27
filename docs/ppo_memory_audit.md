# PPO Memory Scaling & Dataset Duplication Audit

## Current PPO memory behavior (Kaggle: 224 vCPU, ~330 GB RAM)

PPO is fast (>1000% CPU, ~1.5 min/run) but RAM approached the ~330 GB ceiling.
Measured (local 8-core / 17 GB machine, same code path, real processed data):

```text
single worker, parquet backend, all 7 instruments loaded:
    dataset (parquet -> pandas M1+M5)   +2,947 MB   (98% of worker RSS)
    EnvWorker + envs + builders         ~   0 MB   (lazy)
    first episode stepping              + ~34 MB   (timeline precompute)

per-worker full footprint (all instruments sampled): ~2.95-3.3 GB
```

## Primary memory bottleneck

**Every spawned worker independently materialises the full 7-instrument M1+M5
parquet dataset into its own private pandas frames (~2.95 GB).** With
`spawn`, a worker cannot inherit the parent's memory, so 102 workers x ~3 GB
~= **~300 GB** - which matches the observed ~330 GB ceiling. The trainer
(parent) adds another ~3 GB copy.

The raw parquet itself is only ~800 MB on disk (M1 is ~96 MB/instrument); the
inflation to ~3 GB in RAM is pandas float64/datetime64 materialisation, and it
is duplicated once per worker.

Quantified contributions (per worker):

```text
M1  ~48.6M rows x 7 instruments (datetime64 + 4x float64)  ~2.4 GB
M5  ~8.9M  rows x 7 instruments (+ n_observations, is_complete) ~0.5 GB
environment timelines (per touched instrument)             ~0.03 GB
PPO rollout buffer (4096 steps x ~3 KB)                    ~0.01 GB
IPC copies (pickled Transition batches per round)          negligible
```

## Secondary memory bottlenecks

1. The **trainer (parent)** also loads the full dataset (~3 GB) - fixed by the
   same shared store.
2. Per-environment timeline arrays (exec indexes, minutes-since-last, weekend
   gaps) - ~30 MB per touched instrument per worker; grows as more instruments
   are sampled. Small relative to the dataset but not free.
3. PPO rollout storage: analytically ~2.8 KB/transition (obs float32[351] +
   next_obs + scalars); 4096-transition rollout + stacked update tensors ~
   12-25 MB - **negligible**.

## Fix: read-only shared memory-mapped dataset (`dataset_backend: mmap`)

The preferred long-term design (§9) is implemented:

```text
processed market data
      ->  tools.build_shared_dataset  (once, per-column .npy files)
      ->  workers open the SAME files via np.load(mmap_mode='r')
      ->  zero-copy pandas frames via pd.DataFrame({col: mmap}, copy=False)
      ->  OS shares the physical pages across all workers
```

* `forexmind/training/dataset_mmap.py`: `build_shared_store`,
  `open_instrument_data`, `make_mmap_dataset`, `resolve_dataset`.
* `tools/build_shared_dataset.py`: one-time builder (write 2.4 GB of .npy).
* `configs/ppo_stable.yaml`: `compute.dataset_backend: mmap` (+
  `num_workers: auto`, `memory_limit_fraction: 0.70`).
* `ComputeConfig.dataset_backend`: `auto` (use store if built) | `parquet` |
  `mmap`.
* `base_config` is unchanged; `spawn` stays (portable, no fork).

Values/dtypes are **bit-identical** to `pd.read_parquet` (M1 timestamps are
`datetime64[us]`, M5 `datetime64[ns]`), verified in
`tests/test_dataset_mmap.py`, and the environment produces identical
observations/rewards/accounting for identical seeds/action sequences.

### Measured local win (real data, 1-2 workers, 17 GB machine)

```text
                        parquet        mmap
1 worker  tree RSS      3,700 MB       1,317 MB
          worker USS      1,665 MB       276 MB
2 workers tree RSS      5,466 MB       1,995 MB
          worker USS agg 2,487 MB       500 MB
          steps/s           486           702
```

Worker **USS (private memory) drops ~5x** with mmap; throughput does not drop
(it improved locally, within noise). Scaling to ~100 workers: parquet
approaches 120-300 GB private RAM (the observed ceiling); mmap stays at roughly
one shared dataset copy + ~0.25-0.5 GB private per worker.

## Memory diagnostics added

`runtime_diagnostics.py`: `process_rss_mb`, `process_uss_mb`, `memory_report`
(trainer RSS, worker RSS min/median/mean/p90/max/aggregate, worker USS
aggregate/median, process-tree RSS), `print_memory_report`.

The trainer now prints a `STARTUP MEMORY` block and a `FINAL MEMORY` block and
writes `trainer_rss_mb`, `worker_rss_aggregate_mb`, `worker_uss_aggregate_mb`,
`process_tree_rss_mb`, `dataset_backend` to `training_summary.json`.

## Worker selection & safety

* `compute.num_workers: auto` (new): estimates a RAM-safe worker count from
  detected RAM, the dataset backend's per-worker footprint, a 2 GB trainer
  reserve, and a cap. On the local 17 GB box it resolved 24 workers for mmap.
* `compute.memory_limit_fraction` (new): when set, prints the estimated
  process-tree memory and **warns** if it exceeds `fraction * total RAM`
  (e.g. 0.70), suggesting fewer workers or `dataset_backend: mmap`.
* `memory_planning.py` holds the pure helpers.

## Benchmark: tools/benchmark_ppo_memory.py

```bash
python -m tools.build_shared_dataset                     # once
python -m tools.benchmark_ppo_memory \
    --workers 1,2,4,8,16,32,64,96,128,160,192,204 \
    --dataset-backend mmap --json data/reports/ppo_memory_bench.json
```

Reports trainer RSS, per-worker RSS min/median/mean/p90/max/aggregate,
worker USS, process-tree RSS, steps/s, the analytical rollout memory, and a
recommended worker count. Stages: A = after worker startup + first episode;
B = after a representative rollout (the tool reports both). Full-PPO stage C
is intentionally NOT part of the memory benchmark (see §4/§16); it is covered
by a short PPO training run instead.

Local smoke (1-2 workers) passed for both backends; the JSON rows are saved
under `data/reports/bench_local_mem_{parquet,mmap}.json`.

## Practical worker count

Do NOT assume 204 workers is desirable. The Kaggle benchmark determines the
throughput knee; the recommendation is a configuration that keeps process-tree
RAM well under the limit (60-70% is safer than 95%). With mmap, RAM is no
longer the binding constraint, so the knee is CPU/collector throughput. Run
`benchmark_ppo_memory` on Kaggle before choosing a final explicit
`num_workers`.

## Correctness regression

`tests/test_dataset_mmap.py` verifies bit-exact column roundtrip and that a
worker stepping with the mmap backend produces identical observations,
rewards, actions, and done flags to the parquet backend (same seeds/episodes).
Full suite (`python -m pytest`), `ruff`, and `mypy` are green.

## Commands (Kaggle)

```bash
python -m tools.build_shared_dataset                        # 1x, ~2.4 GB .npy
python -m tools.benchmark_ppo_memory --workers 1,2,4,8,16,32,64,96,128,160,192,204 \
    --dataset-backend mmap --json data/reports/ppo_memory_bench.json
python -m forexmind.training.train_ppo --config configs/ppo_stable.yaml --seeds 1
```

Local correctness (1-2 workers, small run):
```bash
python -m tools.benchmark_ppo_memory --workers 1,2 --measured-steps 1024
```
