# Stage 3.5 — valid PPO checkpoint selection and chronological validation

Audit date: 2026-09-08. This stage changed evaluation and checkpoint-selection
semantics only. Reward, PPO loss, GAE, categorical actions, observations,
execution, accounting, conversion, spreads, and training hyperparameters remain
unchanged.

## Old bug and corrected selector

Training previously selected `best.pt` from Sharpe minus drawdown calculated on
a 512-point relative-timestep average of independently reset episodes. That
series is not an investable account path. Cross-episode averaging suppresses
volatility, so its annualized Sharpe is not valid for checkpoint selection.
The exact path was `BaseTrainer._evaluate_validation` to
`PolicyEvaluator.evaluate`, then `mean_log_return_series` /
`aggregate_across_instruments`, `compute_series_metrics`, and finally
`selection_score("sharpe_drawdown")`.

Checkpoint selection now uses:

```text
selection_score = mean(episode cumulative log return)
```

Every episode is measured independently. Training uses one seed, count, and
saved set of balanced episode specifications for all checkpoint candidates.
The selection sampler rejects overlap in each instrument across both context
and decision windows. The historical averaged series remains available only as
`cross_episode_mean_return_series`, with `is_portfolio=false` and
`diagnostic_only=true`. Sampled reports set portfolio Sharpe, Sortino, and
Calmar to null, and the selector rejects the legacy metric names.

New checkpoints store the selection schema and metric, current and best scores,
best step, validation seed/count/specifications, episode summaries, and
selection history. Resume restores this state. For an old checkpoint,
`best_validation_score` becomes `legacy_selection_score`; the new best score is
unavailable until a corrected validation runs.

## Re-evaluated checkpoint curve

The deleted working-tree checkpoints were read from the current Git commit into
a temporary directory. The deleted files were not restored or modified. All
five files from `stage34_categorical_verified_seed42` were evaluated on the
same 100 balanced, non-overlapping validation episodes with seed 42 and 512
decisions per episode.

| Checkpoint | Step | Mean log return | Mean return | Median return | Profitable | p10 | p90 | Mean turnover | Mean executions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `final.pt` / `step_100352.pt` | 100,352 | +0.000156% | +0.000742% | -0.001128% | 50% | -0.385143% | +0.347475% | 5.942106 | 23.02 |
| `best.pt` / `step_51200.pt` | 51,200 | -0.036355% | -0.035667% | 0.000000% | 46% | -0.378391% | +0.303915% | 1.942989 | 7.70 |
| `step_0.pt` | 0 | -0.340449% | -0.337633% | -0.283663% | 29% | -1.131120% | +0.370183% | 37.312456 | 85.22 |

The existing legacy `best.pt` is not best under the corrected metric. The
corrected winner is the step-100,352 actor stored in both `final.pt` and
`step_100352.pt`. The two alias pairs reproduce exactly.

The corrected winner chooses HOLD for 95.503906% of decisions, FLAT for 0%,
long actions for 0.867188%, and short actions for 3.628906%. All sampled
checkpoint reports have zero overlapping episode pairs, zero overlap fraction,
and zero largest overlap duration.

## Sampled per-instrument results for the corrected winner

| Instrument | Episodes | Mean return | Median return | Profitable | Mean turnover | Mean executions |
|---|---:|---:|---:|---:|---:|---:|
| EURUSD | 15 | -0.011728% | -0.063058% | 33.33% | 2.017145 | 7.87 |
| GBPUSD | 15 | +0.164770% | +0.151394% | 60.00% | 2.335103 | 9.20 |
| USDJPY | 14 | +0.012487% | +0.008130% | 57.14% | 4.857963 | 17.71 |
| USDCHF | 14 | -0.168644% | -0.149559% | 35.71% | 16.501994 | 61.43 |
| AUDUSD | 14 | -0.036154% | -0.031102% | 42.86% | 2.180580 | 8.93 |
| USDCAD | 14 | +0.013301% | +0.028798% | 64.29% | 5.214545 | 19.86 |
| NZDUSD | 14 | +0.020337% | +0.067863% | 57.14% | 9.025409 | 38.21 |

USDJPY contributes 11.2066% of total positive episode profit. Overall mean
episode return is +0.000742%, while mean return excluding USDJPY is -0.001170%.
The sign still depends on including USDJPY, although USDJPY does not dominate
positive profits as strongly as it did in the previous overlapping audit.

## Matched static-exposure baselines

Each baseline uses the exact same specifications, costs, simulator, and initial
capital. It enters on decision zero and then uses true HOLD. Values are
percentage returns or percentage-point PPO advantages.

| Baseline | Mean baseline return | Mean PPO advantage | Median advantage | PPO win rate |
|---|---:|---:|---:|---:|
| FLAT | 0.000000% | +0.000742% | -0.001128% | 50% |
| LONG_25 | -0.026910% | +0.027652% | +0.029958% | 60% |
| LONG_50 | -0.053821% | +0.054563% | +0.001981% | 52% |
| LONG_75 | -0.080731% | +0.081473% | +0.054982% | 59% |
| LONG_100 | -0.107641% | +0.108383% | +0.079928% | 62% |
| SHORT_25 | +0.021739% | -0.020997% | -0.061977% | 43% |
| SHORT_50 | +0.043477% | -0.042735% | -0.109359% | 39% |
| SHORT_75 | +0.065216% | -0.064474% | -0.159722% | 36% |
| SHORT_100 | +0.086954% | -0.086212% | -0.202539% | 37% |

PPO beats the long baselines in this bearish sampled set but loses to every
short baseline. Its near-zero mean and 50% FLAT win rate do not establish timing
skill.

## Chronological validation

The corrected winner was run once from the first usable validation observation
to the end of the split for each instrument. Each instrument has one account
lifecycle, strictly increasing unique timestamps, no overlap, no resets, and an
explicit periods-per-year value derived from decisions divided by elapsed
calendar years. Independent instrument accounts are not averaged into a
portfolio.

| Instrument | Periods | Total return | Annual return | Sharpe | Sortino | Max DD | Turnover | Executions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EURUSD | 219,973 | -5.8551% | -1.9929% | -0.6749 | -0.6600 | 6.8994% | 527.781 | 2,307 |
| GBPUSD | 221,282 | +2.9726% | +0.9821% | +0.2420 | +0.2411 | 7.6750% | 175.180 | 1,122 |
| USDJPY | 219,373 | -7.9624% | -2.7303% | -0.6517 | -0.6625 | 12.7176% | 2,257.594 | 8,095 |
| USDCHF | 214,956 | -62.3871% | -27.8366% | -6.0329 | -5.9684 | 62.6774% | 9,717.467 | 37,450 |
| AUDUSD | 219,729 | -5.6811% | -1.9324% | -0.3551 | -0.3457 | 16.8560% | 258.757 | 1,527 |
| USDCAD | 220,163 | -15.6543% | -5.5217% | -1.2990 | -1.2868 | 17.7785% | 2,585.697 | 9,566 |
| NZDUSD | 219,155 | -51.3744% | -21.3809% | -2.7593 | -2.7303 | 52.6185% | 4,308.441 | 21,598 |

Only GBPUSD is positive, and its Sharpe is 0.242. Six instruments lose more
than 5%, two lose more than 50%, and turnover/execution counts are extreme on
USDCHF, USDCAD, and NZDUSD. These genuine chronological results contradict any
claim that the old Sharpe near 11 represented robust strategy performance.

## Remaining concerns and readiness

- This checkpoint curve covers one training seed and a short 100,352-step run.
- The corrected winner was chosen on validation. Its untouched test split has
  not been run and the validation results must not be presented as test results.
- Mean sampled return is close to zero, median sampled return is negative, and
  the result changes sign when USDJPY is excluded.
- Every static short baseline beats PPO on mean return in the fixed sample.
- Six of seven full chronological paths lose money, with severe drawdown and
  turnover on several instruments.
- Chronological metrics are valid per instrument. A combined multi-instrument
  portfolio remains unavailable until one simulator owns one shared account
  across instruments.

## Implementation and verification

Primary Stage 3.5 files:

- `forexmind/episodes/sampler.py`: deterministic balanced non-overlapping specs.
- `forexmind/evaluation/sampled.py`: episode distributions, overlap metadata,
  and guarded diagnostic curve.
- `forexmind/evaluation/chronological.py`: one continuous account per instrument.
- `forexmind/training/evaluator.py`: valid selector and matched baselines.
- `forexmind/training/trainer.py`: fixed specs, best-checkpoint selection,
  metadata, resume isolation, USDJPY reporting, and baseline caching.
- `forexmind/training/checkpoint.py` and `config.py`: new schema/default metric.
- `forexmind/training/evaluate_checkpoint.py`, `benchmark.py`, and supporting
  tools: corrected public reporting and legacy-config compatibility.
- `tools/evaluate_checkpoint_curve.py`: fixed-spec multi-checkpoint ranking.
- `tools/evaluate_chronological_checkpoint.py`: atomic resumable chronological
  evaluation.
- `tests/test_stage35_validation.py`, `test_sampled_evaluation.py`, and updated
  training tests: selection, determinism, legacy resume, Sharpe guards, overlap,
  ranking, metadata, baseline cache, and chronological integrity.
- `README.md` and this report: corrected public semantics and commands.

The evaluation and selection infrastructure is ready for future runs. This
100,352-step PPO model is not ready to freeze as a credible model-free
performance baseline: sampled performance is essentially flat, static short
exposures beat it, and six of seven chronological paths lose money. ForexMind
should train a fresh categorical PPO run under the corrected selector, repeat
across multiple seeds, and evaluate the untouched test split before making a
fair frozen PPO-versus-MuZero performance comparison. MuZero implementation can
proceed against the now-valid evaluation contract, but performance claims
should wait for that corrected PPO baseline.

Artifacts:

- `data/reports/stage35_checkpoint_curve/checkpoint_curve.json`
- `data/reports/stage35_checkpoint_curve/checkpoint_curve.csv`
- `data/reports/stage35_checkpoint_curve/validation_episode_specs.json`
- `data/reports/stage35_chronological/chronological_validation.json`
- one atomic chronological JSON report per instrument in the same directory.
