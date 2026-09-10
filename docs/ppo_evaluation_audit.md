# PPO evaluation audit

Audit date: 2026-09-07. Frozen evaluation: validation split, 100 episodes,
seed 42, and 512 M5 decisions per episode. No training, reward, PPO, action,
execution, accounting, conversion, or hyperparameter behavior was changed.

The requested `runs/ppo_stable_seed1/checkpoints/best.pt` was absent. The
available `forexmind/best.pt` matches the requested run: `run_id=ppo_stable`,
training seed 1, categorical PPO, 503,808 environment steps, the same saved
selection score, and the exact expected action/execution counts. Its SHA-256 is
`b9c3d63ccc9b0c5deffd80a966b017593e14fa34e80dce2577c69394bc09b77b`.

## Finding

The old Sharpe near 11.415 is not a valid portfolio Sharpe. The evaluator
averages unrelated episode returns by relative timestep, which reduces measured
volatility, and then treats the resulting 512 values as one annualized history.

The pipeline is:

1. `EpisodeSampler.sample` samples instruments uniformly and then valid random
   M5 starts. Every episode resets its account to USD 10,000. Market windows can
   overlap.
2. `EvaluationRunner.run_episode` records rewards, mark-to-market equity,
   positions, executions, and `log(equity[t+1] / equity[t])`. Existing episode
   metrics are calculated independently.
3. `mean_log_return_series` stacks each instrument's episodes and executes
   `mean(axis=0)`. `aggregate_across_instruments` then averages the seven
   instrument series.
4. `compute_series_metrics` receives that 512-element synthetic vector,
   reconstructs equity and simple returns, and reports `n_periods=512`.

Therefore 100 x 512 = 51,200 real decisions become 512 relative-step averages.
Weights are equal by instrument, then equal among the episodes for an instrument;
they are not equal across all 100 episodes because instrument counts differ.
This is neither an average of episode Sharpes nor a chronological portfolio.

## Exact old Sharpe

For instrument `i`, episode `e`, and relative timestep `t`, define
`L[t] = mean_i(mean_e(log_return[i,e,t]))`. The simple return passed to Sharpe
is mathematically `exp(L[t]) - 1`.

| Input | Reproduced value |
|---|---:|
| Raw environment rewards | 51,200 |
| Sharpe observations | 512 |
| Mean simple return | 0.000001041863926658964 |
| Sample standard deviation, ddof=1 | 0.00002467227295964508 |
| `periods_per_year` | 73,082.65921532846 |
| `sqrt(periods_per_year)` | 270.33804618537965 |
| Risk-free rate | 0 |
| Old diagnostic Sharpe | 11.415869903218823 |
| Old diagnostic Sortino | 12.526562754723516 |
| Saved Sharpe-minus-drawdown score | 11.415361131760143 |

Sharpe is `mean / sample_std * sqrt(periods_per_year)`. Subtracting the old
maximum drawdown, 0.0005087714586796688, reproduces the saved selection score.
The arithmetic is correct for its input, but the input is not an investable
return path. Cross-sectional averaging reduces volatility and can materially
inflate the ratio.

Annualization divides mean observed M5 rows per instrument by the validation
split's calendar duration, using 365.25 days per year. The split is 2019-01-01
through 2022-01-01. Configured-order row counts are 220038, 221347, 219438,
215021, 219794, 220228, and 219220. Observed closures and gaps are included.
This is not a 12 x 24 x 365 assumption or an explicit FX holiday calendar, and
it counts bars rather than exactly the number of bar-to-bar transitions.

Annualized return uses geometric growth raised to `periods_per_year / n`.
Volatility, Sharpe, and Sortino use `sqrt(periods_per_year)`. Sortino retains
the existing zero-target conditional downside convention: RMS over negative
simple returns only. Short-episode annualized returns can be extreme; overflow
is now explicitly unavailable instead of serialized as a non-finite number.

## Corrected evaluation

Random, independently reset episodes do not form one capital path. Corrected
combined `n_periods`, `total_return`, Sharpe, and Sortino are therefore null,
with an explanation. The old curve remains under the explicit name
`cross_episode_mean_return_series`, with `is_portfolio=false` and diagnostic
metric names.

| Metric | Old | Corrected |
|---|---:|---:|
| `n_periods` | 512 synthetic | Unavailable; 512 per episode |
| Total return | 0.053342% synthetic | Unavailable |
| Mean episode return | Not reported | 0.055082% |
| Median episode return | Not reported | 0.006950% |
| Profitable episodes | Not reported | 52% |
| Sharpe | 11.415870 | Unavailable |
| Sortino | 12.526563 | Unavailable |
| Turnover | 145.801390, ambiguous | Explicit total 145.801390 |
| Mean turnover per episode | Not reported | 1.458014 |
| Actual executions | 483 | 483 |

Episode return sample standard deviation is 0.478185%; worst is -1.561418%
and best is 2.148538%. The p10, p25, p75, and p90 returns are -0.413252%,
-0.177480%, 0.217022%, and 0.629279%. The JSON contains mean, median, sample
standard deviation, min, max, p10, p25, p75, and p90 for every required
per-episode statistic.

## Turnover, actions, and censoring

PPO decision turnover is:

`abs(executed_units) * execution_mid * quote_to_account_factor / pre_decision_equity`

This is one-way account-notional turnover, summed over decisions. It is not
divided by two or normalized by the episode's step count. The old PPO evaluator
first calculated pooled executed notional divided by pooled initial capital,
then overwrote that field with the sum of step-normalized turnover across all
episodes. This explains the ambiguous 145.801390.

Corrected total / mean / median / min / max episode turnover is
145.801390 / 1.458014 / 1.0 / 0.25 / 6.253504. Forced turnover is separate and
zero.

Raw invariants all pass: 51,200 decisions; 50,717 HOLD actions; zero HOLD
executions or unit changes; 483 executions; 483 position changes including
entries; 483 trade-log records; two sign reversals; zero forced executions.
The generic legacy `trade_statistics.n_position_changes` omits initial entry by
differencing only post-step positions. Corrected categorical counts use raw
before/after units and cross-check the trade log.

All 100 episodes finish with an open long position. `censored_positions=100`
means their final holding spells reach the sampling boundary without an
observed exit. Same-direction resizing retains the holding clock. Mean/median
position-spell duration, including censored spells, is 204.664 / 10 decisions.
Mean/median/max HOLD streak is 147.006 / 51 / 511.

Final exposure mean/median/range is 0.716875 / 0.501319 / 0.248778-1.005864.
Nearest-level descriptive bins are LONG_25=4, LONG_50=48, LONG_75=5,
LONG_100=43. Exact fractions remain in per-episode output.

Mean/median terminal equity is USD 10,005.508215 / 10,000.694981. Mean/median
unrealized PnL is USD 5.558460 / 0.862041, ranging from -156.141762 to
214.853839. Terminal equity equals balance plus marked unrealized PnL and
includes costs already incurred. Hypothetical closing costs are excluded.
Episode-end closure was disabled and no position was forced closed.

## PPO behavior and matched baselines

The first position is long in 97 episodes and short in three; none remain flat.
Entry delay, defined as zero-based decisions before the first position, has
mean 29.47, median 4, and maximum 245. Thirty episodes enter immediately and
53 enter within five decisions. Executions have mean 4.83, median 2, and range
1-28; 29 episodes execute once. PPO is strongly long-biased with extended HOLD
periods, but it is not immediate-entry buy-and-hold in every episode. These
results do not establish reliable timing skill.

Each baseline uses the exact same 100 specifications, costs, initial equity,
and simulator. Exposure baselines enter on decision zero and then choose true
HOLD, preserving exact units. FLAT chooses HOLD from its initial flat state.
Pairing uses the full episode specification. Returns and advantages below are
percentage points.

| Baseline | Mean baseline return | Mean PPO advantage | Median advantage | PPO win rate |
|---|---:|---:|---:|---:|
| FLAT | 0.000000% | 0.055082% | 0.006950% | 52% |
| LONG_25 | 0.010055% | 0.045027% | 0.002791% | 52% |
| LONG_50 | 0.020109% | 0.034973% | 0.000423% | 51% |
| LONG_75 | 0.030164% | 0.024918% | 0.018271% | 56% |
| LONG_100 | 0.040219% | 0.014863% | 0.006828% | 53% |
| SHORT_25 | -0.015152% | 0.070234% | 0.000500% | 50% |
| SHORT_50 | -0.030303% | 0.085386% | 0.006938% | 51% |
| SHORT_75 | -0.045455% | 0.100537% | 0.013701% | 51% |
| SHORT_100 | -0.060607% | 0.115689% | 0.020464% | 51% |

PPO's win rates of 50-56% and small median advantages do not demonstrate a
robust timing advantage. The checkpoint was selected on this same validation
configuration, so the comparisons are selection-biased.

## Per instrument

| Instrument | Episodes | Mean return | Median return | Profitable | Mean turnover | Mean executions |
|---|---:|---:|---:|---:|---:|---:|
| EURUSD | 12 | 0.005178% | 0.001462% | 50.00% | 0.624773 | 1.2500 |
| GBPUSD | 11 | 0.025470% | -0.003040% | 45.45% | 3.136850 | 11.1818 |
| USDJPY | 12 | 0.381812% | 0.007550% | 50.00% | 1.521329 | 5.4167 |
| USDCHF | 20 | 0.037002% | 0.018588% | 60.00% | 1.900504 | 6.6000 |
| AUDUSD | 13 | -0.046330% | 0.041819% | 61.54% | 1.615503 | 5.0769 |
| USDCAD | 20 | 0.087902% | -0.005921% | 50.00% | 1.025034 | 2.4000 |
| NZDUSD | 12 | -0.109303% | -0.026158% | 41.67% | 0.502544 | 2.8333 |

USDJPY supplies 83.18% of the sum of episode profits. Its two largest gains,
2.148538% and 1.631283%, come from overlapping windows starting 2020-03-17
15:35 and 2020-03-18 03:40. Its median is nearly zero and six of 12 samples
profit. Excluding USDJPY, mean episode return is only 0.010528%. The headline
mean is dominated by a few correlated winners.

## Implementation and verification

Changed files:

- `forexmind/evaluation/runner.py`: optional terminal/account-state capture.
- `forexmind/evaluation/sampled.py`: independent distributions, chronology
  guard, diagnostic curve labeling, paired comparisons, and true-HOLD agents.
- `forexmind/training/evaluator.py`: separate corrected reporting API.
- `forexmind/training/evaluate_checkpoint.py`: corrected standalone output.
- `tools/audit_ppo_evaluation.py`: reproducible matched audit, raw archives,
  CSVs, and resumable execution.
- `tests/test_sampled_evaluation.py`: evaluation regression coverage.
- `README.md` and this report: corrected semantics and usage.

Verification: 427 passed, 1 skipped in the full suite; 37 targeted tests passed
after the final numerical reporting fix. Coverage includes averaging inflation,
episode independence, turnover totals versus means, annualization, chronology
restrictions, nine baseline HOLD paths, terminal marked equity, raw-accounting
failures, missing action records, overlap disclosure, strict JSON output, and
separation from training selection. The public checkpoint CLI passed a
two-episode smoke check. Lint and changed-module type checks passed.

Training's historical selection score remains unchanged as requested and still
uses the flawed aggregation. Standalone checkpoint and Phase-2 benchmark
outputs now report independent episode distributions and retain the old curve
only under its explicit diagnostic name. Changing training selection is
separate work. No single strategy Sharpe is available until evaluation produces
a real, chronological, non-overlapping account path.

Artifacts are in `data/reports/ppo_evaluation_audit/`: `metadata.json`,
`PPO.json`, one report and compressed raw JSONL archive per policy,
`episodes.csv`, `paired_summary.csv`, `paired_differences.csv`,
`per_instrument.csv`, `action_distribution.csv`, `before_after.csv`, and
`legacy_reproduction.json`.
