# ForexMind — Phases 1, 2 & 3

A research-grade, **deterministic** Forex market/data environment and learning
protocol for reinforcement-learning research.

- **Phase 1** builds the reliable foundation later agents (SAC, PPO, Dreamer,
  MuZero, Stochastic MuZero, offline RL) will consume: a canonical M1 data
  layer, dataset validation, deterministic M1→M5 resampling, an
  instrument-aware dataset, an execution-cost model, portfolio / margin
  accounting, and a Gymnasium-style environment.
- **Phase 2** adds the leakage-free learning/evaluation layer: temporal
  train/validation/test splits, a causal observation encoder, a deterministic
  episode sampler, baseline strategies, and a full evaluation framework.
- **Phase 3** adds model-free RL training infrastructure: SAC (twin critics +
  auto entropy temperature) and PPO on the MLP(351) observation baseline,
  multi-CPU experience collection, checkpoints with resume, validation-based
  checkpoint selection, and automatic final test-split comparison tables
  against the seven baselines.

> Phases 1-2 deliberately contain **no** neural networks, no RL algorithms, no
> MCTS, no technical-indicator feature libraries, no live trading. They build
> the deterministic simulator and evaluation protocol that later phases reuse.
> Phase 3 adds model-free RL only — no MuZero/MCTS/Dreamer/world models.

---

## 1. Data & the OHLC-only limitation

The supplied raw data comes from MetaTrader and contains **only**:

```
timestamp, open, high, low, close
```

There is **no historical spread, bid, ask, or tick data** in the dataset. The
simulator therefore distinguishes strictly between:

- **Observed market data** — OHLC only.
- **Configured execution assumptions** — `spread`, `slippage`, `commission`.

These execution-cost parameters are **model parameters, not historical
observations**. We never infer a spread from OHLC. The execution-cost model is
isolated in `forexmind/environment/costs.py` so that real historical
bid/ask/tick data can be introduced later without redesigning the rest of the
system.

### Instruments

Phase 1 supports the seven major pairs present in the data:
`EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD`. Instrument identity
is explicit (`MarketDataset`), and new instruments can be added without
modifying core simulator logic.

### Temporal resolution

| Level            | Source                            |
| ---------------- | --------------------------------- |
| Raw dataset      | M1 (1-minute OHLC)                |
| Decision interval| M5 (derived from M1)              |

M5 data is **built from M1** — no separate M5 dataset is required. The M1 data
remains the source of truth.

---

## 2. Timestamp rules

- All timestamps are parsed and stored as **naive `datetime64[ns]`**.
- If a source is timezone-aware it is normalised to UTC first.
- If the timezone is **ambiguous** (MetaTrader CSV exports are broker server
  time, typically UTC+2/+3), we **do not silently guess**: timestamps stay
  naive and the dataset manifest records `timezone: null` (unknown/server
  time). Session labels (`ASIA`, `LONDON`, …) are computed from the raw
  wall-clock with a configurable `utc_offset_hours` (default 0).
- **No candles are fabricated across missing periods.** If data ends Friday
  and resumes Sunday, there are no fake Saturday/Sunday candles — the gap
  remains detectable (`weekend_gap`).

Gap classification (`GapType`):

| Type          | Meaning                                          |
| ------------- | ------------------------------------------------ |
| `NORMAL`      | exactly the expected 1-minute interval           |
| `SHORT_GAP`   | a few missing bars (thin session / dropout)      |
| `WEEKEND_GAP` | Friday/Saturday → Sunday/Monday market closure   |
| `LARGE_GAP`   | a long unexplained gap (≥ 60 min, e.g. holiday)  |

Sunday → Monday midnight is **continuous trading**, never a weekend gap.

---

## 3. M5 aggregation rules

For each aligned 5-minute bucket:

```
open  = first M1 open
high  = max M1 high
low   = min M1 low
close = last M1 close
```

- The M5 bar is labelled with the **bucket start** (e.g. `10:00` covers
  `10:00..10:04`).
- Completeness is policy-driven:
  - `STRICT` (default): an M5 bar is emitted only when the bucket holds exactly
    5 **contiguous** 1-minute observations. Gaps inside a bucket suppress the
    bar; large temporal gaps are never bridged into a fake candle.
  - `PARTIAL`: any non-empty bucket is emitted, with `n_observations` and
    `is_complete` recorded.
- Instrument identity is preserved throughout.

---

## 4. Execution semantics & no-lookahead policy

The convention is documented and encoded in tests:

```
At M5 close:     agent receives the completed M5 observation.
Action:          target position (target exposure).
Execution:       the NEXT M1 bar's open, adjusted by configured costs.
```

This means an agent acting on the observation at M5 bucket `10:00` (which
closes at `10:05`) executes at the M1 open of `10:05` — never at a price
inside the observed bar. **Same-bar look-ahead is impossible.** Execution
timing is configurable (`execution_timing`) for future experiments, and the M1
path is retained for later intrabar execution work.

The anti-lookahead property is tested explicitly in
`tests/test_no_lookahead.py` using synthetic data where the next open jumps
away from the observed close.

---

## 5. Price model (given OHLC-only data)

Deterministic execution model (`ExecutionConfig`):

```
mid_price  = next M1 open
buy_price  = mid + spread/2   (+ slippage if configured)
sell_price = mid - spread/2   (- slippage if configured)
commission = commission_per_unit * |units|   (per execution side)
```

- `spread_mode`: `fixed` (only mode in Phase 1).
- `slippage_mode`: `none` | `fixed`.
- `commission_per_unit`: quote-currency cost per base unit, per side.
- `ExecutionConfig.from_pips(pip_size, spread_pips, slippage_pips)` is a
  convenience for expressing spread/slippage in pips.

These are **assumptions** — clearly documented, configurable, and isolated —
because historical bid/ask is unavailable.

---

## 6. Accounting model

Portfolio accounting uses `decimal.Decimal` with a fixed context precision
(`DECIMAL_PRECISION = 50`) so results are exact and deterministic across runs.

### Account currency (Phase 3.1)

The simulator has an **explicit account currency** (`EnvironmentConfig.account_currency`,
default **USD**). Every account-level monetary quantity — balance, equity,
realized/unrealized PnL, gross exposure, margin, and drawdown — is expressed
in that currency. The architecture supports `EUR`, `GBP`, `JPY`, `CHF`, `CAD`,
`AUD`, `NZD` account currencies without rewriting the portfolio engine;
conversion is delegated to the FX service (`forexmind/environment/fx_conversion.py`).

### Pair representation

Each instrument is represented as `BASE/QUOTE` (`1 BASE = price QUOTE`), with
explicit metadata (`pip_size`, `price_precision`, `base_currency`,
`quote_currency`) in `forexmind/environment/instruments.py` (e.g.
`EURUSD` pip `0.0001`, `USDJPY` pip `0.01`).

### PnL and conversion

Raw trade PnL is computed in the instrument's **quote currency**:
`PnL_quote = N * (P_exit - P_entry)`, then converted to the account currency
before entering any account-level field. For a USD account:

- USD-quote pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD): `PnL_USD = PnL_quote` (no
  conversion needed).
- USD/XXX pairs (USDJPY, USDCHF, USDCAD): `PnL_USD = PnL_quote / USDXXX` using
  the pair's own **contemporaneous** price (never future prices).

```
balance          = initial + net realised PnL (account currency; closes/commissions only)
raw_pnl_quote    = units * (mid - entry)              (quote currency)
unrealized_pnl   = raw_pnl_quote * quote_to_account_factor   (account currency)
equity           = balance + unrealized_pnl
realized_pnl     = balance - initial
gross_exposure   = account-currency exposure (see below)
```

The identity `equity - initial == realized + unrealized` holds to ~1e-48
relative (average-cost entries involve one division under the fixed precision);
it is checked with a very tight tolerance in tests.

Position transitions (`flat→long`, `flat→short`, `flat→flat`, `long→flat`,
`short→flat`, `long→short`, `short→long`) are handled with correct realised-PnL
accounting for closed exposure and average-cost entry for same-direction
increases.

### Gross exposure (account currency)

`gross_exposure = |units| * price * quote_to_account_factor`, so it is directly
comparable with account equity. For USD/XXX pairs one base unit of USD is one
USD of notional, so `gross_exposure ≈ |units|` regardless of the JPY/CHF/CAD
price scale (e.g. USDJPY exposure `≈ |N|` USD).

### Margin & leverage (`MarginConfig`)

All margin quantities are in the account currency — never JPY margin against
USD equity:

```
margin_used     = gross_exposure_account * margin_requirement   (default 1/leverage)
free_margin     = equity - margin_used
margin_call     = free_margin < 0
liquidation     = equity <= margin_used * maintenance_margin_ratio  (position open)
max_leverage    = optional hard cap on gross_exposure_account / equity
leverage_used   = gross_exposure_account / equity   (dimensionless)
```

Liquidation is **deterministic**: when triggered, the position is force-closed
at the current mark price. No specific broker margin model is assumed — it is
configuration-driven.

### Execution costs

Commission is defined as a quote-currency cost per base unit; the resulting
commission is converted into the account currency before it affects
balance/equity. Spread is configured **per instrument** (see
`ExecutionConfig.instrument_spreads`) using each pair's correct pip size
(e.g. JPY pairs `0.02` for 2 pips vs `0.0002` for non-JPY pairs), not one raw
`0.0002` across every instrument. These are *configured assumptions* — the
dataset has no historical bid/ask.

---

## 7. Actions

The action space is **target exposure**, not BUY/SELL/HOLD:

```
-1.0  fully short     -0.5  half short     0.0  flat
+0.5  half long       +1.0  fully long
```

The action means *"adjust the portfolio to the requested target exposure"*.
Position sizing maps an exposure in `[-1, +1]` to base units so that the
**account-currency gross exposure** equals `|exposure| * equity`:

- `equity_fraction` (default, account-currency aware):
  `units = (exposure * equity) / (price * quote_to_account_factor)`
  For USD-quote pairs this is `exposure * equity / price`; for USD/XXX pairs
  it is `≈ exposure * equity` (one base USD == one USD notional).
- `fixed_units`: `units = exposure * fixed_units`

This formulation maps naturally to discrete MuZero actions later and can be
extended to continuous targets.

---

## 8. Reward

The reward is the log change in **account-currency** equity (transaction costs
already reflected):

```
reward_t = ln(equity_{t+1} / equity_t)
```

Because P&L and margin are now currency-consistent, the reward is economically
meaningful across all instruments. The reward service is extensible for later
experiments (risk-adjusted, drawdown-penalised, cost-penalised). Raw financial
quantities are exposed independently of the reward via the environment `info`.

---

## 9. Environment API

```python
obs, info                = env.reset(seed=..., instrument=..., start_index=..., horizon=...)
obs, reward, term, trunc, info = env.step(action)   # action = int index or float exposure
```

`info` includes `timestamp`, `instrument`, `equity`, `balance`, `position`,
`position_units`, `entry_price`, `unrealized_pnl`, `realized_pnl`,
`trade_cost`, `execution_price`, `drawdown`, `margin_used`, `free_margin`,
`leverage_used`, `session`, `minutes_since_last_bar`, `is_weekend_gap`, …
The neural agent must not depend on `info`.

Observations are raw structured state (`Observation`): a window of recent M5
bars, account state, and time metadata. No feature engineering lives in the
simulator.

### Determinism

Given the same dataset, configuration, seed, starting state and action
sequence, the simulator produces **identical** observations, execution prices,
portfolio states, rewards, and termination conditions. All randomness is
restricted to episode-start sampling via `numpy.random.default_rng(seed)`.

---

## 10. Project layout

```
forexmind/
    config.py              # Execution/Margin/Reward/Environment config (Decimal)
    data/
        schema.py          # canonical columns, MarketBar, validation
        loaders.py         # CSV/TSV/parquet loaders, column normalisation
        validator.py       # structural/OHLC checks, gap classification
        resampler.py       # M1 -> M5 (STRICT / PARTIAL)
        dataset.py         # MarketDataset / InstrumentData
    environment/
        costs.py           # ExecutionCostModel (spread/slippage/commission)
        execution.py       # ExecutionEngine
        portfolio.py       # Portfolio / Position / accounting (Decimal)
        margin.py          # MarginModel
        actions.py         # target-exposure action model
        state.py           # Observation / TimeInfo / AccountState / sessions
        reward.py          # RewardService
        forex_env.py       # ForexEnvironment (Gymnasium-style)
tools/
    inspect_dataset.py     # dataset discovery + report (section 25)
    process_data.py        # raw -> processed parquet + manifest (section 26)
    run_smoke.py           # real-data end-to-end smoke test (section 29)
tests/                     # 112 unit/integration tests
```

---

## 11. Usage

All commands run with the project virtualenv `forex_env` (Python 3.11.9):

```powershell
# 1. Inspect the raw datasets and write a report to data/reports/
python -m tools.inspect_dataset

# 2. Build processed parquet + manifest (raw files untouched)
python -m tools.process_data

# 3. Run the real-data end-to-end smoke test (deterministic + multi-instrument)
python -m tools.run_smoke --instrument EURUSD --second GBPUSD

# 4. Run tests, formatter, linter, type checker
python -m pytest
python -m ruff format forexmind tools tests
python -m ruff check forexmind tools tests
python -m mypy
```

### Minimal end-to-end example

```python
from forexmind.config import default_config
from forexmind.data.dataset import InstrumentData, MarketDataset
from forexmind.data.loaders import LoadConfig, load_many_concat
from forexmind.environment import ForexEnvironment
from tools.common import instrument_files

res = load_many_concat(instrument_files("EURUSD"), LoadConfig(sep=",", has_header=False))
data = InstrumentData.from_m1("EURUSD", res.frame)
ds = MarketDataset()
ds.add(data)

env = ForexEnvironment(ds, default_config(initial_balance="10000", leverage=50))
obs, info = env.reset(seed=0, start_index=1000, horizon=200)
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(2)  # stay flat / inspect
    if terminated or truncated:
        break
print(info["equity"], info["balance"], info["realized_pnl"])
```

---

## 12. Dataset findings (inspection)

Raw data: 138 MT5 CSV files, ~2.6 GB, one file per instrument per year
(2006–2025). Format: `DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOLUME`, no header,
`YYYY.MM.DD HH:MM` timestamps in broker server time (timezone ambiguous →
recorded as unknown). Per-instrument results are reported by
`tools/inspect_dataset` and stored in `data/reports/dataset_report.json`
plus the manifest at `data/processed/manifest.json`.

| Instrument | M1 rows | M5 rows (STRICT) | validation | weekend gaps | short gaps | large gaps |
| ---------- | ------- | ---------------- | ---------- | ------------ | ---------- | ---------- |
| EURUSD     | 7,173,235 | 1,342,224 | PASS | 1,037 | 152,303 | 719 |
| GBPUSD     | 7,153,230 | 1,341,367 | PASS | 1,037 | 151,602 | 716 |
| USDJPY     | 7,103,279 | 1,300,428 | PASS | 1,037 | 198,274 | 730 |
| USDCHF     | 7,096,976 | 1,307,828 | PASS | 1,037 | 185,799 | 716 |
| AUDUSD     | 7,068,492 | 1,295,021 | PASS | 1,037 | 204,529 | 708 |
| USDCAD     | 6,467,864 | 1,146,506 | PASS | 985  | 274,330 | 695 |
| NZDUSD     | 6,521,221 | 1,149,879 | PASS | 985  | 277,601 | 707 |

All pairs share the range `2006-01-02 19:00 .. 2025-12-31 16:57` and have
**0 duplicate timestamps, 0 NaN, 0 infinite values**. The lower STRICT-M5
counts reflect incomplete 5-minute buckets dropped around thin/holiday
periods (by design).

---

## 13. Known limitations

- Historical bid/ask/tick data is unavailable; spread/slippage/commission are
  configurable assumptions, never inferred from OHLC.
- **No historical overnight-financing (swap) data.** Phase 3.1 ships a
  zero-cost `FinancingModel` interface only — no swap rates are fabricated.
- **No fabricated cross rates.** The FX conversion service only converts via
  the seven documented USD-major pairs; an unavailable conversion raises an
  error rather than silently using `1.0`.
- Source timezone is unknown (MT5 server time); session labels assume a
  configurable UTC offset (default 0).
- One net position per instrument (no order book, no multi-position book).
- No intrabar stop-loss / take-profit simulation yet (architecture allows it).
- `STRICT` M5 bars require 5 contiguous M1 minutes; thin/holiday periods
  produce fewer M5 bars.
- Commission/slippage models are simple and deterministic; pair-specific or
  regime-dependent spreads are deferred (per-instrument spreads are
  configurable since Phase 3.1).
- Prices are `float64` in the market-data layer (standard for quotes);
  accounting is `Decimal` with 50-digit precision.

## 14. Next milestone

Phase 3: model-free RL baselines (SAC, PPO) with a training loop, replay and
checkpointing, experiment tracking, and validation-based model selection. The
simulator, observation pipeline, episode sampler, and evaluation framework stay
agent-agnostic and reusable by those algorithms.

---

# ForexMind — Phase 2 (Learning & Evaluation Layer)

Phase 2 adds the leakage-free learning/evaluation protocol on top of the Phase 1
simulator: temporal splits, a causal observation encoder, a deterministic
episode sampler, baseline strategies, and a full evaluation framework. **No
neural networks / RL algorithms / MCTS are implemented here.**

## 15. Learning protocol

### Temporal splits (`forexmind/data/splits.py`)

Splits are strictly chronological and never random, applied independently to
every instrument (exact timestamps, half-open `[start, end)`):

| Split      | Period          |
| ---------- | --------------- |
| TRAIN      | 2006-01-01 .. 2019-01-01 |
| VALIDATION | 2019-01-01 .. 2022-01-01 |
| TEST       | 2022-01-01 .. 2026-01-01 |

The `SplitDataset` enforces `max(train_ts) < min(validation_ts) < min(test_ts)`
for every instrument, verifies non-empty splits, and produces a machine-readable
manifest. `SplitConfig` is serializable.

### Context window (`forexmind/observation/window.py`)

`context_length = 64` by default. The window is exactly
`[current_index - context_length + 1, current_index]` — **no future rows** —
plus a `prior_close` (the bar before the window) so the first bar's return
features are defined. Policies:

- `strict_split` (default): the window **and** `prior_close` must come from the
  same split → zero cross-split contamination.
- `historical_warmup`: pre-split bars may be used as context; the episode
  itself stays in-split.

An invalid context raises `WindowError` (never silently padded with future
data).

### Observation representation (`forexmind/observation/`)

`EncodedObservation` is structured (not a raw dict):

| Field            | Shape      | Content |
| ---------------- | ---------- | ------- |
| `market`         | (64, 5)    | open/high/low/close/log returns vs. previous close |
| `account`        | (10,)      | normalized account state (below) |
| `time`           | (14,)      | cyclic hour/min/dow, minutes-since-bar, weekend flag, session one-hot |
| `instrument_vec` | (7,)       | deterministic one-hot instrument identity |
| `closes`         | (64,)      | raw closes (causal baselines / analysis; not the model input) |
| `prior_close`    | scalar     | close immediately before the window |

The flat `encoded` concatenation (shape `(351,)`) is the model-facing input.
Instrument order is the fixed canonical order
`EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD` (a learned embedding
can replace one-hot later without changing the environment).

Account features (normalized by initial balance / equity): `position_exposure`,
`position_units_normalized`, `entry_distance`, `unrealized_pnl_normalized`,
`realized_pnl_normalized`, `equity_return_from_initial`, `drawdown_normalized`,
`margin_utilization`, `free_margin_ratio`, `leverage_used`.

### Normalization (`forexmind/observation/normalization.py`)

No fitted statistics are used in Phase 2 (returns / ratios / relative prices
are inherently local and scale-independent), so **normalization leakage is
impossible**. A `Normalizer` abstraction (`identity` / `standard`) exists for
future features that need fitted statistics, with the hard rule
`fit(train) -> transform(train/validation/test)` — never fit on validation or
test.

### Episode sampling (`forexmind/episodes/`)

`EpisodeSampler(dataset, EpisodeConfig(split, horizon, context_length, seed))`
returns reproducible `EpisodeSpec`s:

- instruments sampled **uniformly** (EURUSD does not dominate by row count);
- valid starts sampled uniformly among starts that fit context + horizon inside
  the split;
- gap-aware: `GapPolicy(allow_cross_weekend=True, max_bar_gap_minutes=None)` by
  default — weekends may be crossed (the gap stays explicit in time features)
  and oversized gaps can be configured to invalidate starts;
- episodes never leave their split and never observe future data.

The sampler only *chooses* episodes; `ForexEnvironment` executes them (no
second simulation engine). Stratification by year/volatility/session is a
documented extension point.

## 16. Baselines (`forexmind/baselines/`)

All baselines implement `TradingAgent` (`reset(seed)`, `act(observation)`) and
only ever read the causal `EncodedObservation`:

| Agent             | Definition |
| ----------------- | ---------- |
| `flat`            | target 0.0 every step (no-trading reference) |
| `long` / `short`  | constant +1 / -1 target exposure (exposure "buy-and-hold" equivalents) |
| `random`          | uniform from {-1,-0.5,0,+0.5,+1} with a seeded RNG; multi-seed aggregation required |
| `momentum`        | `lookback=24`, `threshold=0.001`: return over lookback > thr → +1, < -thr → -1, else 0 |
| `mean_reversion`  | `lookback=32`, thresholds ±1.0: z-score of close vs. rolling mean/std → short/long/flat |
| `sma_crossover`   | `short_window=5`, `long_window=20`: short SMA > long SMA → +1 etc. |

Parameters are fixed, documented defaults — **not tuned on the test set**.
Default parameters are separated from experiment parameters (§30).

## 17. Evaluation framework (`forexmind/evaluation/`)

`EvaluationRunner` drives every agent through the identical protocol
(reset → observe → act → step → record) and never touches internal env state
except documented `info`. Results are per-`Trajectory` (equity curve, returns,
positions, trade log, metrics) with the environment/window/encoder cached per
instrument.

**Metrics** (return/risk/risk-adjusted/trading) — see `metrics.py`:

- Return: `total_return`, `cumulative_log_return`, `annualized_return`.
- Risk: per-period and annualized volatility, maximum & average drawdown,
  downside deviation.
- Risk-adjusted: Sharpe, Sortino, Calmar.
- Trading: position changes, trades, avg trade duration, turnover, gross/net
  PnL, transaction costs, winning/losing trades, win rate, avg win/loss,
  profit factor.
- **Annualization**: default `auto` — uses the *actual* number of valid M5
  observations per year in the evaluated split (≈ 73k), configurable.

**Reporting** is per-instrument and per-period, then equal-weighted aggregate
(so EURUSD's row count does not dominate). Reports are JSON with full
reproducibility metadata (dataset version, split config, env / execution /
reward / episode / agent config, seeds, project version) plus a human-readable
summary. Random baselines are aggregated across seeds.

## 18. Leakage policy

- **No random temporal splitting** — splits are chronological per instrument.
- **No future normalization** — analytical transforms only; fitted statistics
  are train-only.
- **No same-bar future information** — the observation window ends at the
  current M5 close; execution uses the next M1 open (Phase 1 convention).
- **No test-set tuning** — validation is used for model/parameter selection;
  the test split is final and untouched.
- The backtest-integrity test (`tests/test_backtest_integrity.py`) constructs
  a dataset with a huge future jump and verifies no baseline can access it.

## 19. Phase 2 usage

```powershell
python -m tools.inspect_splits                     # split boundaries + integrity
python -m tools.inspect_observation --instrument EURUSD --split train --index 1000
python -m tools.run_baselines --split validation --episodes 100 --seed 42
python -m tools.run_baselines --split test --episodes 100 --seed 42 --seeds 1 2 3 4 5
python -m tools.evaluate_run --agent momentum --split validation --episodes 50 --seed 7
```

## 20. Phase 2 project layout (additions)

```
forexmind/
    data/splits.py             # SplitConfig, SplitDataset, integrity checks
    observation/               # schema, window, normalization, encoder
    episodes/                  # config, sampler, trajectory, action adapters
    baselines/                 # base + flat/random/buy_hold/momentum/mean_reversion/sma_crossover
    evaluation/                # runner, metrics, aggregation, report
tools/
    inspect_splits.py, inspect_observation.py, run_baselines.py, evaluate_run.py
tests/                         # +65 Phase 2 tests (177 total)
```

## 21. Phase 2 known limitations

- Baselines use only the causal observation window; SMA/mean-reversion lookbacks
  must be ≤ `context_length` (defaults are).
- Equal-weighted aggregation aligns episodes by step index (all benchmarks use
  a common horizon).
- Per-period drawdown is within-period (resets at each period start).
- `periods_per_year` is estimated from the split's M5 count; it is an
  approximation of the annualization factor and is stored in every report.
- Instrument identity is one-hot (a learned embedding is the documented
  replacement path).

---

# Phase 3 — Model-free RL training infrastructure

Phase 3 implements **model-free RL training** (SAC, then PPO) on the stable
Phases 1–2 foundation. It intentionally does **not** include MuZero, MCTS,
Dreamer, or a world model — those remain future phases.

Goals:

- Continuous **target-exposure** actions in $[-1, 1]$ (long/flat/short).
- **SAC** with twin critics + target critics and **automatic entropy
  temperature**; **PPO** with a clipped Gaussian policy and GAE.
- **Multi-CPU** training: worker processes are independent of the learner and
  configurable (`num_workers`), with CPU oversubscription prevented by setting
  `torch.set_num_threads` + `OMP/MKL_NUM_THREADS`.
- **Meaningful learning units**: `env_steps` and `gradient_updates` are
  tracked **separately**.
- **Checkpoints with resume**, automatic validation-based checkpoint selection
  (`Score = Sharpe − λ·MaxDD` default), and a final **test protocol** that
  freezes the best checkpoint and runs the untouched test split.
- **Automatic final result tables** (SAC vs. 7 baselines, per-instrument,
  per-year) in JSON/CSV/text.
- **Multi-seed** support (`--seeds 1 2 3 4 5`) with per-seed run directories.

## 22. Leakage-free protocol (unchanged from Phase 2)

1. **Train split only** enters the replay buffer / rollouts.
   (`environment.split: train`; the collector is wired to the train range.)
2. **Validation** is used *only* to select the best checkpoint
   (`Score = Sharpe − λ·max_drawdown_pct`, default λ = 1.0).
3. **Test** is evaluated exactly once at the end with the frozen best
   checkpoint — never during training.
4. Deterministic evaluation: the policy mean, no exploration noise.

## 23. Running training

```powershell
# Smoke test (tiny, deterministic, in-process)
python -m forexmind.training.train_sac --config configs/sac_smoke.yaml

# Real experiment on a dedicated multi-CPU machine
python -m forexmind.training.train_sac --config configs/sac_cpu.yaml --seeds 1 2 3 4 5
python -m forexmind.training.train_ppo  --config configs/ppo_cpu.yaml  --seeds 1 2 3 4 5

# Resume from a checkpoint
python -m forexmind.training.train_sac --config configs/sac_cpu.yaml --resume runs/sac_cpu_seed42/checkpoints/latest.pt

# Evaluate a frozen checkpoint (validation or test)
# --checkpoint accepts a .pt path, a run directory, or a run name (searched under --run-root)
python -m forexmind.training.evaluate_checkpoint --checkpoint runs/sac_cpu_seed42/checkpoints/best.pt --split validation
python -m forexmind.training.evaluate_checkpoint --checkpoint runs/sac_cpu_seed42 --split validation
python -m forexmind.training.evaluate_checkpoint --checkpoint sac_cpu_seed42 --run-root runs --split validation

# Final benchmark tables (SAC vs 7 baselines on untouched test)
python -m forexmind.training.evaluate_checkpoint --checkpoint runs/sac_cpu_seed42 --benchmark --out data/reports/benchmark_sac

# Worker-throughput sweep for machine sizing
python -m tools.benchmark_training --workers 1 2 4 8 16
```

### Checkpoints

Every run directory (e.g. `runs/sac_cpu_seed42/`) contains `checkpoints/` with:

| File | When written |
| ---- | ------------ |
| `step_0.pt` | at training start (so a run that is interrupted before the first periodic interval still leaves a resumable checkpoint) |
| `step_<env_steps>.pt` | every `checkpoint_every_env_steps` |
| `best.pt` | each time validation improves (`Score = Sharpe − λ·MaxDD`), i.e. the checkpoint selected by validation |
| `final.pt` | when a run completes |
| `rescue_step_<env_steps>.pt` | on interruption (SIGINT/SIGTERM/exception) so the run can be resumed |

`latest.txt` points at the most recent checkpoint, which is what `--resume` uses. The training launcher prints `Run directory` and the list of checkpoints at the end of each run, and `evaluate_checkpoint` prints a helpful list of existing checkpoints if you mistype a path.

### Progress bar

Training shows a live `tqdm` bar with the current / total environment steps, elapsed time, **ETA** and rate (env steps/s), plus live diagnostics in the suffix (gradient updates, recent mean episode return, and SAC `alpha`/`entropy`/`q1` or PPO `entropy`/`actor_loss`). `tqdm` is an optional dependency (install with `pip install tqdm` or `pip install -e .[train]`); if it is missing, training still runs and prints the periodic progress blocks instead.

`ExperimentConfig` is YAML-serializable and is persisted into every run
directory, checkpoint, and manifest, so runs are reproducible.

## 24. Phase 3 project layout (additions)

```
forexmind/training/
    __init__.py            # SACTrainer, PPOTrainer, ExperimentConfig
    config.py              # ExperimentConfig + nested dataclasses (YAML/JSON)
    networks.py            # MLP, SquashedGaussianActor, TwinQCritic, GaussianPolicy, ValueNet
    replay.py              # high-throughput numpy ring buffer (train split only)
    data.py                # processed-parquet dataset access + startup report
    policies.py            # build_policy_network, sample_action, PolicyAgent (eval)
    collector.py           # EnvWorker, SyncCollector, ProcessCollector (multi-CPU)
    checkpoint.py          # CheckpointManager, build_checkpoint_state
    metrics.py             # MetricStore, TrainerLogger, warnings (pathological policies)
    evaluator.py           # PolicyEvaluator + selection_score (uses Phase-2 runner)
    trainer.py             # BaseTrainer (loop, schedules, resume, finalize)
    sac.py                 # SACTrainer
    ppo.py                 # PPOTrainer
    benchmark.py           # final test tables (SAC vs baselines)
    cli.py, train_sac.py, train_ppo.py, evaluate_checkpoint.py
configs/                   # sac_cpu.yaml, ppo_cpu.yaml, sac_smoke.yaml, ppo_smoke.yaml
tools/benchmark_training.py
tests/                     # Phase 3 tests (SAC, workers, data, eval, repro)
```

## 25. Phase 3 known limitations

- The replay buffer is **not** persisted in checkpoints (kept small); a resumed
  run starts with an empty buffer and refills it. Only buffer metadata is saved.
- Checkpoints are written at `checkpoint_every_env_steps` plus `step_0` at
  start, `best` on validation improvement, `final` on completion, and a
  `rescue_step_*` on interruption. If a run is killed by the OS (e.g. Kaggle
  session timeout) and cannot run the signal handler, only `step_0.pt` is
  guaranteed to exist.
- Full bitwise reproducibility is guaranteed in the deterministic `sync`
  backend (and for worker/episode sampling in the `process` backend); the
  `process` backend's exact transition ordering can vary with OS scheduling.
- The default network is the flat `MLP(351)` observation baseline — a
  structured temporal encoder is the documented replacement path.
- Training is compute-heavy: 20M env steps × 32 workers is designed for a
  dedicated machine. Start with the smoke config and the throughput sweep
  (`tools/benchmark_training.py`) to size the box. On a shared/notebook
  session (e.g. Kaggle) keep `num_workers` small (2–4) and `total_env_steps`
  modest for a first run; each worker re-imports torch and loads the parquet
  dataset, so hundreds of workers will OOM the session.


