# ForexMind — Phase 1

A research-grade, **deterministic** Forex market/data environment for
reinforcement-learning research. Phase 1 builds the reliable foundation that
later agents (SAC, PPO, Dreamer, MuZero, Stochastic MuZero, offline RL) will
consume: a canonical M1 data layer, dataset validation, deterministic M1→M5
resampling, an instrument-aware dataset, an execution-cost model, portfolio /
margin accounting, and a Gymnasium-style environment.

> Phase 1 deliberately contains **no** neural networks, no RL algorithms, no
> MCTS, no technical-indicator features, no live trading. It is a simulator.

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

Conventions (mirroring a broker margin account):

```
balance          = initial + net realised PnL (changes only on closes/commissions)
unrealized_pnl   = units * (mid - entry)              (floating PnL)
equity           = balance + unrealized_pnl
realized_pnl     = balance - initial
gross_exposure   = |units| * price
```

The identity `equity - initial == realized + unrealized` holds to ~1e-48
relative (average-cost entries involve one division under the fixed precision);
it is checked with a very tight tolerance in tests.

Position transitions (`flat→long`, `flat→short`, `flat→flat`, `long→flat`,
`short→flat`, `long→short`, `short→long`) are handled with correct realised-PnL
accounting for closed exposure and average-cost entry for same-direction
increases.

### Margin & leverage (`MarginConfig`)

```
margin_used     = gross_exposure * margin_requirement   (default 1/leverage)
free_margin     = equity - margin_used
margin_call     = free_margin < 0
liquidation     = equity <= margin_used * maintenance_margin_ratio  (position open)
max_leverage    = optional hard cap on gross_exposure / equity
```

Liquidation is **deterministic**: when triggered, the position is force-closed
at the current mark price. No specific broker margin model is assumed — it is
configuration-driven.

---

## 7. Actions

The action space is **target exposure**, not BUY/SELL/HOLD:

```
-1.0  fully short     -0.5  half short     0.0  flat
+0.5  half long       +1.0  fully long
```

The action means *"adjust the portfolio to the requested target exposure"*.
Position sizing maps an exposure in `[-1, +1]` to base units:

- `equity_fraction` (default): `units = exposure * equity / execution_price`
- `fixed_units`: `units = exposure * fixed_units`

This formulation maps naturally to discrete MuZero actions later and can be
extended to continuous targets.

---

## 8. Reward

The Phase-1 reward is based on the change in account equity (transaction costs
already reflected in equity):

```
reward_t = ln(equity_{t+1} / equity_t)
```

The reward service is extensible for later experiments (risk-adjusted,
drawdown-penalised, cost-penalised). Raw financial quantities are exposed
independently of the reward via the environment `info`.

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
- Source timezone is unknown (MT5 server time); session labels assume a
  configurable UTC offset (default 0).
- One net position per instrument (no order book, no multi-position book).
- No intrabar stop-loss / take-profit simulation yet (architecture allows it).
- `STRICT` M5 bars require 5 contiguous M1 minutes; thin/holiday periods
  produce fewer M5 bars.
- Commission/slippage models are simple and deterministic; pair-specific or
  regime-dependent spreads are deferred.
- Prices are `float64` in the market-data layer (standard for quotes);
  accounting is `Decimal` with 50-digit precision.

## 14. Next milestone

Phase 2: observation encoder + RL training loop (SAC/PPO baselines), or the
MuZero/Dreamer planning stack. The simulator stays agent-agnostic: market data,
environment, execution, portfolio, reward, and agent remain separate layers.
