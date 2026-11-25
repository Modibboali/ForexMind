# PPO Diagnosis Tool

## Overview

`tools/diagnose_ppo.py` is a comprehensive diagnostic tool that evaluates the PPO policy on identical validation episodes and compares it against 6 baseline strategies (flat, long, short, momentum, mean_reversion, sma_crossover).

## Usage

```bash
# Default: 100 validation episodes, seed 42
python -m tools.diagnose_ppo --checkpoint best.pt

# Custom episodes and seed
python -m tools.diagnose_ppo --checkpoint best.pt --episodes 100 --seed 42

# Custom output directory
python -m tools.diagnose_ppo --checkpoint best.pt --out data/reports/custom_diagnosis
```

## Output Files

All results saved to `data/reports/ppo_diagnosis/`:

1. **diagnosis.json** - Complete raw results (metrics, per-instrument, action stats)
2. **DIAGNOSIS.md** - Detailed human-readable report with analysis and recommendations
3. **agents_summary.csv** - All agents' aggregate metrics (return, Sharpe, turnover, etc.)
4. **ppo_per_instrument.csv** - PPO performance broken down by currency pair
5. **ppo_action_stats.csv** - PPO action distribution (mean/std, % long/short/flat, etc.)

## Key Metrics

### Per-Agent Aggregates
- **total_return**: Cumulative return over all episodes
- **sharpe**: Sharpe ratio (higher = better risk-adjusted return)
- **sortino**: Sortino ratio (Sharpe with only downside volatility)
- **max_drawdown_pct**: Maximum peak-to-trough loss (positive = magnitude)
- **turnover**: Total notional traded / initial capital
- **mean_reward**: Average per-step reward from environment

### PPO Action Statistics
- **action_mean / action_std / action_min / action_max**: Action distribution
- **action_mean_abs**: Mean absolute action value (engagement level)
- **pct_long / pct_short / pct_flat**: Percentage of actions by direction
- **position_changes**: Number of times position flips (long→short or vice versa)
- **n_trades**: Total number of executed trades

## How It Works

1. **Load checkpoint**: Extract PPO policy and experiment config from `best.pt`
2. **Build infrastructure**: Dataset, environment, encoder, window config
3. **Create identical episode specs**: Use `EpisodeSampler(seed=42)` to generate deterministic episodes
4. **Run all agents**: PPO + 6 baselines on the exact same episode specifications
5. **Compute metrics**: Per-episode and aggregate metrics for all agents
6. **Extract action stats**: Analyze PPO's action distribution and trading behavior
7. **Save results**: JSON, CSV, and markdown reports

## Interpretation Guide

### If PPO outperforms baselines:
- Policy learned a profitable strategy ✓
- Ready for production testing (on test split)
- Next: Evaluate on test split for final validation

### If PPO underperforms (current case):
- Policy failed to learn profitable strategy ✗
- **Possible causes:**
  1. **Reward function**: Incentivizes volume/action rather than return
  2. **Market regime change**: Training (2006-2019) vs validation (2019-2022) different dynamics
  3. **Observation/normalization**: Multi-pair normalization issues (JPY vs EUR scale)
  4. **Entropy coefficient**: Too high, encourages excessive exploration
  5. **Insufficient training**: Only 6M env_steps might not be enough

### Action Statistics Interpretation
- **High action_mean_abs** (>0.5): Policy is engaged, not flat
- **Low pct_flat** (near 0%): Policy avoids neutral positions
- **High position_changes**: Policy is reactive, flips frequently
- **High turnover**: Trading costs dominate returns
- **Biased actions** (pct_long >> pct_short): Learned directional bias

## Current Findings (best.pt)

**Status: ⏹️ NON-VIABLE**

- PPO return: **-0.000099** vs Long baseline: **+0.000378** (PPO loses by 477 bp)
- PPO Sharpe: **-1.523** vs Long Sharpe: **+5.473** (PPO 7 points worse)
- PPO turnover: **7.97x** vs Long: **1.05x** (excessive cost)
- PPO trades: **51,200** vs Long: **0** (constant churning)
- PPO actions: **78.57% long, 21.43% short, 0% flat** (never holds neutral)
- Position changes: **46,133** per 100 episodes (flip-flopping)

**Diagnosis:** PPO learned a high-turnover churning strategy that underperforms a simple "buy and hold" baseline. Transaction costs (not fully modeled in training) destroy profitability.

**Recommendation:** 
1. Inspect PPO training reward function (currently using default P&L-based reward)
2. Check if market regime change (training 2006-2019 vs validation 2019-2022) is the culprit
3. If training reward is clean, reduce entropy coefficient and retrain
4. Do NOT use this checkpoint in production

## Dependencies

- torch: Load checkpoint
- numpy: Action statistics computation
- pandas: (implicit via evaluation)
- All forexmind modules: Infrastructure, policies, evaluation

## Integration with Training Pipeline

This tool is **evaluation-only** and does not modify training. Use results to:
- Diagnose PPO policy issues before retraining
- Validate reward function changes
- Compare against baselines with zero training overhead
- Identify per-instrument performance issues

To retrain with fixes:
```bash
python -m tools.train_ppo --config configs/ppo_cpu.yaml --seed 42
```

## Notes

- Uses **validation split**, not test split (reserved for final evaluation)
- All agents evaluated on **identical episode specifications** (deterministic, seed=42)
- PPO policy is **frozen** (uses stored checkpoint, not retrained)
- Baseline agents use **deterministic policies** (no randomness)
- Transaction costs modeled as: spread (JPY 0.02, others 0.0002) + no slippage + no commission
