# PPO `best.pt` Evaluation Findings

Date: 2026-08-30
Checkpoint: `C:\Users\Chindo\Documents\ForexMind\best.pt`

## 1. Checkpoint contents

| Field | Value |
| --- | --- |
| Algorithm | PPO |
| Env steps | 9,502,720 |
| Gradient updates | 2,320 |
| Episodes | 18,432 |
| Dataset version | processed-parquet-2026 |
| Non-finite tensors | **0** (safe to load) |
| Sampling log_std | -1.103 (std 0.332) |

This checkpoint is the best-selected artifact of a run that exhibited the
divergence described in `docs/ppo_math_audit.md` (KL 0.01 -> 93M, clip -> 0.99,
`mean_abs_action` -> 1.0). It is retained for forensics only; it is the
**divergent** model, not a stable result.

## 2. Evaluation results (deterministic policy, 100 episodes)

| Metric | Validation | Test |
| --- | --- | --- |
| Sharpe | **-3.50** | **+6.80** |
| Sortino | -3.53 | +7.00 |
| Total return | -0.024% | +0.046% |
| Max drawdown | 0.094% | 0.059% |
| **Turnover** | **13.11** | **13.24** |

Command reproduced:

```bash
python -m forexmind.training.evaluate_checkpoint --checkpoint best.pt --split validation --episodes 100
python -m forexmind.training.evaluate_checkpoint --checkpoint best.pt --split test --episodes 100
```

## 3. Action-distribution probe (why the metrics are not credible)

Direct probe of the policy's deterministic actions over 2000 real training
observations and a full validation episode:

```text
training observations (2000):
    min=-1.0000  max=1.0000  mean=-0.2141  std=0.9742
    frac |a| < 1e-3 (flat): 0.000
    frac a > 0.999:          0.387
    frac a < -0.999:         0.604
    unique actions (3dp):    21

one validation episode (512 steps):
    mean action 0.8900, std 0.4512
    frac a > 0.999: 0.943
    frac a < -0.999: 0.049
    trades: 512 (a trade every step)
```

This is a **saturated / degenerate policy**: deterministic actions are pinned
at the ±1 bounds ~95-99% of the time and it churns the account (turnover ~13x
capital, one trade per step), producing only ~1e-4 magnitude returns. This is
exactly the divergence signature from `docs/ppo_math_audit.md`. Validation
Sharpe is negative while test is positive over this near-constant
full-long/full-short behavior - not a robust or credible result.

**Conclusion:** the earlier "19.94 Sharpe / 19.71 Sortino" and this `best.pt`
are UNVERIFIED. They come from the divergent run. A fresh run with the
corrected PPO implementation (`configs/ppo_stable.yaml`) is required before any
performance claim.

## 4. Bug found while evaluating: turnover metric

Every previous evaluation reported `turnover 0.0`. This was a **metric bug**,
not a flat/untrading policy.

Root cause: `_pooled_turnover` (in `forexmind/training/evaluator.py` and the
equivalent in `forexmind/training/benchmark.py`) used `_f(price)` to coerce the
trade execution price, but `_f` only handled `int`/`float`. In the trade log
`execution_price` is stored as a **string** (a Decimal serialization, e.g.
`'1.20088'`), so `_f` returned the default `0.0` and summed notional was 0 ->
turnover was always 0 regardless of how much the policy traded.

Fix: `_f` now coerces numeric strings too (validated in
`tests/test_training_eval.py::test_pooled_turnover_coerces_string_price`).
With the fix, this policy's true turnover is ~13.1-13.2x.

## 5. Files changed / decision impact

- `forexmind/training/evaluator.py` - `_f` string coercion (turnover now real)
- `forexmind/training/benchmark.py` - `_f` string coercion (same)
- `tests/test_training_eval.py` - regression test for string-price turnover

All existing tests, `ruff`, and `mypy` pass.

## 6. Next steps (Kaggle)

```bash
python -m forexmind.training.train_ppo --config configs/ppo_stable.yaml --seeds 1 --max-env-steps 100000   # 100k gate
python -m forexmind.training.train_ppo --config configs/ppo_stable.yaml --seeds 1                          # fresh stable run
python -m forexmind.training.evaluate_checkpoint --checkpoint runs/.../checkpoints/best.pt --split validation
```

Evaluate the **new** best checkpoint with the corrected turnover metric. Do not
use `best.pt` (the divergent artifact) as a production model.
