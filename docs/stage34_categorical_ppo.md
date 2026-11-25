# Stage 3.4 — Categorical PPO action system

## 1. Files changed

- Actions/simulator: `forexmind/environment/actions.py`, `forex_env.py`, and `forexmind/episodes/action_adapter.py`.
- Policy/training: `forexmind/training/networks.py`, `policies.py`, `collector.py`, `ppo.py`, `trainer.py`, `config.py`, and `checkpoint.py`.
- Diagnostics/evaluation: new `forexmind/training/action_diagnostics.py`, `evaluator.py`, `benchmark.py`, and `forexmind/evaluation/runner.py`.
- Tools: new `tools/validate_categorical_ppo.py`; updated `diagnose_ppo.py` and `smoke_ppo_correctness.py`.
- Tests: new `tests/test_categorical_trading.py`; updated action, environment, lookahead, PPO, parallel PPO, and evaluation regressions.
- Documentation/dependencies: `README.md`, this report, `pyproject.toml`, and `uv.lock` (Gymnasium and its two dependencies).

## 2. Old action semantics

PPO sampled a tanh-Gaussian scalar in (-1, 1), targeting exposure on every decision. Repeating an exposure recalculated units from equity and execution price. There was no explicit no-op. The simulator also supported an older five-index exposure grid.

## 3. New action semantics

| ID | Action | Exposure target |
|---:|---|---:|
| 0 | HOLD | Preserve exact units |
| 1 | FLAT | 0.00 |
| 2 | SHORT_100 | -1.00 |
| 3 | SHORT_75 | -0.75 |
| 4 | SHORT_50 | -0.50 |
| 5 | SHORT_25 | -0.25 |
| 6 | LONG_25 | +0.25 |
| 7 | LONG_50 | +0.50 |
| 8 | LONG_75 | +0.75 |
| 9 | LONG_100 | +1.00 |

The public integer space is `gymnasium.spaces.Discrete(10)`. Raw float exposures remain supported for SAC and baseline compatibility. Existing fixed-unit sizing remains available; the PPO experiment uses account-currency equity-fraction sizing.

## 4. HOLD implementation

`Action(None)` represents HOLD. The simulator branches before `_target_units`, preserves the existing Decimal units, and skips both the execution engine and portfolio adjustment. No policy trade or trading cost occurs. Mark-to-market, financing, reward, margin liquidation, and configured end-of-episode closure retain their existing timing. Forced closures are counted separately from policy executions.

## 5. FLAT implementation

Targets literal `Decimal(0)` without a sizing calculation. A nonzero position closes through the normal execution/accounting path. An already-flat position produces no execution, including direct environment calls that bypass the policy mask.

## 6. Masking and observation audit

`env.action_masks()` uses current signed account-currency gross exposure divided by equity. HOLD is always valid. FLAT is invalid only when units are zero, so tiny residual positions remain closable. The nearest nonzero target is masked only within an absolute exposure tolerance of **0.001**, or **0.1 percentage point of equity**. A position at 0.5005 masks LONG_50; at 0.502 it does not. No next-bar prices enter the mask.

The mask is supplied separately from the observation. Existing account features already provide signed exposure/direction, normalized units, unrealized PnL, equity return from initial capital, free-margin ratio, margin utilization, leverage, and entry distance. No observation shape, normalization, or future-information changes were needed. Time-in-position was not an existing observation feature.

## 7. Actor architecture

The existing MLP produces ten logits through `CategoricalPolicy.logits_net`. Invalid logits become the dtype's minimum finite value, then PyTorch `Categorical(logits=...)` computes normalized probabilities. Sampling, deterministic masked argmax, log-probability, and entropy all use this distribution. A missing/malformed mask, masked stored action, or nonfinite logits fail explicitly.

There is no Gaussian mean, standard deviation, tanh action transform, or pre-tanh state in categorical PPO. The old Gaussian class remains for legacy-code compatibility tests. Legacy configuration fields are parsed but not used or forwarded by PPO.

## 8. Rollouts, PPO math, and checkpoint handling

Transitions contain integer `action`, boolean `action_mask`, old log-probability, observation/next observation, value/bootstrap value, reward, terminal/truncation flags, worker/trajectory/fragment IDs, and execution diagnostics. `action_raw` was removed. Updates evaluate the same integer with its saved mask. The ratio and clipped loss, including the pre-existing numerical log-ratio guard, are unchanged. Every update reports the maximum deviation from ratio 1 before optimization.

GAE grouping, gamma, lambda, normalization, reward, accounting, conversions, execution timing, and spreads remain unchanged. Continuous actor keys are rejected before loading any weights, with an explicit instruction to start a new categorical run. Checkpoints carry action-policy metadata and persist action diagnostics and learner RNG state.

One concrete resume accounting bug was fixed: collectors restart episodes, so the first fresh trajectory must clear any saved partial worker episode totals. Completed episode history remains intact. This does not provide bit-exact restoration of simulator/worker state; that was not supported by the existing collector.

## Diagnostic definitions

- Action percentages use policy decisions as denominator and include all ten names, plus HOLD/FLAT/long/short groups.
- Position changes/executions/reversals count policy-driven changes; forced executions and turnover are separate.
- Turnover in categorical diagnostics is the sum of absolute account-currency executed notional divided by equity before each decision. It grows with the number of decisions and should not be confused with the older evaluator's turnover per initial episode capital.
- HOLD streaks span rollout fragments within a worker episode. Position duration runs from entry to flat or sign reversal; same-direction resizing retains the clock. Durations use M5 decision steps, not elapsed wall time across market gaps.
- Episode/resume boundaries censor unfinished positions. The reported mean/median include censored durations and their count is exposed.
- Transition counts cover FLAT -> LONG/SHORT, LONG -> HOLD/FLAT/SHORT, and SHORT -> HOLD/FLAT/LONG. Their percentages use all decisions as denominator.

## 9. Tests and verification

**406 passed, 1 skipped.** The skipped end-to-end test requires absent raw MetaTrader files; the processed seven-instrument dataset was used for the real experiment. New tests cover all ten mappings, both USD-quote and USD-base sizing, true HOLD with spread/commission/slippage and equity/price changes, FLAT closes, mask tolerance, categorical probabilities/entropy/sampling, unchanged-policy ratios, hand-computed clipping, old-checkpoint rejection, duration/worker isolation, resume accounting, forced-close event counting, and preservation of all update metrics.

Ruff passes on changed Python files; mypy passes across all **88 source files**. Existing SAC, accounting, reward, observation, lookahead, and parallel PPO regressions pass.

## 10. Corrected smoke-run metrics

Completed **100,352 environment steps**, **49 PPO updates**, and **196 episodes of 512 steps**, with two worker processes, all seven instruments, seed 42, 256x256 ReLU MLP, gamma 0.99, lambda 0.95, unchanged log-equity reward, and strict finite checking. Training plus periodic validation took 1021.73 seconds. Final checkpoint evaluation used 28 fixed-seed validation episodes (seed 34042), 14,336 decisions. The exact episode specifications are saved.

| PPO diagnostic across all 49 updates | Minimum | Mean | Maximum | Final |
|---|---:|---:|---:|---:|
| entropy | 1.3907944 | 1.9019392 | 2.1980856 | 1.4136162 |
| approx_kl_log | 0.0011064451 | 0.0049219421 | 0.0080386469 | 0.0061714006 |
| clip_fraction | 0.0005859375 | 0.057085061 | 0.12060547 | 0.08515625 |
| actor_loss | -0.035338975 | -0.029843084 | -0.024164013 | -0.027299691 |
| critic_loss | 4.3230113e-07 | 3.2568346e-06 | 3.2843338e-05 | 5.7910618e-07 |
| initial_ratio_max_error | 4.7683716e-07 | 9.0258462e-07 | 1.9073486e-06 | 1.9073486e-06 |

All validation checks passed: no nonfinite tensors, no invalid actions sampled, all ten actions reached in stochastic training, zero HOLD executions, zero unexpected forced closes, unchanged-policy ratios within 1e-5, valid episode lengths, two producing workers, and all 49 updates recorded.

Fixed validation: return **-0.1033%**, Sharpe **-18.4461**, maximum drawdown **0.1404%**, mean step reward **-6.9080677e-07**. Return/Sharpe use the existing evaluator aggregation; these are correctness observations, not evidence of profitability.

## 11. Action distribution and trading behavior

| Action | Training count | Training % | Validation count | Validation % |
|---|---:|---:|---:|---:|
| HOLD | 27,311 | 27.2152 | 13,674 | 95.3823 |
| FLAT | 7,355 | 7.3292 | 0 | 0.0000 |
| SHORT_100 | 7,440 | 7.4139 | 258 | 1.7997 |
| SHORT_75 | 11,453 | 11.4128 | 254 | 1.7718 |
| SHORT_50 | 8,676 | 8.6456 | 23 | 0.1604 |
| SHORT_25 | 8,627 | 8.5967 | 7 | 0.0488 |
| LONG_25 | 9,483 | 9.4497 | 48 | 0.3348 |
| LONG_50 | 8,009 | 7.9809 | 72 | 0.5022 |
| LONG_75 | 7,981 | 7.9530 | 0 | 0.0000 |
| LONG_100 | 4,017 | 4.0029 | 0 | 0.0000 |

| Diagnostic | Training | Validation |
|---|---:|---:|
| pct_hold | 27.2152 | 95.3823 |
| pct_flat | 7.3292 | 0 |
| pct_long | 29.3866 | 0.837054 |
| pct_short | 36.069 | 3.78069 |
| position_changes | 73041 | 662 |
| actual_executions | 73041 | 662 |
| sign_reversals | 25298 | 6 |
| turnover | 49264.8 | 180.026 |
| forced_executions | 0 | 0 |
| forced_turnover | 0 | 0 |
| hold_executions | 0 | 0 |
| mean_consecutive_hold | 1.53648 | 51.2135 |
| median_consecutive_hold | 1 | 8 |
| max_hold_streak | 20 | 457 |
| mean_position_holding_duration | 2.75982 | 359.382 |
| median_position_holding_duration | 2 | 384.5 |
| censored_positions | 177 | 28 |

| Transition | Training count | Validation count |
|---|---:|---:|
| FLAT -> LONG | 3,628 | 22 |
| FLAT -> SHORT | 3,904 | 6 |
| LONG -> HOLD | 11,034 | 8,664 |
| LONG -> FLAT | 2,809 | 0 |
| LONG -> SHORT | 13,023 | 5 |
| SHORT -> HOLD | 13,885 | 2,893 |
| SHORT -> FLAT | 4,546 | 0 |
| SHORT -> LONG | 12,275 | 1 |

**The deterministic validation policy is mostly HOLD (95.38%).** FLAT accounts for 0.00%; combined LONG_100/SHORT_100 account for 1.80%. No reward shaping, inactivity penalty, or exposure restriction was added to change this behavior.

## 12. Remaining concerns

- This is one seed and a short experiment. Deterministic HOLD dominance and negative validation performance remain; stochastic training still churns substantially.
- Masks use a documented tolerance; meaningful drift beyond it makes rebalancing available again.
- Position durations include censored episodes. Worker environments restart on resume; learner/optimizer/RNG/history restore is supported, but bit-exact simulator continuation is not.
- The initial run exposed false forced-close counts from Decimal rounding and a stability summary limited to the rolling log window. Both were fixed and tested, then the full bounded experiment was repeated. Use the verified run for reporting; the initial artifacts remain for audit.
- The repeated run has identical final actor weights to the initial run: **True**. Reporting fixes did not alter learned actions.

## 13. Full-training readiness

**Yes, for a new categorical PPO baseline run on implementation correctness grounds.** The requested action-system and parallel-rollout checks pass. This does not establish profitability or remove the observed HOLD collapse. Old continuous PPO checkpoints must not be resumed.

Artifacts: [verified report](../runs/stage34_categorical_verified_seed42/validation_report.json), [all PPO updates](../runs/stage34_categorical_verified_seed42/ppo_updates.json), [run config](../runs/stage34_categorical_verified_seed42/config.yaml), [final checkpoint](../runs/stage34_categorical_verified_seed42/checkpoints/final.pt).
