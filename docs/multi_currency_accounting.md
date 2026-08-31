# ForexMind Phase 3.1 — Multi-Currency Accounting

This document describes the corrected financial/accounting layer that makes the
simulator economically correct and internally consistent for multi-pair
training. It accompanies the README's accounting section with the rationale,
timing conventions, and design decisions required by the Phase 3.1 task.

## 1. Account currency

The simulator has an **explicit account currency**:

```python
EnvironmentConfig(account_currency="USD")   # default
```

Every account-level monetary quantity — `balance`, `equity`, `realized_pnl`,
`unrealized_pnl`, `gross_exposure`, `margin_used`, `free_margin`, `drawdown` —
is expressed in this currency. It is configuration-driven, never assumed to be
USD in the implementation. The architecture supports future `EUR`, `GBP`,
`JPY`, `CHF`, `CAD`, `AUD`, `NZD` account currencies without rewriting the
portfolio engine; the default remains **USD**.

## 2. Pair representation

Each instrument is `BASE/QUOTE`, meaning `1 BASE = price QUOTE`. Explicit
metadata lives in `forexmind/environment/instruments.py`:

| Instrument | Base | Quote | pip_size | price_precision |
| --- | --- | --- | --- | --- |
| EURUSD | EUR | USD | 0.0001 | 5 |
| GBPUSD | GBP | USD | 0.0001 | 5 |
| USDJPY | USD | JPY | 0.01 | 3 |
| USDCHF | USD | CHF | 0.0001 | 5 |
| AUDUSD | AUD | USD | 0.0001 | 5 |
| USDCAD | USD | CAD | 0.0001 | 5 |
| NZDUSD | NZD | USD | 0.0001 | 5 |

Pip size is explicit metadata, **not** derived dynamically from arbitrary price
values. JPY pairs use the `0.01` pip convention.

## 3. PnL conversion

For `N` base units, price movement produces P&L in the **quote currency**:

```
PnL_quote = N * (P_exit - P_entry)
```

That quote-currency P&L is converted into the account currency before it enters
any account-level field, via `forexmind/environment/fx_conversion.py`
(`CurrencyConverter.quote_to_account_factor`). For a USD account:

- **USD-quote pairs** (EURUSD, GBPUSD, AUDUSD, NZDUSD): quote == account, so
  no conversion (`factor == 1`). `PnL_USD = PnL_quote`. **Never double-convert.**
- **USD/XXX pairs** (USDJPY, USDCHF, USDCAD): `factor == 1 / USDXXX`, so
  `PnL_USD = PnL_quote / USDXXX`.

The general `CurrencyConverter.convert(amount, from, to, rates)` handles
same-currency, direct-pair, and inverse-pair conversions, and **raises** on any
conversion that cannot be derived from the available instruments (no fabricated
cross rates).

## 4. Timing convention

Conversion never uses future prices:

- **Realized PnL** (execution time): converted using the pair's price at that
  same execution timestamp.
- **Unrealized PnL** (mark): converted using the current mark price.
- For USD/XXX pairs the pair's own current price is the conversion price.

## 5. Gross exposure

Account-currency gross exposure:

```
gross_exposure_account = |units| * price * quote_to_account_factor
```

For USD/XXX pairs this is `≈ |units|` USD (one base unit of USD is one USD of
notional), correctly comparable with USD equity. This account-currency exposure
is used for margin, leverage, risk diagnostics, and exposure limits.

## 6. Margin

Margin is expressed consistently with account equity:

```
margin_used     = gross_exposure_account * margin_requirement
free_margin     = equity - margin_used
leverage_used   = gross_exposure_account / equity   (dimensionless)
```

JPY margin is never compared against USD equity or equivalent mismatched
currencies.

## 7. Position sizing

`equity_fraction` sizing targets an account-currency gross exposure:

```
target_gross_exposure_account = |exposure| * equity
units = target_gross_exposure_account / (price * quote_to_account_factor)
```

For a USD-quote pair: `units ≈ exposure * equity / price`. For USDJPY: one base
unit of USD is one USD of notional, so `units ≈ exposure * equity`. The action
API is unchanged (`-1.0, -0.5, 0, +0.5, +1.0` = target notional exposure
fractions).

## 8. Execution costs

Commission is defined as a quote-currency cost per base unit and converted to
the account currency before affecting balance/equity. Spread is configured
**per instrument** (`ExecutionConfig.instrument_spreads`) using the correct pip
size per pair; it is not one raw `0.0002` value blindly applied to every
instrument. These remain configured assumptions (no historical bid/ask).

## 9. Bid/ask marking

`mtm_price` supports `"mid"` and `"bid_ask"` modes. With `bid_ask`, a long
position is marked at the bid and a short at the ask (liquidation occurs on the
adverse side of the spread), reflected in unrealized account equity. The default
remains `"mid"`.

## 10. Overnight financing

`forexmind/environment/financing.py` provides a `FinancingModel` interface and
a `ZeroCostFinancing` default. Phase 3.1 has **no historical swap data**, so no
broker-specific swap tables are implemented or fabricated. The interface is
architecturally capable of computing a financing cost from a position, elapsed
time, and instrument later.

## 11. Reward

```
reward_t = ln(equity_{t+1} / equity_t)
```

operating on **correct account-currency equity**. No arbitrary risk penalties
are added in this task.

## 12. Data limitation

There is no historical bid/ask/tick/financing data. `spread`, `slippage`, and
`financing` remain configured assumptions and are never presented as observed
historical facts.

## 13. Diagnostic tools

- `python -m tools.test_multi_currency_accounting` — synthetic per-pair raw /
  converted PnL and account-currency gross exposure table (diagnostic only).
- `python -m tools.smoke_multi_currency` — real-data smoke test across pairs
  printing raw PnL, converted PnL, equity, gross exposure, margin, free margin,
  and reward.
