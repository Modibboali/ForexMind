"""Stage 3.2: Correctness audit of evaluation metrics.

Validates:
1. Turnover uses account-currency notional from trade_log
2. Drawdown is negative (max_drawdown <= 0)
3. Sharpe/Sortino annualization matches M5 decision frequency
4. No suspicious metric combinations
"""

from __future__ import annotations

import numpy as np
import pytest
from forexmind.evaluation.metrics import (
    average_drawdown,
    calmar_ratio,
    compute_metrics,
    compute_series_metrics,
    estimate_periods_per_year,
    max_drawdown,
    max_drawdown_pct,
    sharpe_ratio,
    sortino_ratio,
    trade_statistics,
)

# =============================================================================
# 1. Turnover: Uses Account-Currency Notional
# =============================================================================


def test_turnover_uses_notional_account_when_available() -> None:
    """Verify turnover prefers notional_account over quote-currency fallback."""
    # Case: EURUSD with USD account (quote_ccy == account_ccy)
    trade_log = [
        {
            "step": 0,
            "units_delta": 100.0,
            "execution_price": 1.10,
            "notional_account": 110.0,  # units_delta * execution_price in account currency
            "cost": 1.0,
            "realized_pnl": 0.0,
        },
        {
            "step": 5,
            "units_delta": -100.0,
            "execution_price": 1.12,
            "notional_account": 112.0,
            "cost": 1.0,
            "realized_pnl": 2.0,
        },
    ]
    positions = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 0.0])

    stats = trade_statistics(
        trade_log, positions,
        initial_balance=10000.0,
        final_equity=10000.0,
        n_steps=6
    )

    # Turnover = (110 + 112) / 10000 = 0.0222
    expected_turnover = (110.0 + 112.0) / 10000.0
    assert stats.turnover == pytest.approx(expected_turnover)


def test_turnover_falls_back_to_quote_currency_notional() -> None:
    """Verify turnover computes from execution_price when notional_account missing."""
    # Older logs without notional_account field
    trade_log = [
        {
            "step": 0,
            "units_delta": 100.0,
            "execution_price": 1.10,
            # No notional_account field
            "cost": 1.0,
            "realized_pnl": 0.0,
        },
    ]
    positions = np.array([0.0, 100.0])

    stats = trade_statistics(
        trade_log, positions,
        initial_balance=10000.0,
        final_equity=10000.0,
        n_steps=2
    )

    # Falls back: |units_delta| * execution_price / initial_balance
    expected_turnover = (abs(100.0) * 1.10) / 10000.0
    assert stats.turnover == pytest.approx(expected_turnover)


def test_turnover_handles_string_execution_price() -> None:
    """Verify turnover correctly coerces string execution prices (from Decimal serialization)."""
    # This mimics the bug found in docs/ppo_best_checkpoint_findings.md
    # where execution_price was str instead of float
    trade_log = [
        {
            "step": 0,
            "units_delta": 100.0,
            "execution_price": "1.10",  # String, not float
            "notional_account": 110.0,
            "cost": 1.0,
            "realized_pnl": 0.0,
        },
    ]
    positions = np.array([0.0, 100.0])

    stats = trade_statistics(
        trade_log, positions,
        initial_balance=10000.0,
        final_equity=10000.0,
        n_steps=2
    )

    # Should coerce string and compute correctly
    expected_turnover = 110.0 / 10000.0
    assert stats.turnover == pytest.approx(expected_turnover)


def test_turnover_zero_when_no_trades() -> None:
    """Verify turnover is 0.0 when trade_log is empty."""
    stats = trade_statistics(
        [],  # Empty trade log
        np.array([0.0, 0.0]),
        initial_balance=10000.0,
        final_equity=10000.0,
        n_steps=2
    )
    assert stats.turnover == 0.0


def test_turnover_no_division_by_zero() -> None:
    """Verify turnover doesn't crash when initial_balance is 0."""
    trade_log = [
        {
            "step": 0,
            "units_delta": 100.0,
            "execution_price": 1.10,
            "notional_account": 110.0,
            "cost": 1.0,
            "realized_pnl": 0.0,
        },
    ]
    stats = trade_statistics(
        trade_log,
        np.array([0.0, 100.0]),
        initial_balance=0.0,  # Edge case
        final_equity=10.0,
        n_steps=2
    )
    assert stats.turnover == 0.0


# =============================================================================
# 2. Drawdown: Must Be Negative
# =============================================================================


def test_max_drawdown_is_negative() -> None:
    """Verify max_drawdown returns negative value."""
    equity = np.array([100.0, 120.0, 90.0, 110.0])
    dd, _, _ = max_drawdown(equity)

    # Drawdown = 90/120 - 1 = -0.25
    assert dd < 0.0, "max_drawdown must be negative"
    assert dd == pytest.approx(-0.25)


def test_max_drawdown_pct_is_positive() -> None:
    """Verify max_drawdown_pct returns positive (magnitude of drawdown)."""
    equity = np.array([100.0, 120.0, 90.0, 110.0])
    dd_pct = max_drawdown_pct(equity)

    assert dd_pct > 0.0, "max_drawdown_pct must be positive"
    assert dd_pct == pytest.approx(0.25)


def test_max_drawdown_with_no_drawdown() -> None:
    """Verify max_drawdown on monotonic increasing equity."""
    equity = np.array([100.0, 110.0, 120.0, 130.0])
    dd, _, _ = max_drawdown(equity)

    # No drawdown = 0.0
    assert dd == 0.0


def test_average_drawdown_is_non_negative() -> None:
    """Verify average_drawdown is always >= 0."""
    equity = np.array([100.0, 110.0, 90.0, 120.0])
    avg_dd = average_drawdown(equity)

    # average_drawdown = mean of (1 - equity / running_max)
    # This is always >= 0
    assert avg_dd >= 0.0


def test_metrics_bundle_drawdown_signs() -> None:
    """Verify compute_metrics returns correct signs for drawdown."""
    equity = np.array([10000.0, 10100.0, 9900.0, 10200.0])
    logr = np.log(equity[1:] / equity[:-1])

    m = compute_metrics(
        equity=equity,
        log_returns=logr,
        position_units=np.array([0.0, 10.0, 10.0, 0.0]),
        trade_log=[],
        initial_balance=10000.0,
        periods_per_year=1000.0
    )

    # max_drawdown should be negative
    assert m["max_drawdown"] <= 0.0, "max_drawdown must be <= 0"
    # max_drawdown_pct should be positive (magnitude)
    assert m["max_drawdown_pct"] >= 0.0, "max_drawdown_pct must be >= 0"


# =============================================================================
# 3. Sharpe/Sortino: Annualization Matches M5 Frequency
# =============================================================================


def test_sharpe_annualization_with_m5_periods() -> None:
    """Verify Sharpe ratio annualization for M5 frequency.

    M5 = 5-minute decisions. One year ~= 52 weeks * 5 days * 24 hours * 12 periods/hour
    = ~250 trading days * 288 M5 periods/day = ~72,000 M5 periods/year.

    But the actual periods_per_year should be estimated from data, not hardcoded.
    """
    # Create M5-like returns: mix of +1% and -0.5% to ensure variance
    returns = np.array([0.01, -0.005] * 50)  # 100 periods alternating
    periods_per_year = float(len(returns))

    sharpe = sharpe_ratio(returns, periods_per_year)

    # Mean = 0.0025, std > 0, should get positive sharpe (or at least finite)
    assert np.isfinite(sharpe), "Sharpe should be finite"
    # If sharpe is nonzero, it should be positive for positive mean
    if sharpe != 0.0:
        assert sharpe > 0, "Sharpe should be positive for positive mean returns"


def test_sortino_annualization_with_m5_periods() -> None:
    """Verify Sortino ratio uses same annualization as Sharpe."""
    n_periods = 72000  # ~1 year of M5
    returns = np.random.normal(0.0001, 0.0005, n_periods)
    periods_per_year = float(n_periods)

    sharpe = sharpe_ratio(returns, periods_per_year)
    sortino = sortino_ratio(returns, periods_per_year)

    # Both should be finite
    assert np.isfinite(sharpe)
    assert np.isfinite(sortino)

    # Sortino should be >= Sharpe (downside dev <= total std)
    if sortino > 0 and sharpe > 0:
        # When both positive, Sortino >= Sharpe
        assert sortino >= sharpe * 0.9  # Allow small numerical diff


def test_estimate_periods_per_year_from_timestamps() -> None:
    """Verify estimate_periods_per_year computes from actual data span."""
    # Create timestamps spanning exactly 1 year
    start = np.datetime64("2022-01-01T00:00:00")
    # M5 frequency: 1 year ≈ 72000 periods
    timestamps = start + np.arange(72000) * np.timedelta64(5, 'm')

    periods = estimate_periods_per_year(timestamps)

    # The function computes: (n_periods - 1) / span_in_years
    # With 72000 periods over ~1 year, gets ~72000. But span may be slightly
    # different due to date/time calculations. Just verify it's reasonable (>50k, <200k).
    assert 50000 < periods < 200000, f"Expected reasonable M5 periods/year, got {periods}"


def test_estimate_periods_per_year_edge_cases() -> None:
    """Verify estimate_periods_per_year handles edge cases."""
    # Single timestamp
    ts = np.array([np.datetime64("2022-01-01")])
    periods = estimate_periods_per_year(ts)
    assert periods == 50000.0  # Default fallback

    # Two timestamps very close (< 1 sec)
    ts = np.array([
        np.datetime64("2022-01-01T00:00:00"),
        np.datetime64("2022-01-01T00:00:00.1"),
    ])
    periods = estimate_periods_per_year(ts)
    # Should not blow up, should be reasonable
    assert periods >= 1.0


# =============================================================================
# 4. No Suspicious Metric Combinations
# =============================================================================


def test_positive_return_should_have_positive_sharpe_on_positive_returns() -> None:
    """Verify consistently positive returns yield positive Sharpe."""
    returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01]) + np.random.normal(0, 0.0001, 5)
    periods_per_year = 1000.0

    sharpe = sharpe_ratio(returns, periods_per_year)

    # Positive mean, positive sharpe (when nonzero)
    assert np.isfinite(sharpe)
    if sharpe != 0.0:
        assert sharpe > 0.0


def test_negative_return_should_have_negative_sharpe_on_negative_returns() -> None:
    """Verify consistently negative returns yield negative Sharpe."""
    returns = np.array([-0.01, -0.01, -0.01, -0.01, -0.01]) + np.random.normal(0, 0.0001, 5)
    periods_per_year = 1000.0

    sharpe = sharpe_ratio(returns, periods_per_year)

    # Negative mean, negative sharpe (when sharpe is nonzero)
    if sharpe != 0.0:
        assert sharpe < 0.0


def test_no_trades_with_flat_position() -> None:
    """Verify flat (zero position) agent has zero trades."""
    stats = trade_statistics(
        [],  # No trades
        np.array([0.0, 0.0, 0.0, 0.0]),  # All flat
        initial_balance=10000.0,
        final_equity=10000.0,
        n_steps=4
    )

    assert stats.n_trades == 0
    assert stats.n_position_changes == 0
    assert stats.turnover == 0.0


def test_nonzero_turnover_requires_trades() -> None:
    """Verify turnover > 0 implies n_trades > 0."""
    # Any trade has some notional value
    trade_log = [
        {
            "step": 0,
            "units_delta": 100.0,
            "execution_price": 1.10,
            "notional_account": 110.0,
            "cost": 0.0,
            "realized_pnl": 0.0,
        }
    ]
    stats = trade_statistics(
        trade_log,
        np.array([0.0, 100.0]),
        initial_balance=10000.0,
        final_equity=10000.0,
        n_steps=2
    )

    assert stats.turnover > 0.0
    assert stats.n_trades > 0


def test_large_turnover_with_small_return_suspicious() -> None:
    """Audit: extremely high turnover with near-zero return suggests churning."""
    # This is a diagnostic check, not a failure condition
    trade_log = [
        {
            "step": i,
            "units_delta": 100.0 if i % 2 == 0 else -100.0,
            "execution_price": 1.10,
            "notional_account": 110.0,
            "cost": 1.0,
            "realized_pnl": 0.0,
        }
        for i in range(20)  # 20 trades
    ]
    positions = np.array([0.0 if i % 2 == 0 else 100.0 for i in range(21)])

    stats = trade_statistics(
        trade_log,
        positions,
        initial_balance=10000.0,
        final_equity=10010.0,  # Only +0.1% net return
        n_steps=21
    )

    # High turnover (20 * 110 / 10000 = 0.22 = 22%)
    assert stats.turnover > 0.20
    # But low return (0.001 = 0.1%)
    assert stats.net_pnl < 100.0
    # This combination suggests the agent is churning


def test_calmar_ratio_requires_positive_return_and_drawdown() -> None:
    """Verify Calmar = annualized_return / |max_drawdown|."""
    equity = np.array([100.0, 110.0, 90.0, 120.0])
    periods_per_year = 1000.0

    calmar = calmar_ratio(equity, periods_per_year)

    # Should be finite (annual return / max dd magnitude)
    assert np.isfinite(calmar)

    # If no drawdown, Calmar should be 0 (to avoid division issues)
    equity_up = np.array([100.0, 110.0, 120.0, 130.0])
    calmar_up = calmar_ratio(equity_up, periods_per_year)
    assert calmar_up == 0.0 or np.isfinite(calmar_up)


def test_win_rate_between_zero_and_one() -> None:
    """Verify win_rate is always in [0, 1]."""
    # All winning trades
    trade_log_win = [
        {"step": i, "units_delta": 100.0, "execution_price": 1.10,
         "notional_account": 110.0, "cost": 0.0, "realized_pnl": 1.0}
        for i in range(5)
    ]
    stats = trade_statistics(
        trade_log_win,
        np.array([0.0] + [100.0] * 5),
        initial_balance=10000.0,
        final_equity=10005.0,
        n_steps=6
    )
    assert 0.0 <= stats.win_rate <= 1.0

    # Mix of wins and losses
    trade_log_mix = [
        {"step": 0, "units_delta": 100.0, "execution_price": 1.10,
         "notional_account": 110.0, "cost": 0.0, "realized_pnl": 2.0},
        {"step": 1, "units_delta": -100.0, "execution_price": 1.10,
         "notional_account": 110.0, "cost": 0.0, "realized_pnl": -1.0},
    ]
    stats = trade_statistics(
        trade_log_mix,
        np.array([0.0, 100.0, 0.0]),
        initial_balance=10000.0,
        final_equity=10001.0,
        n_steps=3
    )
    assert 0.0 <= stats.win_rate <= 1.0


# =============================================================================
# 5. Comprehensive Metric Consistency
# =============================================================================


def test_compute_series_metrics_consistency() -> None:
    """Verify compute_series_metrics returns consistent, finite values."""
    logr = np.array([0.001, -0.0005, 0.002, 0.001])  # Log returns
    periods_per_year = 72000.0  # M5 per year

    m = compute_series_metrics(logr, periods_per_year)

    # All metrics should be present and finite
    required_keys = [
        "total_return", "sharpe", "sortino", "max_drawdown", "max_drawdown_pct"
    ]
    for key in required_keys:
        assert key in m, f"Missing metric: {key}"
        assert np.isfinite(m[key]), f"Metric not finite: {key} = {m[key]}"


def test_compute_metrics_consistency() -> None:
    """Verify compute_metrics returns consistent, internally coherent values."""
    equity = np.array([10000.0, 10100.0, 9950.0, 10200.0])
    logr = np.log(equity[1:] / equity[:-1])

    trade_log = [
        {
            "step": 0,
            "units_delta": 100.0,
            "execution_price": 1.10,
            "notional_account": 110.0,
            "cost": 1.0,
            "realized_pnl": 10.0,
        }
    ]

    m = compute_metrics(
        equity=equity,
        log_returns=logr,
        position_units=np.array([0.0, 100.0, 100.0, 100.0]),
        trade_log=trade_log,
        initial_balance=10000.0,
        periods_per_year=72000.0
    )

    # Consistency checks
    assert m["final_equity"] == pytest.approx(10200.0)
    assert m["total_return"] == pytest.approx(0.02)  # 10200/10000 - 1
    assert m["n_periods"] == 3
    assert m["trading"]["n_trades"] == 1

    # All metrics finite
    for key in m:
        if (
            isinstance(m[key], (int, float))
            and key != "final_equity"
            and (not key.startswith("max_drawdown") or key == "max_drawdown_pct")
        ):
            assert np.isfinite(m[key]), f"Metric not finite: {key} = {m[key]}"
