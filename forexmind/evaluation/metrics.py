"""Evaluation metrics (Phase 2).

Return / risk / risk-adjusted / trading metrics.  Annualization is
configurable and, by default, derived from the *actual number of valid trading
periods* in the evaluated split (M5 observations per year), not a hard-coded
equity-market convention.

Terminology
-----------
* A "step" is one M5 decision interval.
* ``trade`` = one executed position adjustment (a ``long -> short`` flip is a
  single execution that closes the old exposure and opens the new one).
* Net PnL = ``final_equity - initial_balance``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Return / risk metrics
# ---------------------------------------------------------------------------


def total_return(equity: np.ndarray) -> float:
    """``E_T / E_0 - 1``."""
    if len(equity) < 2 or equity[0] <= 0:
        return 0.0
    return float(equity[-1] / equity[0] - 1.0)


def cumulative_log_return(log_returns: np.ndarray) -> float:
    """Sum of per-step log equity returns."""
    return float(np.sum(log_returns)) if len(log_returns) else 0.0


def annualized_return(equity: np.ndarray, periods_per_year: float) -> float:
    """Per-period geometric mean scaled to one year."""
    n = len(equity) - 1
    if n <= 0 or equity[0] <= 0 or equity[-1] <= 0:
        return 0.0
    total = equity[-1] / equity[0]
    return float(total ** (periods_per_year / n) - 1.0)


def per_period_volatility(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1))


def annualized_volatility(returns: np.ndarray, periods_per_year: float) -> float:
    return per_period_volatility(returns) * math.sqrt(max(periods_per_year, 0.0))


def downside_deviation(returns: np.ndarray, target: float = 0.0) -> float:
    """Root-mean-square of returns below ``target``."""
    if len(returns) == 0:
        return 0.0
    neg = returns[returns < target]
    if len(neg) == 0:
        return 0.0
    return float(math.sqrt(np.mean((neg - target) ** 2)))


def sharpe_ratio(returns: np.ndarray, periods_per_year: float, risk_free: float = 0.0) -> float:
    """Excess mean return over per-period std, annualized."""
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 0.0:
        return 0.0
    excess = float(np.mean(returns)) - risk_free / periods_per_year
    return excess / std * math.sqrt(max(periods_per_year, 0.0))


def sortino_ratio(returns: np.ndarray, periods_per_year: float, risk_free: float = 0.0) -> float:
    """Excess mean return over downside deviation, annualized."""
    if len(returns) < 2:
        return 0.0
    dd = downside_deviation(returns)
    if dd <= 0.0:
        return 0.0
    excess = float(np.mean(returns)) - risk_free / periods_per_year
    return excess / dd * math.sqrt(max(periods_per_year, 0.0))


def max_drawdown(equity: np.ndarray) -> tuple[float, int, int]:
    """Maximum peak-to-trough drawdown as a fraction.

    Returns ``(max_dd, trough_index, peak_index)``; ``max_dd`` is negative.
    """
    if len(equity) == 0:
        return 0.0, 0, 0
    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0
    trough = int(np.argmin(dd))
    peak = int(np.argmax(equity[: trough + 1]))
    return float(dd[trough]), trough, peak


def max_drawdown_pct(equity: np.ndarray) -> float:
    dd, _, _ = max_drawdown(equity)
    return abs(dd)


def average_drawdown(equity: np.ndarray) -> float:
    """Mean of the running drawdown percentages over the whole curve."""
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    dd = 1.0 - equity / running_max
    return float(np.mean(dd))


def calmar_ratio(equity: np.ndarray, periods_per_year: float) -> float:
    """Annualized return divided by max drawdown magnitude."""
    mdd = max_drawdown_pct(equity)
    if mdd <= 0.0:
        return 0.0
    return annualized_return(equity, periods_per_year) / mdd


def estimate_periods_per_year(timestamps: np.ndarray) -> float:
    """Estimate annualization periods from actual observation timestamps."""
    if len(timestamps) < 2:
        return 50000.0
    span_s = float((timestamps[-1] - timestamps[0]) / np.timedelta64(1, "s"))
    years = max(span_s / (365.25 * 24 * 3600.0), 1e-9)
    periods = len(timestamps) - 1
    return max(periods / years, 1.0)


# ---------------------------------------------------------------------------
# Trading metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeStats:
    n_trades: int
    n_position_changes: int
    n_direction_changes: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    gross_pnl: float
    net_pnl: float
    transaction_costs: float
    turnover: float
    avg_trade_duration_steps: float

    def to_dict(self) -> dict[str, object]:
        return {k: v for k, v in self.__dict__.items()}


def position_change_count(positions: np.ndarray) -> int:
    """Number of steps where the signed position changed."""
    if len(positions) < 2:
        return 0
    return int(np.sum(positions[1:] != positions[:-1]))


def direction_change_count(positions: np.ndarray) -> int:
    """Number of steps where the position *direction* changed."""
    if len(positions) < 2:
        return 0
    return int(np.sum(np.sign(positions[1:]) != np.sign(positions[:-1])))


def trade_statistics(
    trade_log: list[dict[str, Any]],
    position_units: np.ndarray,
    *,
    initial_balance: float,
    final_equity: float,
    n_steps: int,
) -> TradeStats:
    """Compute trading metrics from a per-execution trade log.

    Each log entry has ``units_delta``, ``execution_price``, ``cost``,
    ``realized_pnl`` and ``step`` (see the evaluation runner).
    """
    if not trade_log:
        return TradeStats(
            0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, final_equity - initial_balance, 0.0, 0.0, 0.0
        )

    wins = [float(e["realized_pnl"]) for e in trade_log if float(e["realized_pnl"]) > 0]
    losses = [float(e["realized_pnl"]) for e in trade_log if float(e["realized_pnl"]) < 0]
    costs = sum(float(e["cost"]) for e in trade_log)
    turnover = (
        sum(abs(float(e["units_delta"])) * float(e["execution_price"]) for e in trade_log)
        / initial_balance
        if initial_balance > 0
        else 0.0
    )
    steps = [int(e["step"]) for e in trade_log]
    durations = [b - a for a, b in itertools.pairwise(steps)]
    if steps:
        durations.append(n_steps - steps[-1])
    avg_duration = float(np.mean(durations)) if durations else 0.0

    sum_wins = float(np.sum(wins)) if wins else 0.0
    sum_losses = abs(float(np.sum(losses))) if losses else 0.0
    if sum_losses > 0:
        profit_factor = sum_wins / sum_losses
    elif sum_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    win_rate = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0.0

    return TradeStats(
        n_trades=len(trade_log),
        n_position_changes=position_change_count(position_units),
        n_direction_changes=direction_change_count(position_units),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=win_rate,
        avg_win=float(np.mean(wins)) if wins else 0.0,
        avg_loss=float(np.mean(losses)) if losses else 0.0,
        profit_factor=profit_factor,
        gross_pnl=sum_wins + sum_losses,
        net_pnl=final_equity - initial_balance,
        transaction_costs=costs,
        turnover=turnover,
        avg_trade_duration_steps=avg_duration,
    )


# ---------------------------------------------------------------------------
# Full metric bundle
# ---------------------------------------------------------------------------


def compute_series_metrics(log_returns: np.ndarray, periods_per_year: float) -> dict[str, object]:
    """Return/risk metrics from a log-return series (chained relative equity).

    Used for aggregated and per-period reporting where a real position/trade
    log is not available.  Relative equity starts at 1.0.
    """
    n = len(log_returns)
    if n == 0:
        return {
            "n_periods": 0,
            "total_return": 0.0,
            "cumulative_log_return": 0.0,
            "annualized_return": 0.0,
            "per_period_volatility": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "average_drawdown": 0.0,
            "downside_deviation": 0.0,
        }
    equity = np.concatenate([[1.0], np.exp(np.cumsum(log_returns))])
    returns = equity[1:] / equity[:-1] - 1.0
    dd, trough, peak = max_drawdown(equity)
    return {
        "n_periods": n,
        "total_return": total_return(equity),
        "cumulative_log_return": cumulative_log_return(log_returns),
        "annualized_return": annualized_return(equity, periods_per_year),
        "per_period_volatility": per_period_volatility(returns),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "calmar": calmar_ratio(equity, periods_per_year),
        "max_drawdown": dd,
        "max_drawdown_pct": abs(dd),
        "max_drawdown_trough_index": trough,
        "max_drawdown_peak_index": peak,
        "average_drawdown": average_drawdown(equity),
        "downside_deviation": downside_deviation(returns),
    }


def compute_metrics(
    *,
    equity: np.ndarray,
    log_returns: np.ndarray,
    position_units: np.ndarray,
    trade_log: list[dict[str, Any]],
    initial_balance: float,
    periods_per_year: float,
) -> dict[str, object]:
    """Compute the full metric bundle for one trajectory / aggregated series."""
    returns = equity[1:] / equity[:-1] - 1.0 if len(equity) > 1 else np.empty(0)
    trade = trade_statistics(
        trade_log,
        position_units,
        initial_balance=initial_balance,
        final_equity=float(equity[-1]) if len(equity) else initial_balance,
        n_steps=len(log_returns),
    )
    dd, trough, peak = max_drawdown(equity)
    return {
        "n_periods": len(log_returns),
        "total_return": total_return(equity),
        "cumulative_log_return": cumulative_log_return(log_returns),
        "annualized_return": annualized_return(equity, periods_per_year),
        "per_period_volatility": per_period_volatility(returns),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "calmar": calmar_ratio(equity, periods_per_year),
        "max_drawdown": dd,
        "max_drawdown_pct": abs(dd),
        "max_drawdown_trough_index": trough,
        "max_drawdown_peak_index": peak,
        "average_drawdown": average_drawdown(equity),
        "downside_deviation": downside_deviation(returns),
        "final_equity": float(equity[-1]) if len(equity) else 0.0,
        "trading": trade.to_dict(),
    }
