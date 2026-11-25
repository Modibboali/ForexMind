"""Tests for evaluation metrics, trade counting, aggregation, and runner."""

from __future__ import annotations

import math

import numpy as np
import pytest
from forexmind.baselines import make_agent
from forexmind.config import default_config
from forexmind.episodes.config import EpisodeConfig
from forexmind.episodes.sampler import EpisodeSampler, EpisodeSpec
from forexmind.episodes.trajectory import Trajectory
from forexmind.evaluation.aggregation import (
    aggregate_across_instruments,
    mean_log_return_series,
    per_period_report,
)
from forexmind.evaluation.metrics import (
    annualized_return,
    average_drawdown,
    calmar_ratio,
    compute_metrics,
    compute_series_metrics,
    cumulative_log_return,
    downside_deviation,
    max_drawdown,
    max_drawdown_pct,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    trade_statistics,
)
from forexmind.evaluation.runner import EvaluationRunner
from forexmind.observation.encoder import EncoderConfig, ObservationEncoder
from forexmind.observation.window import WindowConfig

from tests.synthetic import (
    make_instrument,
    make_split_dataset,
    timeline_m5,
)

# ---------------------------------------------------------------------------
# Return / risk metric formulas
# ---------------------------------------------------------------------------


def test_total_return() -> None:
    assert total_return(np.array([100.0, 110.0])) == pytest.approx(0.1)
    assert total_return(np.array([100.0, 50.0])) == pytest.approx(-0.5)


def test_cumulative_log_return() -> None:
    logr = np.array([math.log(1.1), math.log(0.95)])
    assert cumulative_log_return(logr) == pytest.approx(math.log(1.1 * 0.95))


def test_max_drawdown() -> None:
    eq = np.array([100.0, 120.0, 90.0, 110.0, 130.0])
    dd, trough, peak = max_drawdown(eq)
    assert dd == pytest.approx(90.0 / 120.0 - 1.0)  # -0.25
    assert trough == 2
    assert peak == 1
    assert max_drawdown_pct(eq) == pytest.approx(0.25)


def test_average_drawdown() -> None:
    eq = np.array([100.0, 100.0, 80.0, 100.0])
    # drawdown pct: [0, 0, 0.2, 0]
    assert average_drawdown(eq) == pytest.approx(0.05)


def test_sharpe_and_sortino_formulas() -> None:
    returns = np.array([0.01, -0.005, 0.015, 0.0, 0.01])
    periods = 1000
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    expected_sharpe = mean / std * math.sqrt(periods)
    assert sharpe_ratio(returns, periods) == pytest.approx(expected_sharpe)
    dd = downside_deviation(returns)
    expected_sortino = mean / dd * math.sqrt(periods)
    assert sortino_ratio(returns, periods) == pytest.approx(expected_sortino)


def test_downside_deviation() -> None:
    returns = np.array([0.01, -0.02, 0.03, -0.04, 0.0])
    dd = downside_deviation(returns)
    assert dd == pytest.approx(math.sqrt(np.mean(np.array([0.02, 0.04]) ** 2)))


def test_annualized_and_calmar() -> None:
    eq = np.array([100.0, 110.0])
    assert annualized_return(eq, 1000) == pytest.approx(1.1**1000 - 1.0)
    eq2 = np.array([100.0, 90.0])
    assert calmar_ratio(eq2, 100) == pytest.approx(annualized_return(eq2, 100) / 0.1)


def test_compute_series_metrics_empty() -> None:
    m = compute_series_metrics(np.empty(0), 1000)
    assert m["total_return"] == 0.0


# ---------------------------------------------------------------------------
# Trading metrics
# ---------------------------------------------------------------------------


def _trade_log() -> list[dict[str, object]]:
    return [
        {
            "step": 0,
            "units_delta": 100.0,
            "execution_price": 1.10,
            "cost": 0.1,
            "realized_pnl": 0.0,
        },
        {
            "step": 5,
            "units_delta": -100.0,
            "execution_price": 1.12,
            "cost": 0.1,
            "realized_pnl": 2.0,
        },
        {
            "step": 8,
            "units_delta": -50.0,
            "execution_price": 1.10,
            "cost": 0.1,
            "realized_pnl": -1.0,
        },
    ]


def test_trade_statistics() -> None:
    log = _trade_log()
    positions = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, -50.0, -50.0])
    stats = trade_statistics(
        log, positions, initial_balance=10000.0, final_equity=10001.0, n_steps=10
    )
    assert stats.n_trades == 3
    assert stats.winning_trades == 1
    assert stats.losing_trades == 1
    assert stats.win_rate == pytest.approx(0.5)
    assert stats.avg_win == pytest.approx(2.0)
    assert stats.avg_loss == pytest.approx(-1.0)
    assert stats.profit_factor == pytest.approx(2.0)
    assert stats.gross_pnl == pytest.approx(3.0)
    assert stats.net_pnl == pytest.approx(1.0)
    assert stats.transaction_costs == pytest.approx(0.3)
    assert stats.n_position_changes == 3
    # turnover = sum(|delta| * price) / initial
    expected_turnover = (100 * 1.10 + 100 * 1.12 + 50 * 1.10) / 10000.0
    assert stats.turnover == pytest.approx(expected_turnover)


def test_compute_metrics_bundle() -> None:
    eq = np.array([10000.0, 10100.0, 9900.0, 10200.0])
    logr = np.log(eq[1:] / eq[:-1])
    m = compute_metrics(
        equity=eq,
        log_returns=logr,
        position_units=np.array([0.0, 10.0, 10.0, 0.0]),
        trade_log=_trade_log(),
        initial_balance=10000.0,
        periods_per_year=1000,
    )
    assert m["total_return"] == pytest.approx(0.02)
    assert "sharpe" in m and "sortino" in m and "calmar" in m
    assert m["trading"]["n_trades"] == 3


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _trajectory(agent: str, instrument: str, log_returns: np.ndarray) -> Trajectory:
    eq = np.concatenate([[1.0], np.exp(np.cumsum(log_returns))])
    n = len(log_returns)
    return Trajectory(
        agent_name=agent,
        spec=EpisodeSpec(instrument, "test", 0, n, n, 0, 0),
        timestamps=np.datetime64("2022-01-01T00:00:00") + np.arange(n).astype("timedelta64[s]"),
        actions=np.zeros(n),
        rewards=log_returns.copy(),
        equity=eq,
        log_returns=log_returns.copy(),
        position_units=np.zeros(n),
    )


def test_aggregation_equal_weight() -> None:
    eur = _trajectory("m", "EURUSD", np.array([0.02, 0.02]))
    gbp = _trajectory("m", "GBPUSD", np.array([-0.02, 0.02]))
    _ts, agg, per_instr = aggregate_across_instruments({"EURUSD": [eur], "GBPUSD": [gbp]})
    # Aggregate per-step = mean across instruments: [0.0, 0.02]
    assert np.allclose(agg, np.array([0.0, 0.02]))
    assert set(per_instr) == {"EURUSD", "GBPUSD"}


def test_mean_log_return_series() -> None:
    a = _trajectory("m", "EURUSD", np.array([0.01, 0.02, 0.03]))
    b = _trajectory("m", "EURUSD", np.array([0.0, 0.0, 0.0]))
    _ts, mean, n = mean_log_return_series([a, b])
    assert n == 2
    assert np.allclose(mean, np.array([0.005, 0.01, 0.015]))


def test_per_period_report() -> None:
    ts = np.array(
        [np.datetime64("2022-01-01"), np.datetime64("2022-06-01"), np.datetime64("2023-01-01")]
    )
    logr = np.array([0.01, 0.02, 0.03])
    report = per_period_report(ts, logr, 1000)
    assert "2022" in report and "2023" in report
    assert report["2022"]["n_periods"] == 2
    assert report["2023"]["n_periods"] == 1


# ---------------------------------------------------------------------------
# Report config round-trip (guards evaluate_run --config)
# ---------------------------------------------------------------------------


def test_report_config_roundtrip() -> None:
    from forexmind.evaluation.report import env_config_to_dict
    from tools.common import encoder_config_from_dict, environment_config_from_dict

    env = default_config(
        initial_balance="10000", leverage=50, spread_value=0.0002, sizing_mode="equity_fraction"
    )
    env2 = environment_config_from_dict(env_config_to_dict(env))
    assert env2.margin.initial_balance == env.margin.initial_balance
    assert env2.execution.spread_value == env.execution.spread_value
    assert env2.sizing.mode == env.sizing.mode

    enc = EncoderConfig(context_length=16, initial_balance="10000")
    enc2 = encoder_config_from_dict(enc.to_dict())
    assert enc2.context_length == 16
    assert enc2.market_features == enc.market_features
    assert enc2.normalizer.market == enc.normalizer.market


# ---------------------------------------------------------------------------
# Runner end-to-end
# ---------------------------------------------------------------------------


def _runner_dataset() -> object:
    dates = ["2020-01-06", "2020-06-01", "2021-03-01", "2021-09-01", "2022-03-01", "2022-09-01"]
    return make_split_dataset(
        {
            "EURUSD": make_instrument("EURUSD", timeline_m5(dates, per_day=30)),
            "GBPUSD": make_instrument("GBPUSD", timeline_m5(dates, per_day=30)),
        }
    )


def test_runner_flat_agent_metrics() -> None:
    ds = _runner_dataset()
    env_config = default_config(
        initial_balance="10000",
        leverage=100,
        spread_value=0.0002,
        sizing_mode="fixed_units",
        fixed_units="10000",
    )
    encoder = ObservationEncoder(EncoderConfig(context_length=8, initial_balance="10000"))
    runner = EvaluationRunner(ds, env_config, encoder, WindowConfig(context_length=8))
    sampler = EpisodeSampler(ds, EpisodeConfig(split="test", horizon=10, context_length=8, seed=1))
    specs = sampler.sample(4, seed=1)
    flat = make_agent("flat")
    ev = runner.run_agent(flat, specs)
    assert ev.agent_name == "flat"
    assert ev.n_episodes() == 4
    assert set(ev.trajectories_by_instrument) <= {"EURUSD", "GBPUSD"}
    for trajs in ev.trajectories_by_instrument.values():
        for traj in trajs:
            assert traj.n_steps == 10
            assert "sharpe" in traj.metrics
            assert traj.metrics["trading"]["n_trades"] == 0  # flat never trades


def test_runner_reproducible() -> None:
    ds = _runner_dataset()
    env_config = default_config(
        initial_balance="10000",
        leverage=100,
        spread_value=0.0002,
        sizing_mode="fixed_units",
        fixed_units="10000",
    )
    encoder = ObservationEncoder(EncoderConfig(context_length=8, initial_balance="10000"))
    runner = EvaluationRunner(ds, env_config, encoder, WindowConfig(context_length=8))
    sampler = EpisodeSampler(ds, EpisodeConfig(split="test", horizon=10, context_length=8, seed=1))
    specs = sampler.sample(4, seed=1)
    momentum = make_agent("momentum")
    ev1 = runner.run_agent(momentum, specs)
    ev2 = runner.run_agent(momentum, specs)
    for instr in ev1.trajectories_by_instrument:
        t1 = ev1.trajectories_by_instrument[instr][0]
        t2 = ev2.trajectories_by_instrument[instr][0]
        assert np.array_equal(t1.equity, t2.equity)
        assert np.array_equal(t1.log_returns, t2.log_returns)
