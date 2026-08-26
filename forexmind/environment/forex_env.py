"""Gymnasium-style deterministic Forex environment (Phase 1).

Timeline convention (documented in README):

* observations are delivered at each M5 close;
* the agent picks a target exposure;
* execution happens at the **next M1 bar's open** (the first M1 bar of the
  following M5 bucket), adjusted by the configured execution costs.

This avoids same-bar look-ahead: the agent never observes a bar and
simultaneously executes at a price inside it.

The environment composes small domain services (execution engine, portfolio,
margin model, reward service) rather than containing all financial logic.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from forexmind.config import EnvironmentConfig, _dec
from forexmind.data.dataset import InstrumentData, MarketDataset
from forexmind.data.schema import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, MarketBar
from forexmind.environment.actions import Action, resolve_action
from forexmind.environment.costs import ExecutionCostModel
from forexmind.environment.execution import ExecutionEngine, ExecutionReport
from forexmind.environment.margin import MarginModel
from forexmind.environment.portfolio import Portfolio, TradeResult
from forexmind.environment.reward import RewardService
from forexmind.environment.state import (
    AccountState,
    Observation,
    TimeInfo,
    session_for_timestamp,
)


class EnvironmentError(RuntimeError):
    """Raised for invalid environment usage."""


class ForexEnvironment:
    """A single-instrument deterministic episode environment.

    The environment owns one instrument timeline at a time; calling
    :meth:`reset` with a different ``instrument`` switches instruments without
    changing any simulator code.
    """

    def __init__(
        self,
        dataset: MarketDataset | InstrumentData,
        config: EnvironmentConfig,
        *,
        instrument: str | None = None,
        utc_offset_hours: int = 0,
    ) -> None:
        if isinstance(dataset, InstrumentData):
            ds = MarketDataset()
            ds.add(dataset)
            self._dataset: MarketDataset = ds
        else:
            self._dataset = dataset
        if len(self._dataset) == 0:
            raise EnvironmentError("dataset contains no instruments")
        self.config = config
        self._default_instrument = instrument
        self._utc_offset_hours = utc_offset_hours

        self._cost_model = ExecutionCostModel(config.execution)
        self._engine = ExecutionEngine(self._cost_model)
        self._margin = MarginModel(config.margin)
        self._reward = RewardService(config.reward)

        # Episode state (re-initialised by reset()).
        self._instrument: str | None = None
        self._data: InstrumentData | None = None
        self._m1: pd.DataFrame | None = None
        self._m5: pd.DataFrame | None = None
        self._portfolio: Portfolio | None = None
        self._exec_m1_idx: np.ndarray | None = None
        self._m5_ts: np.ndarray | None = None
        self._minutes_since_last: np.ndarray | None = None
        self._is_weekend_gap: np.ndarray | None = None
        self._timeline_cache: tuple[
            pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ] | None = None
        self._obs_idx = 0
        self._start_index = 0
        self._horizon_steps = 0
        self._prev_equity = Decimal(0)
        self._rng: np.random.Generator = np.random.default_rng(0)
        self._done = False
        self._last_info: dict[str, object] = {}

    # ------------------------------------------------------------------ reset

    def reset(
        self,
        *,
        seed: int | None = None,
        instrument: str | None = None,
        start_index: int | None = None,
        horizon: int | None = None,
    ) -> tuple[Observation, dict[str, object]]:
        """Start a new episode.

        ``seed`` controls any randomness (episode start sampling).  ``horizon``
        overrides the configured episode length (in steps).  ``start_index``
        pins the first M5 observation index (used for reproducibility and
        evaluation on explicit time ranges).
        """
        instr = self._resolve_instrument(instrument)
        self._instrument = instr
        self._data = self._dataset.get(instr)
        self._m1 = self._data.m1
        self._m5 = self._data.m5
        if len(self._m5) < 2:
            raise EnvironmentError(f"instrument {instr}: need at least 2 M5 bars")
        if len(self._m1) == 0:
            raise EnvironmentError(f"instrument {instr}: empty M1 data")

        self._rng = np.random.default_rng(seed if seed is not None else 0)
        self._obs_idx = 0
        self._done = False

        self._precompute_timeline()

        max_steps = len(self._m5) - 1
        horizon_steps = horizon if horizon is not None else self.config.horizon
        if horizon_steps is None:
            horizon_steps = max_steps
        horizon_steps = max(1, min(int(horizon_steps), max_steps))
        self._horizon_steps = horizon_steps

        max_start = len(self._m5) - 1 - self._horizon_steps
        if start_index is None:
            start_index = int(self._rng.integers(0, max_start + 1))
        elif not (0 <= start_index <= max_start):
            raise EnvironmentError(f"start_index {start_index} out of range [0, {max_start}]")
        self._obs_idx = start_index
        self._start_index = start_index

        # Fresh account.
        self._portfolio = Portfolio(instr, self.config.margin.initial_balance)
        self._mark_at_observation(self._obs_idx)
        self._prev_equity = self._portfolio.equity

        obs = self._build_observation(self._obs_idx)
        info = self._build_info(
            obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            trade=None,
            execution=None,
            liquidation=False,
        )
        self._last_info = info
        return obs, info

    # ------------------------------------------------------------------ step

    def step(self, action: int | float) -> tuple[Observation, float, bool, bool, dict[str, object]]:
        """Apply ``action`` (discrete index or raw target exposure).

        Returns ``(obs, reward, terminated, truncated, info)``.
        """
        if self._done:
            raise EnvironmentError("step() called after episode ended; call reset() first")
        if self._portfolio is None or self._m1 is None or self._m5 is None:
            raise EnvironmentError("environment not reset")
        if self._exec_m1_idx is None:
            raise EnvironmentError("timeline not initialised")

        i = self._obs_idx
        act = resolve_action(action)
        exec_idx = int(self._exec_m1_idx[i])
        if exec_idx >= len(self._m1):
            # No M1 bar available to execute on -> cannot take this step.
            self._done = True
            obs_idx = min(i + 1, len(self._m5) - 1)
            self._obs_idx = obs_idx
            obs = self._build_observation(obs_idx)
            info = self._build_info(
                obs,
                reward=0.0,
                terminated=False,
                truncated=True,
                trade=None,
                execution=None,
                liquidation=False,
            )
            self._last_info = info
            return obs, 0.0, False, True, info

        exec_mid = _dec(float(self._m1.iloc[exec_idx][OPEN]))
        exec_ts = pd.Timestamp(self._m1.iloc[exec_idx][TIMESTAMP])
        current_units = self._portfolio.position.units
        target_units = self._target_units(act, exec_mid)
        delta = target_units - current_units

        report = self._engine.execute(exec_ts, exec_mid, delta)
        trade = self._portfolio.adjust_to_target(
            target_units, report.execution_price, report.commission
        )

        # Mark to market at the next observation's close.
        next_idx = i + 1
        self._mark_at_observation(next_idx)

        terminated = False
        liquidation = False
        # Deterministic liquidation check.
        snap = self._margin.snapshot(
            equity=self._portfolio.equity,
            units=self._portfolio.position.units,
            price=self._portfolio.current_mid or exec_mid,
        )
        if snap.liquidation:
            liquidation = True
            self._liquidate(snap)
            terminated = True

        truncated = False
        if not terminated:
            # Truncate when the number of steps taken reaches the horizon
            # (relative to the episode start index).
            steps_taken = next_idx - self._start_index
            if steps_taken >= self._horizon_steps:
                truncated = True
                if self.config.close_at_episode_end and not self._portfolio.position.is_flat:
                    self._close_at_current_mid()

        reward = self._reward.reward(self._prev_equity, self._portfolio.equity)
        self._prev_equity = self._portfolio.equity
        self._obs_idx = next_idx
        self._done = terminated or truncated

        obs = self._build_observation(self._obs_idx)
        info = self._build_info(
            obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            trade=trade,
            execution=report,
            liquidation=liquidation,
        )
        self._last_info = info
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------- properties

    @property
    def instrument(self) -> str | None:
        return self._instrument

    @property
    def current_obs_index(self) -> int:
        """Absolute M5 index of the most recent observation."""
        return self._obs_idx

    @property
    def portfolio(self) -> Portfolio | None:
        return self._portfolio

    @property
    def done(self) -> bool:
        return self._done

    @property
    def last_info(self) -> dict[str, object]:
        return dict(self._last_info)

    def info_dict(self) -> dict[str, object]:
        """Alias exposing the most recent ``info`` (useful for diagnostics)."""
        return self.last_info

    # ---------------------------------------------------------------- helpers

    def _resolve_instrument(self, instrument: str | None) -> str:
        if instrument is not None:
            if instrument.upper() not in self._dataset:
                raise EnvironmentError(
                    f"unknown instrument {instrument!r}; available: {self._dataset.instruments}"
                )
            return instrument.upper()
        if self._default_instrument is not None:
            return self._default_instrument.upper()
        if len(self._dataset) == 1:
            return self._dataset.instruments[0]
        raise EnvironmentError(
            "instrument must be specified when the dataset has multiple instruments"
        )

    def _precompute_timeline(self) -> None:
        """Precompute the execution index map and gap metadata.

        The result depends only on the instrument's M1/M5 frames, so it is
        cached per frame identity and reused across resets (safe because the
        cached frame reference is identity-checked).
        """
        assert self._m1 is not None and self._m5 is not None
        if self._timeline_cache is not None:
            cached_m5, cached_m1, exec_idx, m5_ts, minutes, weekend = self._timeline_cache
            if cached_m5 is self._m5 and cached_m1 is self._m1:
                self._exec_m1_idx = exec_idx
                self._m5_ts = m5_ts
                self._minutes_since_last = minutes
                self._is_weekend_gap = weekend
                return

        m1_ts = self._m1[TIMESTAMP].to_numpy(dtype="datetime64[ns]")
        m5_ts = self._m5[TIMESTAMP].to_numpy(dtype="datetime64[ns]")
        interval = np.timedelta64(self.config.decision_interval_minutes, "m")
        # First M1 bar at/after each M5 close (bucket start + interval).
        self._exec_m1_idx = np.searchsorted(m1_ts, m5_ts + interval, side="left")
        self._m5_ts = m5_ts

        n = len(self._m5)
        minutes = np.zeros(n, dtype="int64")
        weekend = np.zeros(n, dtype=bool)
        if n > 1:
            gap_minutes = (m5_ts[1:] - m5_ts[:-1]) / np.timedelta64(1, "m")
            minutes[1:] = gap_minutes.astype("int64")
            days = m5_ts.astype("datetime64[D]").astype("int64")
            dow = (days + 3) % 7  # 1970-01-01 was Thursday (dayofweek 3)
            prev_dow, cur_dow = dow[:-1], dow[1:]
            # Weekend = market closure Fri/Sat -> Sun/Mon (Sunday->Monday is not).
            weekend[1:] = ((prev_dow >= 4) & (prev_dow <= 5)) & ((cur_dow == 6) | (cur_dow == 0))
        self._minutes_since_last = minutes
        self._is_weekend_gap = weekend
        self._timeline_cache = (
            self._m5,
            self._m1,
            self._exec_m1_idx,
            self._m5_ts,
            self._minutes_since_last,
            self._is_weekend_gap,
        )

    def _target_units(self, action: Action, exec_mid: Decimal) -> Decimal:
        assert self._portfolio is not None
        sizing = self.config.sizing
        exposure = _dec(action.target_exposure)
        if sizing.mode == "fixed_units":
            return exposure * sizing.fixed_units
        # equity_fraction: exposure * current equity / execution mid price
        equity = self._portfolio.equity
        return exposure * equity / exec_mid

    def _mark_at_observation(self, obs_idx: int) -> None:
        assert self._portfolio is not None and self._m5 is not None
        row = self._m5.iloc[obs_idx]
        mid = _dec(float(row[CLOSE]))
        if self.config.mtm_price == "bid_ask":
            prices = self._cost_model.execution_prices(mid)
            if self._portfolio.position.units > 0:
                self._portfolio.mark_to_market(prices.sell)
            else:
                self._portfolio.mark_to_market(prices.buy)
        else:
            self._portfolio.mark_to_market(mid)

    def _liquidate(self, snap: object) -> None:
        """Deterministically force-close the position at the current mark."""
        assert self._portfolio is not None
        mid = self._portfolio.current_mid
        if mid is None or self._portfolio.position.is_flat:
            return
        prices = self._cost_model.execution_prices(mid)
        close_price = prices.sell if self._portfolio.position.units > 0 else prices.buy
        commission = self._cost_model.commission(self._portfolio.position.units)
        self._portfolio.close_all(close_price, commission)
        self._portfolio.mark_to_market(mid)

    def _close_at_current_mid(self) -> None:
        assert self._portfolio is not None
        mid = self._portfolio.current_mid
        if mid is None:
            return
        prices = self._cost_model.execution_prices(mid)
        close_price = prices.sell if self._portfolio.position.units > 0 else prices.buy
        commission = self._cost_model.commission(self._portfolio.position.units)
        self._portfolio.close_all(close_price, commission)
        self._portfolio.mark_to_market(mid)

    def _truncate_at(self, obs_idx: int) -> None:
        self._obs_idx = obs_idx
        self._done = True

    # ------------------------------------------------------------- observation

    def _build_observation(self, obs_idx: int) -> Observation:
        assert self._m5 is not None and self._portfolio is not None
        ts = pd.Timestamp(self._m5.iloc[obs_idx][TIMESTAMP])

        window = self.config.observation_window
        start = max(0, obs_idx - window + 1)
        sub = self._m5.iloc[start : obs_idx + 1]
        t = sub[TIMESTAMP].to_numpy(dtype="datetime64[ns]")
        o = sub[OPEN].to_numpy(dtype="float64")
        h = sub[HIGH].to_numpy(dtype="float64")
        lo = sub[LOW].to_numpy(dtype="float64")
        cl = sub[CLOSE].to_numpy(dtype="float64")
        bars = tuple(
            MarketBar(
                timestamp=pd.Timestamp(t[i]),
                open=float(o[i]),
                high=float(h[i]),
                low=float(lo[i]),
                close=float(cl[i]),
            )
            for i in range(len(t))
        )

        assert self._m5_ts is not None
        minutes_since = (
            int(self._minutes_since_last[obs_idx]) if self._minutes_since_last is not None else 0
        )
        is_weekend = (
            bool(self._is_weekend_gap[obs_idx]) if self._is_weekend_gap is not None else False
        )

        snap = self._portfolio.snapshot()
        mark_price = self._portfolio.current_mid or _dec(float(self._m5.iloc[obs_idx][CLOSE]))
        margin = self._margin.snapshot(
            equity=snap.equity,
            units=snap.position.units,
            price=mark_price,
        )
        account = AccountState(
            balance=snap.balance,
            equity=snap.equity,
            position_units=snap.position.units,
            entry_price=snap.position.entry_price,
            unrealized_pnl=snap.unrealized_pnl,
            realized_pnl=snap.realized_pnl,
            gross_exposure=snap.gross_exposure,
            margin_used=margin.margin_used,
            free_margin=margin.free_margin,
            drawdown=snap.drawdown,
        )
        time_info = TimeInfo(
            timestamp=ts,
            hour=ts.hour,
            minute=ts.minute,
            day_of_week=ts.dayofweek,
            session=session_for_timestamp(ts, self._utc_offset_hours),
            minutes_since_last_bar=minutes_since,
            is_weekend_gap=is_weekend,
            utc_offset_hours=self._utc_offset_hours,
        )
        return Observation(
            instrument=self._instrument or "",
            step_index=obs_idx,
            timestamp=ts,
            market_window=bars,
            account=account,
            time=time_info,
        )

    def _build_info(
        self,
        obs: Observation,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        trade: TradeResult | None,
        execution: ExecutionReport | None,
        liquidation: bool,
    ) -> dict[str, object]:
        assert self._portfolio is not None
        snap = self._portfolio.snapshot()
        mark_price = self._portfolio.current_mid or _dec(float(obs.market_window[-1].close))
        margin = self._margin.snapshot(
            equity=snap.equity,
            units=snap.position.units,
            price=mark_price,
        )
        info: dict[str, object] = {
            "timestamp": obs.timestamp,
            "instrument": obs.instrument,
            "step_index": obs.step_index,
            "equity": snap.equity,
            "balance": snap.balance,
            "position": snap.position.direction,
            "position_units": snap.position.units,
            "entry_price": snap.position.entry_price,
            "unrealized_pnl": snap.unrealized_pnl,
            "realized_pnl": snap.realized_pnl,
            "gross_exposure": snap.gross_exposure,
            "margin_used": margin.margin_used,
            "free_margin": margin.free_margin,
            "leverage_used": margin.leverage_used,
            "drawdown": snap.drawdown,
            "trade_cost": Decimal(0),
            "execution_price": None,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "liquidation": liquidation,
            "session": obs.time.session.value,
            "minutes_since_last_bar": obs.time.minutes_since_last_bar,
            "is_weekend_gap": obs.time.is_weekend_gap,
        }
        if trade is not None:
            if trade.executed_units:
                spread_cost = abs(trade.executed_units) * abs(
                    self._cost_model.execution_prices(trade.execution_price).mid
                    - trade.execution_price
                )
            else:
                spread_cost = Decimal(0)
            info["trade_cost"] = trade.commission + spread_cost
            info["execution_price"] = trade.execution_price
            info["trade_direction"] = trade.direction
            info["units_delta"] = trade.units_delta
            info["trade_realized_pnl"] = trade.realized_pnl
        if execution is not None:
            info["execution_mid"] = execution.mid_price
            info["execution_price"] = execution.execution_price
            info["execution_commission"] = execution.commission
        return info


def make_environment(
    dataset: MarketDataset | InstrumentData,
    config: EnvironmentConfig,
    *,
    instrument: str | None = None,
) -> ForexEnvironment:
    """Convenience factory for :class:`ForexEnvironment`."""
    return ForexEnvironment(dataset, config, instrument=instrument)
