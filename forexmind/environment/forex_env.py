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
from gymnasium.spaces import Discrete

from forexmind.config import EnvironmentConfig, _dec
from forexmind.data.dataset import InstrumentData, MarketDataset
from forexmind.data.schema import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, MarketBar
from forexmind.environment.actions import Action, resolve_action, valid_action_mask
from forexmind.environment.costs import ExecutionCostModel
from forexmind.environment.execution import ExecutionEngine, ExecutionReport
from forexmind.environment.financing import FinancingModel, ZeroCostFinancing
from forexmind.environment.fx_conversion import CurrencyConverter
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
        self.action_space: Discrete = Discrete(10)
        self._default_instrument = instrument
        self._utc_offset_hours = utc_offset_hours

        self._cost_model = ExecutionCostModel(config.execution)
        self._engine = ExecutionEngine(self._cost_model)
        self._margin = MarginModel(config.margin)
        self._reward = RewardService(config.reward)
        self._converter = CurrencyConverter(config.account_currency)
        self._financing: FinancingModel = ZeroCostFinancing()

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
        self._timeline_cache: (
            tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
        ) = None
        self._obs_idx = 0
        self._start_index = 0
        self._horizon_steps = 0
        self._prev_equity = Decimal(0)
        self._initial_episode_equity = Decimal(0)
        self._steps_taken = 0
        self._trade_opened = False
        self._voluntary_closed = False
        self._timeout = False
        self._termination_reason: str | None = None
        self._terminal_reward_issued = False
        self._entry_execution: ExecutionReport | None = None
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
        self._steps_taken = 0
        self._trade_opened = False
        self._voluntary_closed = False
        self._timeout = False
        self._termination_reason = None
        self._terminal_reward_issued = False
        self._entry_execution = None

        self._precompute_timeline()

        max_steps = len(self._m5) - 1
        horizon_steps = horizon
        if horizon_steps is None:
            horizon_steps = self.config.horizon
        if horizon_steps is None:
            horizon_steps = self.config.max_episode_steps
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
        self._portfolio = Portfolio(
            instr,
            self.config.margin.initial_balance,
            account_currency=self.config.account_currency,
            converter=self._converter,
        )
        self._mark_at_observation(self._obs_idx)
        self._prev_equity = self._portfolio.equity
        self._initial_episode_equity = self._portfolio.equity

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

    def step(
        self, action: Action | int | float
    ) -> tuple[Observation, float, bool, bool, dict[str, object]]:
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
        episodic = self.config.reward.episodic_trade_reward_enabled
        exec_idx = int(self._exec_m1_idx[i])
        if exec_idx >= len(self._m1):
            # The dataset ended before another execution bar.  This is an
            # external data boundary, not the configured trading horizon.
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
        decision_equity = self._portfolio.equity

        # A single-trade episode has only three meaningful position actions:
        # open from flat, hold while open, and close with FLAT.  Directional
        # actions while a position is open are intentionally ignored, which
        # prevents both reversals and a second entry in the same episode.
        ignored_action = False
        if act.is_hold:
            target_units = current_units
        elif act.target_exposure == 0.0:
            target_units = Decimal(0)
        elif current_units != 0:
            target_units = current_units
            ignored_action = True
        else:
            target_units = self._target_units(act, exec_mid)

        delta = target_units - current_units
        report = None
        trade = None
        if delta != 0:
            report = self._engine.execute(exec_ts, exec_mid, delta, instrument=self._instrument)
            closing = current_units != 0 and target_units == 0
            trade = self._portfolio.adjust_to_target(
                target_units,
                report.execution_price,
                report.commission,
                timestamp=exec_ts,
                exit_reason="voluntary_close" if closing else None,
            )
            if current_units == 0 and target_units != 0:
                self._trade_opened = True
                self._entry_execution = report

        # Mark to market at the next observation's close, then apply financing.
        next_idx = i + 1
        self._mark_at_observation(next_idx)
        self._apply_financing(exec_ts, next_idx)
        self._steps_taken += 1

        terminated = False
        truncated = False
        liquidation = False
        forced = False
        exposure_before_forced_close = self._portfolio.snapshot().gross_exposure
        snap = self._margin.snapshot(
            equity=self._portfolio.equity,
            gross_exposure=exposure_before_forced_close,
        )
        if snap.liquidation and not self._portfolio.position.is_flat:
            liquidation = True
            forced = True
            report, trade = self._liquidate(
                snap, timestamp=pd.Timestamp(self._m5.iloc[next_idx][TIMESTAMP])
            )
            terminated = True
            self._termination_reason = "margin_liquidation"
        elif episodic and current_units != 0 and target_units == 0:
            # The requested close was executed at the decision-time execution
            # price above; terminate on that same transition.
            terminated = True
            self._voluntary_closed = True
            self._termination_reason = "voluntary_close"
        elif self._steps_taken >= self._horizon_steps:
            if not episodic:
                truncated = True
                if self.config.close_at_episode_end and not self._portfolio.position.is_flat:
                    forced = True
                    report, trade = self._close_at_current_mid(
                        timestamp=pd.Timestamp(self._m5.iloc[next_idx][TIMESTAMP]),
                        exit_reason="legacy_episode_end",
                    )
                # Legacy mode deliberately retains Gymnasium time-limit
                # semantics and per-step equity rewards.
                self._termination_reason = "legacy_horizon"
            else:
                terminated = True
                forced = not self._portfolio.position.is_flat
                self._timeout = forced
                if forced:
                    report, trade = self._close_at_current_mid(
                        timestamp=pd.Timestamp(self._m5.iloc[next_idx][TIMESTAMP]),
                        exit_reason="timeout",
                    )
                    self._termination_reason = "timeout_open_position"
                else:
                    self._termination_reason = "timeout_no_trade"

        if terminated and episodic:
            self._terminal_reward_issued = True
            # The fresh episode portfolio is the authoritative source for
            # the final account-currency result.  It includes the completed
            # trade's execution costs and any financing cash flow exactly
            # once; the trade ledger remains the diagnostic decomposition.
            net_trade_pnl = self._portfolio.realized_pnl
            if self._termination_reason == "timeout_no_trade":
                reward = self._reward.idle_reward()
            elif self._termination_reason == "timeout_open_position":
                reward = self._reward.timeout_reward(net_trade_pnl, self._initial_episode_equity)
            else:
                reward = self._reward.terminal_trade_reward(
                    net_trade_pnl, self._initial_episode_equity
                )
        elif episodic:
            reward = 0.0
        else:
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
        factor = self._portfolio.converter.quote_to_account_factor(
            self._instrument or "", exec_mid, self.config.account_currency
        )

        info["action_diagnostics"] = {
            "units_before": float(current_units),
            "units_after_policy": float(target_units),
            "units_after_step": float(self._portfolio.position.units),
            "position_changes": int(delta != 0),
            "actual_executions": int(report is not None),
            "sign_reversals": int(current_units * target_units < 0),
            "turnover": float(abs(delta) * exec_mid * factor / decision_equity)
            if decision_equity > 0
            else 0.0,
            "forced_executions": int(forced),
            "forced_turnover": float(exposure_before_forced_close / decision_equity)
            if forced and decision_equity > 0
            else 0.0,
            "ignored_action": int(ignored_action),
        }
        self._last_info = info
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------- properties

    def action_masks(self) -> np.ndarray:
        if self._portfolio is None:
            raise EnvironmentError("environment not reset")
        snap = self._portfolio.snapshot()
        units = snap.position.units
        exposure = float(snap.gross_exposure / snap.equity) if snap.equity > 0 else 0.0
        return valid_action_mask(-exposure if units < 0 else exposure, is_flat=units == 0)

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
        assert action.target_exposure is not None
        exposure = _dec(action.target_exposure)
        if sizing.mode == "fixed_units":
            return exposure * sizing.fixed_units
        # equity_fraction (Phase 3.1, account-currency aware):
        #   target_gross_exposure_account = |exposure| * equity
        #   units = target_gross_exposure_account / (price * quote_to_account_factor)
        #
        # For a USD-quote pair (EURUSD etc.) the quote->account factor is 1, so
        #   units = exposure * equity / price.
        # For a USD/XXX pair (USDJPY etc.) one base unit of USD == one USD of
        # notional, so units ~= exposure * equity (independent of the JPY price
        # scale).  The general formula below yields exactly this.
        equity = self._portfolio.equity
        target_exposure_acct = exposure * equity
        factor = self._portfolio.converter.quote_to_account_factor(
            self._instrument or "", exec_mid, self.config.account_currency
        )
        denom = exec_mid * factor
        if denom == 0:
            return Decimal(0)
        return target_exposure_acct / denom

    def _mark_at_observation(self, obs_idx: int) -> None:
        assert self._portfolio is not None and self._m5 is not None
        row = self._m5.iloc[obs_idx]
        mid = _dec(float(row[CLOSE]))
        if self.config.mtm_price == "bid_ask":
            prices = self._cost_model.execution_prices(mid, instrument=self._instrument)
            if self._portfolio.position.units > 0:
                self._portfolio.mark_to_market(prices.sell)
            else:
                self._portfolio.mark_to_market(prices.buy)
        else:
            self._portfolio.mark_to_market(mid)

    def _liquidate(
        self, snap: object, *, timestamp: pd.Timestamp
    ) -> tuple[ExecutionReport | None, TradeResult | None]:
        """Deterministically force-close the position at the current mark."""
        assert self._portfolio is not None
        mid = self._portfolio.current_mid
        if mid is None or self._portfolio.position.is_flat:
            return None, None
        report = self._engine.execute(
            timestamp,
            mid,
            -self._portfolio.position.units,
            instrument=self._instrument,
        )
        trade = self._portfolio.close_all(
            report.execution_price,
            report.commission,
            timestamp=timestamp,
            exit_reason="margin_liquidation",
        )
        self._portfolio.mark_to_market(mid)
        return report, trade

    def _close_at_current_mid(
        self, *, timestamp: pd.Timestamp, exit_reason: str
    ) -> tuple[ExecutionReport | None, TradeResult | None]:
        assert self._portfolio is not None
        mid = self._portfolio.current_mid
        if mid is None or self._portfolio.position.is_flat:
            return None, None
        report = self._engine.execute(
            timestamp,
            mid,
            -self._portfolio.position.units,
            instrument=self._instrument,
        )
        trade = self._portfolio.close_all(
            report.execution_price,
            report.commission,
            timestamp=timestamp,
            exit_reason=exit_reason,
        )
        self._portfolio.mark_to_market(mid)
        return report, trade

    def set_financing(self, model: FinancingModel) -> None:
        """Install a financing model (default :class:`ZeroCostFinancing`)."""
        self._financing = model

    def _apply_financing(self, exec_ts: pd.Timestamp, mark_obs_idx: int) -> None:
        """Charge overnight financing cost (account currency) on the open position.

        Phase 3.1 uses the zero-cost default.  The interface is wired through
        so real swap tables can be introduced later without redesign.
        """
        assert self._portfolio is not None and self._m5 is not None
        units = self._portfolio.position.units
        if units == 0:
            return
        mark_ts = pd.Timestamp(self._m5.iloc[mark_obs_idx][TIMESTAMP])
        hold_seconds = float((mark_ts - exec_ts) / pd.Timedelta(seconds=1))
        cost = self._financing.financing_cost(
            instrument_spec=self._portfolio.instrument_spec,
            units=units,
            entry_price=self._portfolio.position.entry_price,
            hold_seconds=hold_seconds,
            now=mark_ts,
            entry_time=exec_ts,
        )
        if cost:
            # Financing reduces equity directly through the balance.
            self._portfolio.apply_cash_adjustment(-cost)

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
        margin = self._margin.snapshot(
            equity=snap.equity,
            gross_exposure=snap.gross_exposure,
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
            account_currency=self.config.account_currency,
            base_currency=snap.base_currency,
            quote_currency=snap.quote_currency,
            raw_unrealized_pnl=self._portfolio.raw_unrealized_pnl,
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
        margin = self._margin.snapshot(
            equity=snap.equity,
            gross_exposure=snap.gross_exposure,
        )
        info: dict[str, object] = {
            "timestamp": obs.timestamp,
            "instrument": obs.instrument,
            "step_index": obs.step_index,
            # Currency metadata (Phase 3.1)
            "account_currency": self.config.account_currency,
            "base_currency": snap.base_currency,
            "quote_currency": snap.quote_currency,
            "raw_unrealized_pnl": self._portfolio.raw_unrealized_pnl,
            "conversion_rate": self._portfolio.conversion_rate,
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
            "units_delta": Decimal(0),
            "execution_commission": Decimal(0),
            "execution_price": None,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "liquidation": liquidation,
            "session": obs.time.session.value,
            "minutes_since_last_bar": obs.time.minutes_since_last_bar,
            "is_weekend_gap": obs.time.is_weekend_gap,
            # Single-trade episode diagnostics (kept out of observations).
            "episode_length": self._steps_taken,
            "position_opened": self._trade_opened,
            "voluntary_close": self._voluntary_closed,
            "timeout": self._timeout,
            "no_trade": self._termination_reason == "timeout_no_trade",
            "termination_reason": self._termination_reason,
            "terminal_reward": reward if terminated else 0.0,
            "initial_episode_equity": self._initial_episode_equity,
        }
        if trade is not None:
            if trade.executed_units:
                spread_cost = abs(trade.executed_units) * abs(
                    self._cost_model.execution_prices(
                        trade.execution_price, instrument=self._instrument
                    ).mid
                    - trade.execution_price
                )
            else:
                spread_cost = Decimal(0)
            # spread_cost is quote-currency; convert to account currency.
            spread_cost_acct = spread_cost * self._portfolio.converter.quote_to_account_factor(
                self._instrument or "",
                trade.execution_price,
                self.config.account_currency,
            )
            info["trade_cost"] = trade.commission + spread_cost_acct
            info["execution_price"] = trade.execution_price
            info["trade_direction"] = trade.direction
            info["units_delta"] = trade.units_delta
            info["trade_realized_pnl"] = trade.realized_pnl
            info["raw_pnl"] = trade.raw_pnl
            info["raw_pnl_currency"] = trade.raw_pnl_currency
            info["converted_pnl"] = trade.converted_pnl
            info["trade_conversion_rate"] = trade.conversion_rate
            info["commission_raw"] = trade.commission_raw
        if execution is not None:
            info["execution_mid"] = execution.mid_price
            info["execution_price"] = execution.execution_price
            info["execution_commission"] = execution.commission

        last_trade = self._portfolio.last_trade
        active_trade = self._portfolio.active_trade
        diagnostic_trade = last_trade or active_trade
        if diagnostic_trade is not None:
            info["trade_id"] = diagnostic_trade.trade_id
            info["trade_direction"] = diagnostic_trade.direction
            info["entry_timestamp"] = diagnostic_trade.entry_timestamp
            info["entry_price"] = diagnostic_trade.entry_price
            info["entry_units"] = diagnostic_trade.entry_units
            info["entry_exposure"] = diagnostic_trade.entry_units * diagnostic_trade.entry_price
            info["entry_costs"] = diagnostic_trade.entry_costs
            info["exit_timestamp"] = diagnostic_trade.exit_timestamp
            info["exit_price"] = diagnostic_trade.exit_price
            info["exit_costs"] = diagnostic_trade.exit_costs
            info["gross_trade_pnl"] = diagnostic_trade.gross_realized_pnl
            info["net_trade_pnl"] = (
                self._portfolio.realized_pnl
                if last_trade is not None
                else diagnostic_trade.net_realized_pnl
            )
            info["transaction_costs"] = diagnostic_trade.transaction_costs
            info["exit_reason"] = diagnostic_trade.exit_reason
        else:
            info["gross_trade_pnl"] = Decimal(0)
            info["net_trade_pnl"] = Decimal(0)
            info["transaction_costs"] = Decimal(0)
        return info


def make_environment(
    dataset: MarketDataset | InstrumentData,
    config: EnvironmentConfig,
    *,
    instrument: str | None = None,
) -> ForexEnvironment:
    """Convenience factory for :class:`ForexEnvironment`."""
    return ForexEnvironment(dataset, config, instrument=instrument)
