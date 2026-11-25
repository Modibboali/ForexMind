"""Reward calculations for legacy and single-trade episodes.

The legacy equity-return method remains available for older callers.  The
episodic environment uses :meth:`terminal_trade_reward`, so intermediate
market/equity changes never become policy rewards.
"""

from __future__ import annotations

import math
from decimal import Decimal

from forexmind.config import RewardConfig, _dec


class RewardService:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        # Diagnostic: number of equity-collapse events mapped to the finite
        # reward floor instead of -inf (see RewardConfig.min_reward).
        self._collapse_count = 0

    @property
    def collapse_count(self) -> int:
        """Number of times ``curr_equity <= 0`` mapped to the finite floor."""
        return self._collapse_count

    def reward(
        self,
        prev_equity: float | str | Decimal,
        curr_equity: float | str | Decimal,
    ) -> float:
        """Compute the scalar reward for a transition ``prev_equity -> curr_equity``."""
        prev = _dec(prev_equity)
        curr = _dec(curr_equity)
        if self.config.reward_type == "log_equity_return":
            return self._log_return(prev, curr)
        raise ValueError(
            f"unsupported reward_type {self.config.reward_type!r}; use 'log_equity_return'"
        )

    def terminal_trade_reward(
        self,
        net_trade_pnl: float | str | Decimal,
        initial_episode_equity: float | str | Decimal,
    ) -> float:
        """Return normalized net P&L for a completed trade.

        ``net_trade_pnl`` is supplied by portfolio accounting and therefore
        already includes the applicable execution costs exactly once.
        ``initial_episode_equity`` is captured at reset and is never replaced
        by post-trade equity.
        """
        pnl = _dec(net_trade_pnl)
        equity = _dec(initial_episode_equity)
        if equity <= 0:
            raise ValueError("initial_episode_equity must be > 0")
        if not pnl.is_finite():
            self._collapse_count += 1
            return float(self.config.min_reward)
        result = float(pnl / equity)
        if not math.isfinite(result):
            self._collapse_count += 1
            return float(self.config.min_reward)
        if self.config.max_reward is not None:
            result = min(result, float(self.config.max_reward))
        return result

    def idle_reward(self) -> float:
        """Terminal reward for an episode in which no trade was opened."""
        return -float(self.config.idle_penalty)

    def timeout_reward(
        self,
        net_trade_pnl: float | str | Decimal,
        initial_episode_equity: float | str | Decimal,
    ) -> float:
        """Normalized terminal trade return minus the timeout adjustment."""
        result = self.terminal_trade_reward(net_trade_pnl, initial_episode_equity)
        result -= float(self.config.timeout_penalty)
        return result

    def _log_return(self, prev: Decimal, curr: Decimal) -> float:
        if prev <= 0:
            raise ValueError(f"prev_equity must be > 0 for log return, got {prev}")
        if curr <= 0:
            # Equity collapsed (e.g. liquidation wipe-out).  The log-return
            # limit is -inf, but -inf is unusable for gradient-based RL: it
            # poisons GAE deltas and advantage normalization into NaN.  Return
            # the configured finite floor instead (default -50.0, far below any
            # real log return).  This changes no trading semantics - only the
            # scalar that enters the optimizer.
            self._collapse_count += 1
            return float(self.config.min_reward)
        return float((curr / prev).ln())
