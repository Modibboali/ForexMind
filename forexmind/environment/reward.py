"""Reward service.

Phase 1 reward is based on the change in account equity rather than raw price
movement, with transaction costs already reflected in equity::

    reward_t = ln(equity_{t+1} / equity_t)

The service is extensible so later experiments can compare log return,
risk-adjusted return, drawdown-penalised return, and cost-penalised return.
"""

from __future__ import annotations

from decimal import Decimal

from forexmind.config import RewardConfig, _dec


class RewardService:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config

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

    @staticmethod
    def _log_return(prev: Decimal, curr: Decimal) -> float:
        if prev <= 0:
            raise ValueError(f"prev_equity must be > 0 for log return, got {prev}")
        if curr <= 0:
            # Equity collapsed (e.g. liquidation wipe-out); represent as a large
            # negative return, matching the log-return limit.
            return float("-inf")
        return float((curr / prev).ln())
