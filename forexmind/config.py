"""Central configuration for the ForexMind simulator.

All execution, margin, reward, and environment assumptions are explicit and
configuration-driven.  Nothing here is derived from the historical OHLC data;
in particular the execution-cost parameters (spread / slippage / commission)
are *model assumptions* because the raw dataset contains no bid/ask or tick
data (see README).
"""

from __future__ import annotations

import decimal
from dataclasses import dataclass, field
from decimal import Decimal

# Fixed decimal precision for all accounting.  Fixing this makes results
# deterministic across machines/runs.
DECIMAL_PRECISION = 50
decimal.getcontext().prec = DECIMAL_PRECISION


def _dec(value: float | int | str | Decimal) -> Decimal:
    """Exact conversion from float/str/int to Decimal.

    ``Decimal(str(x))`` is used for floats so the decimal string representation
    is preserved exactly rather than the binary expansion.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


# ---------------------------------------------------------------------------
# Execution-cost assumptions (model parameters, NOT historical observations)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionConfig:
    """Deterministic execution-cost assumptions.

    ``spread_mode``: only ``"fixed"`` is supported in Phase 1.  A fixed
    absolute spread (in price units) is applied symmetrically around the mid:
    ``buy = mid + spread/2``, ``sell = mid - spread/2``.

    ``slippage_mode``: ``"none"`` or ``"fixed"``.  When fixed:
    ``buy += slippage``, ``sell -= slippage``.

    ``commission_per_unit``: quote-currency cost per base unit traded, charged
    on every execution side (open and close).
    """

    spread_mode: str = "fixed"
    spread_value: float = 0.0
    slippage_mode: str = "none"
    slippage_value: float = 0.0
    commission_per_unit: float = 0.0

    def __post_init__(self) -> None:
        if self.spread_mode != "fixed":
            raise ValueError(
                f"unsupported spread_mode {self.spread_mode!r}; only 'fixed' is supported"
            )
        if self.slippage_mode not in ("none", "fixed"):
            raise ValueError(
                f"unsupported slippage_mode {self.slippage_mode!r}; use 'none' or 'fixed'"
            )
        if self.spread_value < 0:
            raise ValueError("spread_value must be >= 0")
        if self.slippage_value < 0:
            raise ValueError("slippage_value must be >= 0")
        if self.commission_per_unit < 0:
            raise ValueError("commission_per_unit must be >= 0")

    @classmethod
    def from_pips(
        cls,
        pip_size: float,
        spread_pips: float = 0.0,
        slippage_pips: float = 0.0,
        commission_per_unit: float = 0.0,
    ) -> ExecutionConfig:
        """Build a config with spread/slippage expressed in pips."""
        return cls(
            spread_mode="fixed",
            spread_value=spread_pips * pip_size,
            slippage_mode="fixed" if slippage_pips else "none",
            slippage_value=slippage_pips * pip_size,
            commission_per_unit=commission_per_unit,
        )

    @property
    def spread_decimal(self) -> Decimal:
        return _dec(self.spread_value)

    @property
    def slippage_decimal(self) -> Decimal:
        return _dec(self.slippage_value)

    @property
    def commission_decimal(self) -> Decimal:
        return _dec(self.commission_per_unit)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionSizingConfig:
    """Maps a target exposure in [-1, +1] to base-currency units.

    ``mode="equity_fraction"``: ``units = exposure * equity / execution_price``
    (exposure is a fraction of current equity).
    ``mode="fixed_units"``: ``units = exposure * fixed_units`` (a fixed notional
    position in base units scaled by exposure).
    """

    mode: str = "equity_fraction"
    fixed_units: Decimal = field(default_factory=lambda: Decimal("100000"))

    def __post_init__(self) -> None:
        if self.mode not in ("equity_fraction", "fixed_units"):
            raise ValueError(
                f"unsupported sizing mode {self.mode!r}; use 'equity_fraction' or 'fixed_units'"
            )
        if self.fixed_units <= 0:
            raise ValueError("fixed_units must be > 0")


# ---------------------------------------------------------------------------
# Margin / leverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarginConfig:
    """Configuration-driven margin model.

    ``margin_requirement`` is the fraction of gross exposure reserved as
    margin; defaults to ``1 / leverage`` when ``None``.
    ``maintenance_margin_ratio`` is the fraction of ``margin_used`` below which
    the equity may not fall; when ``equity <= margin_used * ratio`` the
    position is deterministically liquidated.
    ``max_leverage`` (optional) caps gross exposure / equity; exceeding it
    triggers a margin-call style block.
    """

    initial_balance: Decimal = field(default_factory=lambda: Decimal("10000"))
    leverage: Decimal = field(default_factory=lambda: Decimal("100"))
    margin_requirement: Decimal | None = None
    maintenance_margin_ratio: Decimal = field(default_factory=lambda: Decimal("0.5"))
    max_leverage: Decimal | None = None

    def __post_init__(self) -> None:
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be > 0")
        if self.leverage <= 0:
            raise ValueError("leverage must be > 0")
        if self.maintenance_margin_ratio <= 0 or self.maintenance_margin_ratio > 1:
            raise ValueError("maintenance_margin_ratio must be in (0, 1]")
        if self.margin_requirement is not None and self.margin_requirement <= 0:
            raise ValueError("margin_requirement must be > 0")
        if self.max_leverage is not None and self.max_leverage <= 0:
            raise ValueError("max_leverage must be > 0")

    @property
    def effective_margin_requirement(self) -> Decimal:
        if self.margin_requirement is not None:
            return self.margin_requirement
        return Decimal(1) / self.leverage


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardConfig:
    """Reward service configuration.

    ``reward_type``: ``"log_equity_return"`` (default) computes
    ``ln(equity_{t+1} / equity_t)``.  Extensible for later experiments
    (risk-adjusted, drawdown-penalised, cost-penalised).
    """

    reward_type: str = "log_equity_return"
    risk_free_rate: Decimal = field(default_factory=lambda: Decimal("0"))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvironmentConfig:
    """Top-level environment configuration."""

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    margin: MarginConfig = field(default_factory=MarginConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)

    decision_interval_minutes: int = 5
    # Fixed Phase-1 execution convention: the next M1 bar's open after the
    # observed M5 close (see README, "No-lookahead policy").
    execution_timing: str = "next_m1_open"
    mtm_price: str = "mid"  # "mid" or "bid_ask" (unrealized PnL mark price)
    close_at_episode_end: bool = False
    horizon: int | None = None  # max steps per episode; None = use full data
    observation_window: int = 12  # recent M5 bars exposed in the observation

    def __post_init__(self) -> None:
        if self.decision_interval_minutes <= 0:
            raise ValueError("decision_interval_minutes must be > 0")
        if self.execution_timing != "next_m1_open":
            raise ValueError(
                f"unsupported execution_timing {self.execution_timing!r}; "
                "only 'next_m1_open' is supported"
            )
        if self.mtm_price not in ("mid", "bid_ask"):
            raise ValueError("mtm_price must be 'mid' or 'bid_ask'")
        if self.horizon is not None and self.horizon <= 0:
            raise ValueError("horizon must be > 0 or None")
        if self.observation_window <= 0:
            raise ValueError("observation_window must be > 0")


def default_config(
    *,
    initial_balance: float | str | Decimal = "10000",
    leverage: float | str | Decimal = 100,
    spread_value: float = 0.0002,
    slippage_value: float = 0.0,
    commission_per_unit: float = 0.0,
    sizing_mode: str = "equity_fraction",
    fixed_units: float | str | Decimal = "100000",
) -> EnvironmentConfig:
    """Convenience factory for a sensible default Phase-1 configuration."""
    return EnvironmentConfig(
        execution=ExecutionConfig(
            spread_mode="fixed",
            spread_value=spread_value,
            slippage_mode="fixed" if slippage_value else "none",
            slippage_value=slippage_value,
            commission_per_unit=commission_per_unit,
        ),
        margin=MarginConfig(initial_balance=_dec(initial_balance), leverage=_dec(leverage)),
        sizing=PositionSizingConfig(mode=sizing_mode, fixed_units=_dec(fixed_units)),
    )
