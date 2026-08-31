"""Shared helpers for ForexMind tool scripts."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from forexmind.config import (
    EnvironmentConfig,
    ExecutionConfig,
    MarginConfig,
    PositionSizingConfig,
    RewardConfig,
)
from forexmind.data.dataset import InstrumentData
from forexmind.data.splits import SplitConfig, SplitDataset
from forexmind.observation.encoder import EncoderConfig
from forexmind.observation.normalization import NormalizerConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "historical_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

# Instruments supplied in the raw dataset (Phase 1).
INSTRUMENTS: tuple[str, ...] = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
)


def instrument_files(instrument: str) -> list[Path]:
    """Return the sorted raw source files for one instrument."""
    d = RAW_DATA_DIR / instrument.upper()
    if not d.is_dir():
        raise FileNotFoundError(f"no raw data directory for instrument {instrument}: {d}")
    return sorted(d.glob("*.csv"))


def load_processed_instrument(instrument: str) -> InstrumentData:
    """Load a processed instrument (m1.parquet + m5.parquet) into InstrumentData."""
    key = instrument.upper()
    d = PROCESSED_DATA_DIR / key
    m1_path = d / "m1.parquet"
    m5_path = d / "m5.parquet"
    if not (m1_path.is_file() and m5_path.is_file()):
        raise FileNotFoundError(
            f"processed data missing for {key}; run `python -m tools.process_data` first "
            f"(looked in {d})"
        )
    m1 = pd.read_parquet(m1_path)
    m5 = pd.read_parquet(m5_path)
    return InstrumentData(instrument=key, m1=m1, m5=m5)


def make_split_dataset(
    split_config: SplitConfig | None = None,
    instruments: tuple[str, ...] = INSTRUMENTS,
) -> SplitDataset:
    """Build a SplitDataset backed by the processed parquet files."""
    cfg = split_config or SplitConfig.default()
    return SplitDataset(cfg, load_processed_instrument, instruments)


def environment_config_from_dict(d: dict[str, Any]) -> EnvironmentConfig:
    """Reconstruct an EnvironmentConfig from a report-style dict."""
    ex = dict(d["execution"])
    mg = dict(d["margin"])
    rw = dict(d["reward"])
    sz = dict(d["sizing"])
    return EnvironmentConfig(
        execution=ExecutionConfig(
            spread_mode=str(ex["spread_mode"]),
            spread_value=float(ex["spread_value"]),
            slippage_mode=str(ex["slippage_mode"]),
            slippage_value=float(ex["slippage_value"]),
            commission_per_unit=float(ex["commission_per_unit"]),
            instrument_spreads={
                k: float(v) for k, v in (ex.get("instrument_spreads") or {}).items()
            },
        ),
        margin=MarginConfig(
            initial_balance=Decimal(str(mg["initial_balance"])),
            leverage=Decimal(str(mg["leverage"])),
            maintenance_margin_ratio=Decimal(str(mg["maintenance_margin_ratio"])),
            max_leverage=Decimal(str(mg["max_leverage"]))
            if mg.get("max_leverage") is not None
            else None,
        ),
        reward=RewardConfig(reward_type=str(rw["reward_type"])),
        sizing=PositionSizingConfig(
            mode=str(sz["mode"]), fixed_units=Decimal(str(sz["fixed_units"]))
        ),
        account_currency=str(d.get("account_currency", "USD")),
        decision_interval_minutes=int(d["decision_interval_minutes"]),
        execution_timing=str(d["execution_timing"]),
        mtm_price=str(d["mtm_price"]),
        close_at_episode_end=bool(d["close_at_episode_end"]),
        horizon=None,
        observation_window=int(d["observation_window"]),
    )


def encoder_config_from_dict(d: dict[str, Any]) -> EncoderConfig:
    """Reconstruct an EncoderConfig from a report-style dict."""
    nz = dict(d.get("normalizer") or {})
    return EncoderConfig(
        context_length=int(d["context_length"]),
        market_features=tuple(d["market_features"]),
        initial_balance=d["initial_balance"],
        instrument_order=tuple(d["instrument_order"]),
        dtype=str(d["dtype"]),
        max_leverage_feature=float(d["max_leverage_feature"]),
        normalizer=NormalizerConfig(
            market=str(nz.get("market", "identity")),
            account=str(nz.get("account", "identity")),
            time=str(nz.get("time", "identity")),
        ),
    )
