"""End-to-end test on real MetaTrader data (skipped when data is absent).

Only a small slice of a real file is used so the test stays fast while still
exercising the real MT5 CSV format.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from forexmind.config import (
    EnvironmentConfig,
    ExecutionConfig,
    MarginConfig,
    PositionSizingConfig,
    RewardConfig,
)
from forexmind.data.dataset import InstrumentData, MarketDataset
from forexmind.data.loaders import LoadConfig, load_tabular
from forexmind.data.validator import MarketDataValidator
from forexmind.environment import ForexEnvironment
from tools.common import RAW_DATA_DIR

_REAL_FILE = RAW_DATA_DIR / "EURUSD" / "DAT_MT_EURUSD_M1_2025.csv"

pytestmark = pytest.mark.skipif(
    not _REAL_FILE.is_file(),
    reason="real MT5 data not present",
)


@pytest.fixture
def real_slice(tmp_path: Path) -> Path:
    """A slice of a real MT5 M1 file (first 30000 data rows).

    The first 3000 rows (New Year's Day thin session) contain almost no
    complete 5-minute runs, so a larger slice is used to guarantee STRICT M5
    bars from normal trading days.
    """
    out = tmp_path / "eurusd_slice.csv"
    with _REAL_FILE.open("r", encoding="utf-8") as fh:
        lines = [next(fh) for _ in range(30000)]
    out.write_text("".join(lines), encoding="utf-8")
    return out


def test_real_mt5_format_full_pipeline(real_slice: Path) -> None:
    res = load_tabular(real_slice, LoadConfig(sep=",", has_header=False))
    assert list(res.frame.columns) == ["timestamp", "open", "high", "low", "close"]
    assert res.source_timezone is None  # server time: not guessed

    vresult = MarketDataValidator().validate(res.frame, "EURUSD")
    assert vresult.is_valid, [str(e) for e in vresult.errors]

    data = InstrumentData.from_m1("EURUSD", res.frame)
    assert len(data.m5) > 0

    cfg = EnvironmentConfig(
        execution=ExecutionConfig(spread_value=0.0002),
        margin=MarginConfig(initial_balance=Decimal("10000"), leverage=Decimal("100")),
        sizing=PositionSizingConfig(mode="fixed_units", fixed_units=Decimal("10000")),
        reward=RewardConfig(),
        horizon=5,
    )
    ds = MarketDataset()
    ds.add(data)
    env = ForexEnvironment(ds, cfg)
    obs, info = env.reset(seed=0, start_index=0)
    assert obs.account.equity == Decimal("10000")
    for a in [4, 3, 2, 1, 0]:
        _obs, _reward, _term, _trunc, info = env.step(a)
    assert "equity" in info
