"""Tests for the robust tabular loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from forexmind.data.loaders import (
    load_any,
    load_many_concat,
    load_parquet,
    load_tabular,
)
from forexmind.data.schema import SchemaError

MT5_CSV = """2025.01.06,00:00,1.10000,1.10100,1.09900,1.10050,0
2025.01.06,00:01,1.10050,1.10150,1.10000,1.10100,0
2025.01.06,00:02,1.10100,1.10200,1.10050,1.10150,0
"""

HEADER_CSV = """Date,Time,Open,High,Low,Close,Volume
2025.01.06,00:00,1.10000,1.10100,1.09900,1.10050,0
2025.01.06,00:01,1.10050,1.10150,1.10000,1.10100,0
"""

TSV_DATA = """timestamp\topen\thigh\tlow\tclose
2025-01-06 00:00\t1.10000\t1.10100\t1.09900\t1.10050
2025-01-06 00:01\t1.10050\t1.10150\t1.10000\t1.10100
"""

SINGLE_TS_COLUMN = """Time,Open,High,Low,Close
2025-01-06 00:00,1.10000,1.10100,1.09900,1.10050
2025-01-06 00:01,1.10050,1.10150,1.10000,1.10100
"""


@pytest.fixture
def tmp_files(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, content in {
        "mt5.csv": MT5_CSV,
        "header.csv": HEADER_CSV,
        "data.tsv": TSV_DATA,
        "single_ts.csv": SINGLE_TS_COLUMN,
    }.items():
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        paths[name] = p
    return paths


def _assert_canonical(result) -> None:
    df = result.frame
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["open"].dtype == "float64"


def test_mt5_no_header(tmp_files: dict[str, Path]) -> None:
    result = load_tabular(tmp_files["mt5.csv"])
    _assert_canonical(result)
    assert len(result.frame) == 3
    assert result.frame["timestamp"].iloc[0] == pd.Timestamp("2025-01-06 00:00")
    assert result.frame["close"].iloc[-1] == 1.10150
    assert result.source_timezone is None  # ambiguous -> not guessed


def test_header_csv_variants(tmp_files: dict[str, Path]) -> None:
    result = load_tabular(tmp_files["header.csv"])
    _assert_canonical(result)
    assert len(result.frame) == 2


def test_tsv_auto_sep(tmp_files: dict[str, Path]) -> None:
    result = load_tabular(tmp_files["data.tsv"])
    _assert_canonical(result)
    assert len(result.frame) == 2


def test_single_timestamp_column(tmp_files: dict[str, Path]) -> None:
    result = load_tabular(tmp_files["single_ts.csv"])
    _assert_canonical(result)
    assert result.frame["timestamp"].iloc[1] == pd.Timestamp("2025-01-06 00:01")


def test_missing_required_column(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("Date,Time,Open,High,Close\n2025.01.06,00:00,1.1,1.2,1.15\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="low"):
        load_tabular(p)


def test_no_timestamp_source(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("Open,High,Low,Close\n1.1,1.2,1.0,1.15\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="timestamp"):
        load_tabular(p)


def test_ambiguous_duplicate_open(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text(
        "Time,Open,O,High,Low,Close\n2025.01.06 00:00,1.1,1.2,1.3,1.0,1.15\n",
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="ambiguous"):
        load_tabular(p)


def test_date_only_without_time(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("Date,Open,High,Low,Close\n2025.01.06,1.1,1.2,1.0,1.15\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_tabular(p)


def test_date_full_with_separate_time_is_ambiguous(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text(
        "Date,Time,Open,High,Low,Close\n2025.01.06 00:00:00,00:01,1.1,1.2,1.0,1.15\n",
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="ambiguous"):
        load_tabular(p)


def test_headerless_non_mt5_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("foo,bar,baz\n1,2,3\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_tabular(p)


def test_parquet_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    src.write_text(HEADER_CSV, encoding="utf-8")
    df = load_tabular(src).frame
    pq = tmp_path / "out.parquet"
    df.to_parquet(pq, index=False)
    result = load_parquet(pq)
    _assert_canonical(result)
    assert len(result.frame) == 2
    assert result.frame["timestamp"].iloc[0] == pd.Timestamp("2025-01-06 00:00")


def test_load_any_dispatch(tmp_files: dict[str, Path], tmp_path: Path) -> None:
    assert len(load_any(tmp_files["mt5.csv"]).frame) == 3
    pq = tmp_path / "x.parquet"
    load_tabular(tmp_files["header.csv"]).frame.to_parquet(pq, index=False)
    assert len(load_any(pq).frame) == 2


def test_load_many_concat(tmp_path: Path) -> None:
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    p1.write_text(
        "2025.01.06,00:00,1.10000,1.10100,1.09900,1.10050,0\n"
        "2025.01.06,00:01,1.10050,1.10150,1.10000,1.10100,0\n",
        encoding="utf-8",
    )
    p2.write_text(
        "2025.01.06,00:02,1.10100,1.10200,1.10050,1.10150,0\n",
        encoding="utf-8",
    )
    result = load_many_concat([p1, p2])
    assert len(result.frame) == 3
    assert result.frame["timestamp"].is_monotonic_increasing


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_tabular("does_not_exist.csv")
