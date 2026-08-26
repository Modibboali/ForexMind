"""Tests for temporal dataset splitting (Phase 2)."""

from __future__ import annotations

import pandas as pd
import pytest
from forexmind.data.splits import SplitConfig, SplitDataset, SplitError

from tests.synthetic import (
    make_instrument,
    make_split_dataset,
    make_test_split_config,
    timeline_m5,
)


def _two_instruments() -> SplitDataset:
    dates = ["2020-01-06", "2020-06-01", "2021-03-01", "2021-09-01", "2022-03-01", "2022-09-01"]
    eur_m5 = timeline_m5(dates, per_day=20)
    gbp_m5 = timeline_m5(dates, per_day=20)
    return make_split_dataset(
        {
            "EURUSD": make_instrument("EURUSD", eur_m5),
            "GBPUSD": make_instrument("GBPUSD", gbp_m5),
        }
    )


def test_default_split_config_chronological() -> None:
    cfg = SplitConfig.default()
    cfg.validate()  # must not raise
    assert cfg.range("train") == (pd.Timestamp("2006-01-01"), pd.Timestamp("2019-01-01"))
    assert cfg.range("validation")[0] == pd.Timestamp("2019-01-01")
    assert cfg.range("test") == (pd.Timestamp("2022-01-01"), pd.Timestamp("2026-01-01"))


def test_invalid_split_config_rejected() -> None:
    with pytest.raises(SplitError):
        SplitConfig(
            train_start=pd.Timestamp("2021-01-01"),
            train_end=pd.Timestamp("2019-01-01"),
            validation_start=pd.Timestamp("2019-01-01"),
            validation_end=pd.Timestamp("2022-01-01"),
            test_start=pd.Timestamp("2022-01-01"),
            test_end=pd.Timestamp("2026-01-01"),
        ).validate()
    with pytest.raises(SplitError):
        SplitConfig(
            train_start=pd.Timestamp("2018-01-01"),
            train_end=pd.Timestamp("2020-06-01"),
            validation_start=pd.Timestamp("2019-01-01"),
            validation_end=pd.Timestamp("2022-01-01"),
            test_start=pd.Timestamp("2022-01-01"),
            test_end=pd.Timestamp("2026-01-01"),
        ).validate()  # train overlaps validation


def test_split_config_serialization() -> None:
    cfg = make_test_split_config()
    assert SplitConfig.from_dict(cfg.to_dict()) == cfg


def test_every_instrument_split_independently() -> None:
    ds = _two_instruments()
    for instr in ("EURUSD", "GBPUSD"):
        train = ds.train(instr)
        val = ds.validation(instr)
        test = ds.test(instr)
        assert train.n_bars > 0 and val.n_bars > 0 and test.n_bars > 0
        assert train.last_index < val.first_index
        assert val.last_index < test.first_index


def test_split_boundary_timestamps() -> None:
    ds = _two_instruments()
    for instr in ("EURUSD", "GBPUSD"):
        ts = ds.m5_timestamps(instr)
        train = ds.train(instr)
        val = ds.validation(instr)
        test = ds.test(instr)
        assert ts[train.first_index] >= np_datetime(train.start_ts)
        assert ts[val.first_index] >= np_datetime(val.start_ts)
        assert ts[test.first_index] >= np_datetime(test.start_ts)
        # Max train < min validation < min test.
        assert ts[train.last_index] < ts[val.first_index]
        assert ts[val.last_index] < ts[test.first_index]


def np_datetime(ts: pd.Timestamp) -> object:
    return ts.to_datetime64()


def test_integrity_report() -> None:
    ds = _two_instruments()
    report = ds.verify_integrity(strict=True)
    assert report["ok"] is True
    assert set(report["instruments"]) == {"EURUSD", "GBPUSD"}  # type: ignore[union-attr]


def test_instrument_with_no_rows_in_split_fails() -> None:
    # m5 only in 2020 -> train has rows, validation/test have none.
    m5 = timeline_m5(["2020-01-06", "2020-06-01"], per_day=20)
    ds = make_split_dataset({"EURUSD": make_instrument("EURUSD", m5)})
    with pytest.raises(SplitError, match="no rows"):
        ds.validation("EURUSD")
    with pytest.raises(SplitError):
        ds.verify_integrity(strict=True)


def test_split_manifest_machine_readable() -> None:
    ds = _two_instruments()
    manifest = ds.manifest()
    assert "split_config" in manifest
    assert "instruments" in manifest
    assert manifest["ok"] is True  # type: ignore[union-attr]


def test_periods_per_year_estimate() -> None:
    ds = _two_instruments()
    ppy = ds.periods_per_year("train")
    assert ppy > 0
