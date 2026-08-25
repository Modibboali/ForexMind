"""Instrument-aware market dataset abstraction.

Each instrument owns an independent chronological timeline.  Bars from
different instruments are never concatenated as adjacent temporal
observations; every episode operates within a single instrument.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import pandas as pd

from forexmind.data.resampler import (
    CompletenessPolicy,
    ResampleConfig,
    resample_m1_to_m5,
)
from forexmind.data.schema import TIMESTAMP


@dataclass(frozen=True)
class InstrumentMeta:
    """Provenance and quality metadata for one instrument."""

    instrument: str
    source_file: str | None = None
    row_count: int = 0
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None
    timezone: str | None = None  # "UTC (explicit)" or None (unknown/server time)
    missing_bar_stats: Mapping[str, int] = field(default_factory=dict)
    m5_row_count: int | None = None
    validation_status: str = "unvalidated"
    validation_details: tuple[str, ...] = ()


@dataclass
class InstrumentData:
    """Canonical M1 data plus derived M5 data for a single instrument."""

    instrument: str
    m1: pd.DataFrame
    m5: pd.DataFrame
    meta: InstrumentMeta | None = None

    @classmethod
    def from_m1(
        cls,
        instrument: str,
        m1: pd.DataFrame,
        resample_config: ResampleConfig | None = None,
        meta: InstrumentMeta | None = None,
    ) -> InstrumentData:
        cfg = resample_config or ResampleConfig()
        m5 = resample_m1_to_m5(m1, cfg)
        return cls(instrument=instrument, m1=m1.reset_index(drop=True), m5=m5, meta=meta)

    @property
    def first_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.m1[TIMESTAMP].iloc[0])

    @property
    def last_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.m1[TIMESTAMP].iloc[-1])


class MarketDataset:
    """Collection of per-instrument timelines.

    ``instruments`` are stored in insertion order.  Lookups are by exact
    instrument name (case-insensitive, uppercased internally).
    """

    def __init__(self, instruments: Mapping[str, InstrumentData] | None = None) -> None:
        self._data: dict[str, InstrumentData] = {}
        for _name, data in (instruments or {}).items():
            self.add(data)

    def add(self, data: InstrumentData) -> None:
        key = data.instrument.upper()
        if key in self._data:
            raise ValueError(f"duplicate instrument in dataset: {data.instrument}")
        self._data[key] = data

    @property
    def instruments(self) -> list[str]:
        """Instrument names (upper-case) in insertion order."""
        return list(self._data)

    def get(self, instrument: str) -> InstrumentData:
        key = instrument.upper()
        if key not in self._data:
            raise KeyError(f"unknown instrument {instrument!r}; available: {self.instruments}")
        return self._data[key]

    def __contains__(self, instrument: str) -> bool:
        return instrument.upper() in self._data

    def __iter__(self) -> Iterator[InstrumentData]:
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def m1(self, instrument: str) -> pd.DataFrame:
        return self.get(instrument).m1

    def m5(self, instrument: str) -> pd.DataFrame:
        return self.get(instrument).m5

    def build_m5(
        self,
        resample_config: ResampleConfig | None = None,
        completeness: CompletenessPolicy | None = None,
    ) -> None:
        """(Re)build the M5 frames for every instrument."""
        cfg = resample_config or ResampleConfig()
        if completeness is not None:
            cfg = ResampleConfig(
                bucket_minutes=cfg.bucket_minutes,
                expected_bars=cfg.expected_bars,
                completeness=completeness,
            )
        for _key, data in self._data.items():
            data.m5 = resample_m1_to_m5(data.m1, cfg)
            if data.meta is not None:
                data.meta = InstrumentMeta(**{**data.meta.__dict__, "m5_row_count": len(data.m5)})

    def summary(self) -> str:
        lines = [f"MarketDataset with {len(self)} instrument(s)"]
        for data in self._data.values():
            lines.append(
                f"  {data.instrument:<8} M1 rows={len(data.m1):,} "
                f"M5 rows={len(data.m5):,} "
                f"{data.first_timestamp} .. {data.last_timestamp}"
            )
        return "\n".join(lines)
