"""Temporal dataset splitting (Phase 2).

Splits are strictly chronological, never random::

    train < validation < test

within every instrument.  Exact timestamps are used internally (year labels
are only conventions).  The default experiment periods are::

    TRAIN       2006-01-01 .. 2019-01-01
    VALIDATION  2019-01-01 .. 2022-01-01
    TEST        2022-01-01 .. 2026-01-01

Note: source timestamps are MT5 server time (unknown offset, see Phase 1), so
split boundaries live in the same timezone convention.  No timezone guess is
made.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from forexmind.data.dataset import InstrumentData
from forexmind.data.schema import TIMESTAMP

DEFAULT_INSTRUMENT_ORDER: tuple[str, ...] = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
)

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")


class SplitError(ValueError):
    """Raised for invalid split configurations or failed integrity checks."""


@dataclass(frozen=True)
class SplitConfig:
    """Explicit, serializable train/validation/test boundaries (half-open)."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @classmethod
    def default(cls) -> SplitConfig:
        """Default Phase-2 periods: 2006-2018 / 2019-2021 / 2022-2025."""
        return cls(
            train_start=pd.Timestamp("2006-01-01"),
            train_end=pd.Timestamp("2019-01-01"),
            validation_start=pd.Timestamp("2019-01-01"),
            validation_end=pd.Timestamp("2022-01-01"),
            test_start=pd.Timestamp("2022-01-01"),
            test_end=pd.Timestamp("2026-01-01"),
        )

    def range(self, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the half-open ``[start, end)`` range for a split name."""
        ranges = {
            "train": (self.train_start, self.train_end),
            "validation": (self.validation_start, self.validation_end),
            "test": (self.test_start, self.test_end),
        }
        if split not in ranges:
            raise SplitError(f"unknown split {split!r}; expected {list(ranges)}")
        return ranges[split]

    def validate(self) -> None:
        """Verify strict chronology and non-overlap; raise on violation.

        Boundaries are half-open ``[start, end)``, so ``train_end`` may equal
        ``validation_start`` (they do not overlap).
        """
        intervals = [
            ("train_start", "train_end", self.train_start, self.train_end),
            ("validation_start", "validation_end", self.validation_start, self.validation_end),
            ("test_start", "test_end", self.test_start, self.test_end),
        ]
        for name_a, name_b, a, b in intervals:
            if not (a < b):
                raise SplitError(f"split interval {name_a}={a} not < {name_b}={b}")
        if self.train_end > self.validation_start:
            raise SplitError("train overlaps validation")
        if self.validation_end > self.test_start:
            raise SplitError("validation overlaps test")
        if self.validation_start != self.train_end:
            # Allowed (a gap between splits is fine), just document intent.
            pass

    def to_dict(self) -> dict[str, str]:
        return {
            k: v.isoformat()
            for k, v in {
                "train_start": self.train_start,
                "train_end": self.train_end,
                "validation_start": self.validation_start,
                "validation_end": self.validation_end,
                "test_start": self.test_start,
                "test_end": self.test_end,
            }.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> SplitConfig:
        return cls(**{k: pd.Timestamp(v) for k, v in data.items()})


@dataclass(frozen=True)
class InstrumentSplit:
    """The position of one split within one instrument's M5 timeline.

    ``first_index``/``last_index`` are absolute indices into the instrument's
    full M5 frame (the frame the environment runs on).  This keeps Phase 1's
    environment unchanged: the sampler guarantees episodes never leave the
    split, and the context-window builder enforces ``strict_split``.
    """

    instrument: str
    split: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    first_index: int
    last_index: int
    n_bars: int

    def __post_init__(self) -> None:
        if self.first_index > self.last_index:
            raise SplitError(
                f"split {self.split!r} of {self.instrument} has no rows "
                f"(first={self.first_index} > last={self.last_index})"
            )


class SplitDataset:
    """Instrument-aware, lazily-loaded dataset with formal temporal splits.

    ``loader`` returns the full :class:`InstrumentData` for an instrument
    (typically from processed parquet); it is cached.  All split arithmetic is
    done on the *full* M5 timeline using absolute indices, so the Phase 1
    environment and its execution semantics are untouched.
    """

    def __init__(
        self,
        split_config: SplitConfig,
        loader: Callable[[str], InstrumentData],
        instruments: tuple[str, ...] = DEFAULT_INSTRUMENT_ORDER,
    ) -> None:
        split_config.validate()
        self.split_config = split_config
        self._loader = loader
        self.instruments = tuple(instruments)
        self._cache: dict[str, InstrumentData] = {}
        self._m5_ts: dict[str, np.ndarray] = {}

    # -- loading --------------------------------------------------------------

    def load(self, instrument: str) -> InstrumentData:
        key = instrument.upper()
        if key not in self._cache:
            self._cache[key] = self._loader(key)
        return self._cache[key]

    def m1(self, instrument: str) -> pd.DataFrame:
        return self.load(instrument).m1

    def m5(self, instrument: str) -> pd.DataFrame:
        return self.load(instrument).m5

    def m5_timestamps(self, instrument: str) -> np.ndarray:
        """Sorted M5 timestamps as ``datetime64[ns]`` (cached)."""
        key = instrument.upper()
        if key not in self._m5_ts:
            self._m5_ts[key] = self.m5(instrument)[TIMESTAMP].to_numpy(dtype="datetime64[ns]")
        return self._m5_ts[key]

    # -- split arithmetic -----------------------------------------------------

    def _bounds(self, instrument: str, split: str) -> tuple[pd.Timestamp, pd.Timestamp, int, int]:
        start_ts, end_ts = self.split_config.range(split)
        ts = self.m5_timestamps(instrument)
        first = int(np.searchsorted(ts, np.datetime64(start_ts), side="left"))
        last = int(np.searchsorted(ts, np.datetime64(end_ts), side="left")) - 1
        return start_ts, end_ts, first, last

    def split(self, instrument: str, split: str) -> InstrumentSplit:
        start_ts, end_ts, first, last = self._bounds(instrument, split)
        return InstrumentSplit(
            instrument=instrument.upper(),
            split=split,
            start_ts=start_ts,
            end_ts=end_ts,
            first_index=first,
            last_index=last,
            n_bars=max(0, last - first + 1),
        )

    def train(self, instrument: str) -> InstrumentSplit:
        return self.split(instrument, "train")

    def validation(self, instrument: str) -> InstrumentSplit:
        return self.split(instrument, "validation")

    def test(self, instrument: str) -> InstrumentSplit:
        return self.split(instrument, "test")

    # -- integrity ------------------------------------------------------------

    def verify_integrity(self, *, strict: bool = True) -> dict[str, Any]:
        """Validate chronology, non-overlap, and per-instrument row presence.

        Returns a machine-readable report; raises :class:`SplitError` when
        ``strict`` and a check fails.
        """
        issues: list[str] = []
        per_instrument: dict[str, dict[str, object]] = {}
        for instr in self.instruments:
            splits: dict[str, object] = {}
            bounds: dict[str, tuple[int, int]] = {}
            for name in SPLIT_NAMES:
                try:
                    isp = self.split(instr, name)
                    splits[name] = {
                        "first_index": isp.first_index,
                        "last_index": isp.last_index,
                        "n_bars": isp.n_bars,
                        "start": str(isp.start_ts),
                        "end": str(isp.end_ts),
                        "first_ts": str(self.m5_timestamps(instr)[isp.first_index]),
                        "last_ts": str(self.m5_timestamps(instr)[isp.last_index]),
                    }
                    bounds[name] = (isp.first_index, isp.last_index)
                except SplitError as exc:
                    issues.append(f"{instr} {name}: {exc}")
                    splits[name] = {"error": str(exc)}
            if (
                "train" in bounds
                and "validation" in bounds
                and (bounds["train"][1] >= bounds["validation"][0])
            ):
                issues.append(f"{instr}: train overlaps validation")
            if (
                "validation" in bounds
                and "test" in bounds
                and bounds["validation"][1] >= bounds["test"][0]
            ):
                issues.append(f"{instr}: validation overlaps test")
            # Verify actual timestamps: max(train) < min(validation) < min(test).
            ts = self.m5_timestamps(instr)
            if all(bounds.get(n) is not None for n in SPLIT_NAMES):
                max_tr = ts[bounds["train"][1]]
                min_val = ts[bounds["validation"][0]]
                min_test = ts[bounds["test"][0]]
                if not (max_tr < min_val):
                    issues.append(f"{instr}: max(train_ts) >= min(validation_ts)")
                if not (min_val < min_test):
                    issues.append(f"{instr}: min(validation_ts) >= min(test_ts)")
            per_instrument[instr] = splits

        report: dict[str, Any] = {
            "split_config": self.split_config.to_dict(),
            "instruments": per_instrument,
            "issues": issues,
            "ok": len(issues) == 0,
        }
        if strict and issues:
            raise SplitError(
                "; ".join(issues[:10])
                + (f" (+{len(issues) - 10} more)" if len(issues) > 10 else "")
            )
        return report

    def manifest(self) -> dict[str, Any]:
        """Machine-readable split manifest (row counts + boundaries)."""
        return self.verify_integrity(strict=False)

    def periods_per_year(self, split: str) -> float:
        """Estimate annualization periods (valid trading observations / year).

        Uses the actual number of M5 observations inside ``split`` across all
        instruments divided by the split's span in years.  Falls back to a
        documented constant when the span is too small to estimate.
        """
        start_ts, end_ts = self.split_config.range(split)
        span_years = max((end_ts - start_ts).total_seconds() / (365.25 * 24 * 3600), 1e-9)
        total = 0
        for instr in self.instruments:
            total += self.split(instr, split).n_bars
        periods = total / max(len(self.instruments), 1)
        if periods <= 0:
            return 50000.0
        return periods / span_years
