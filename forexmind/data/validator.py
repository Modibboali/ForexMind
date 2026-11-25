"""Dataset validation and temporal gap classification.

The validator checks structural, OHLC-consistency and temporal problems and
*classifies* gaps rather than treating every missing bar as an error.  Forex
is ~24/5 with different activity regimes, so the temporal structure must be
preserved: a Friday->Sunday break is a ``WEEKEND_GAP``, a missing minute is a
``SHORT_GAP``, and so on.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forexmind.data.schema import CLOSE, HIGH, LOW, OPEN, TIMESTAMP


class GapType(enum.Enum):
    """Classification of the gap between two consecutive bars."""

    NORMAL = "normal"  # exactly the expected bar interval
    SHORT_GAP = "short_gap"  # a few missing bars (thin period / dropout)
    WEEKEND_GAP = "weekend_gap"  # break across Saturday/Sunday
    LARGE_GAP = "large_gap"  # a long unexplained gap (e.g. holiday)
    UNKNOWN = "unknown"  # first bar, or interval not otherwise classified


@dataclass(frozen=True)
class GapConfig:
    """Thresholds controlling gap classification.

    ``bar_interval`` is the expected interval in minutes (1 for M1 data).
    Gaps strictly larger than ``bar_interval`` but smaller than
    ``large_gap_minutes`` are ``SHORT_GAP``.  Weekend gaps are detected
    from the calendar (Friday->Sunday / Friday->Monday etc.) regardless of
    duration.  Remaining gaps >= ``large_gap_minutes`` are ``LARGE_GAP``.
    """

    bar_interval: int = 1
    large_gap_minutes: int = 60


@dataclass(frozen=True)
class GapRecord:
    """One classified gap between ``prev`` and ``next`` bars."""

    gap_type: GapType
    prev_timestamp: pd.Timestamp
    next_timestamp: pd.Timestamp
    gap_minutes: float


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation problem."""

    code: str
    message: str
    is_error: bool = True
    index: int | None = None

    def __str__(self) -> str:
        loc = f" @row {self.index}" if self.index is not None else ""
        return f"[{self.code}]{loc}: {self.message}"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one instrument's M1 series."""

    instrument: str
    issues: tuple[ValidationIssue, ...] = ()
    gap_counts: dict[str, int] = field(default_factory=dict)
    gaps: tuple[GapRecord, ...] = ()
    row_count: int = 0

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.is_error)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if not i.is_error)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def classify_gap(prev: pd.Timestamp, next_: pd.Timestamp, cfg: GapConfig) -> GapType:
    """Classify the temporal gap between two consecutive bars."""
    delta = next_ - prev
    gap_minutes = delta.total_seconds() / 60.0

    if gap_minutes <= cfg.bar_interval:
        # Exactly the expected interval (or, permissively, slightly less due to
        # rounding) is normal.
        return GapType.NORMAL

    if _is_weekend_gap(prev, next_):
        return GapType.WEEKEND_GAP

    if gap_minutes >= cfg.large_gap_minutes:
        return GapType.LARGE_GAP

    return GapType.SHORT_GAP


def _is_weekend_gap(prev: pd.Timestamp, next_: pd.Timestamp) -> bool:
    """True when the break is the market closure across Saturday/Sunday.

    The trading week runs Sunday 17:00 -> Friday close, so Sunday->Monday is
    *continuous* trading (never a weekend gap).  A weekend gap is a bar on
    Friday (or Saturday) followed by a bar on Sunday (market reopen) or
    Monday (when no Sunday bars exist).
    """
    prev_dow, next_dow = prev.dayofweek, next_.dayofweek
    return prev_dow in (4, 5) and next_dow in (6, 0)


class MarketDataValidator:
    """Validates a canonical M1 (or any-interval) OHLC frame."""

    def __init__(
        self,
        gap_config: GapConfig | None = None,
        max_gap_records: int = 2000,
    ) -> None:
        self._gap_config = gap_config or GapConfig()
        self._max_gap_records = max(1, max_gap_records)

    def validate(self, frame: pd.DataFrame, instrument: str = "") -> ValidationResult:
        issues: list[ValidationIssue] = []
        row_count = len(frame)

        for col in (TIMESTAMP, OPEN, HIGH, LOW, CLOSE):
            if col not in frame.columns:
                issues.append(ValidationIssue("missing_column", f"missing required column {col!r}"))
        if any(i.code == "missing_column" for i in issues):
            return ValidationResult(instrument, tuple(issues))

        ts = frame[TIMESTAMP]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            issues.append(
                ValidationIssue("bad_timestamp_dtype", "timestamp column is not datetime64")
            )
            return ValidationResult(instrument, tuple(issues))

        if ts.isna().any():
            n = int(ts.isna().sum())
            issues.append(ValidationIssue("null_timestamp", f"{n} null timestamp(s)"))

        # Monotonicity & duplicates.
        diff = ts.diff()
        if (diff < pd.Timedelta(0)).any():
            idx = int(np.flatnonzero((diff < pd.Timedelta(0)).to_numpy())[0])
            issues.append(
                ValidationIssue(
                    "unsorted_timestamps",
                    f"timestamps not monotonically increasing (first inversion at row {idx})",
                    index=idx,
                )
            )
        if ts.duplicated().any():
            n = int(ts.duplicated().sum())
            issues.append(ValidationIssue("duplicate_timestamps", f"{n} duplicate timestamp(s)"))
            dup_mask = ts.duplicated(keep=False).to_numpy()
            sub = frame.loc[dup_mask, [OPEN, HIGH, LOW, CLOSE]]
            if sub.duplicated().any():
                issues.append(
                    ValidationIssue(
                        "duplicate_bars",
                        "duplicate timestamp(s) with identical OHLC",
                    )
                )

        # Numeric & finite checks + OHLC consistency.
        for col in (OPEN, HIGH, LOW, CLOSE):
            s = frame[col]
            if not pd.api.types.is_numeric_dtype(s):
                issues.append(ValidationIssue("non_numeric_ohlc", f"column {col!r} is not numeric"))
                continue
            if s.isna().any():
                n = int(s.isna().sum())
                issues.append(ValidationIssue("nan_value", f"column {col!r} has {n} NaN value(s)"))
            inf_mask = np.isinf(s.to_numpy(dtype="float64"))
            if inf_mask.any():
                idx = int(np.flatnonzero(inf_mask)[0])
                issues.append(
                    ValidationIssue(
                        "infinite_value", f"column {col!r} has infinite value(s)", index=idx
                    )
                )

        # Row-level OHLC consistency, fully vectorised.
        o = frame[OPEN].to_numpy(dtype="float64")
        h = frame[HIGH].to_numpy(dtype="float64")
        lo = frame[LOW].to_numpy(dtype="float64")
        cl = frame[CLOSE].to_numpy(dtype="float64")
        finite = (
            frame[OPEN].notna().to_numpy()
            & frame[HIGH].notna().to_numpy()
            & frame[LOW].notna().to_numpy()
            & frame[CLOSE].notna().to_numpy()
            & np.isfinite(o)
            & np.isfinite(h)
            & np.isfinite(lo)
            & np.isfinite(cl)
        )
        ok = (
            (h >= lo)
            & (h >= o)
            & (h >= cl)
            & (lo <= o)
            & (lo <= cl)
            & (o > 0)
            & (h > 0)
            & (lo > 0)
            & (cl > 0)
        ) | ~finite
        bad_idx = np.flatnonzero(~ok)
        for r_idx in bad_idx[:10]:
            ts_ = frame.iloc[r_idx][TIMESTAMP]
            issues.append(
                ValidationIssue(
                    "ohlc_inconsistent",
                    f"{ts_}: OHLC invariants violated "
                    f"(open={o[r_idx]}, high={h[r_idx]}, low={lo[r_idx]}, close={cl[r_idx]})",
                    index=int(r_idx),
                )
            )
        if len(bad_idx) > 10:
            issues.append(
                ValidationIssue(
                    "ohlc_inconsistent",
                    f"OHLC invariants violated on {len(bad_idx)} row(s); first 10 shown",
                )
            )

        # Gap classification, vectorised.
        ts_np = ts.to_numpy(dtype="datetime64[ns]")
        gap_minutes = np.zeros(row_count, dtype="float64")
        gtype = np.empty(row_count, dtype=object)
        if row_count:
            gtype[0] = GapType.UNKNOWN
        if row_count > 1:
            gap_minutes[1:] = (ts_np[1:] - ts_np[:-1]) / np.timedelta64(1, "m")
            days = ts_np.astype("datetime64[D]").astype("int64")
            dow = (days + 3) % 7  # 1970-01-01 was Thursday (dayofweek 3)
            prev_dow, cur_dow = dow[:-1], dow[1:]
            # Weekend = market closure: Fri/Sat -> Sun/Mon. Sunday->Monday is
            # continuous trading and must NOT be flagged.
            weekend = ((prev_dow >= 4) & (prev_dow <= 5)) & ((cur_dow == 6) | (cur_dow == 0))
            gap_diff = gap_minutes[1:]
            normal = gap_diff <= self._gap_config.bar_interval
            large = gap_diff >= self._gap_config.large_gap_minutes
            # Order matters: NORMAL < LARGE < WEEKEND precedence (matches the
            # scalar classify_gap: weekend beats large, normal beats all).
            gtype[1:] = GapType.SHORT_GAP
            gtype[1:][normal] = GapType.NORMAL
            gtype[1:][large] = GapType.LARGE_GAP
            gtype[1:][weekend] = GapType.WEEKEND_GAP

        gap_counts: dict[str, int] = {}
        for gt in (GapType.NORMAL, GapType.SHORT_GAP, GapType.WEEKEND_GAP, GapType.LARGE_GAP):
            cnt = int(np.count_nonzero(gtype == gt))
            if cnt:
                gap_counts[gt.value] = cnt

        # Keep full detail for small frames; for large frames keep the first bar
        # plus a bounded sample of non-normal gaps to bound memory.
        if row_count <= self._max_gap_records:
            keep = np.arange(row_count)
        else:
            keep = np.flatnonzero(gtype != GapType.NORMAL)[: self._max_gap_records]
        gaps = tuple(
            GapRecord(
                gtype[i],
                pd.Timestamp(ts_np[i - 1]) if i > 0 else pd.Timestamp(ts_np[i]),
                pd.Timestamp(ts_np[i]),
                float(gap_minutes[i]),
            )
            for i in keep
        )

        return ValidationResult(
            instrument=instrument,
            issues=tuple(issues),
            gap_counts=dict(gap_counts),
            gaps=gaps,
            row_count=row_count,
        )


def summarize_gaps(result: ValidationResult, top_n: int = 8) -> str:
    """Render a human-readable gap summary (used by inspection scripts)."""
    lines = [f"gaps (excluding first bar): {result.gap_counts}"]
    interesting = [g for g in result.gaps if g.gap_type in (GapType.WEEKEND_GAP, GapType.LARGE_GAP)]
    for g in interesting[:top_n]:
        lines.append(
            f"  {g.gap_type.value:<12} {g.prev_timestamp} -> {g.next_timestamp} "
            f"({g.gap_minutes:,.0f} min)"
        )
    if len(interesting) > top_n:
        lines.append(f"  ... and {len(interesting) - top_n} more")
    return "\n".join(lines)
