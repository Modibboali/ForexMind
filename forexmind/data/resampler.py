"""Deterministic M1 -> M5 resampling.

Aggregation rule for a 5-minute bucket::

    open  = first M1 open
    high  = max  M1 high
    low   = min  M1 low
    close = last M1 close

The M5 bar is labelled with the bucket start timestamp (e.g. ``10:00`` covers
``10:00..10:04``).  Completeness is policy-driven:

* ``STRICT``: an M5 bar is emitted only when the bucket holds exactly the
  expected number of contiguous 1-minute observations.  Gaps inside a bucket
  suppress the bar; large temporal gaps are never bridged into a fake candle.
* ``PARTIAL``: any non-empty bucket is emitted, but ``n_observations`` and
  ``is_complete`` record exactly what was present.

Instrument identity is preserved by the caller (:class:`MarketDataset`).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import pandas as pd

from forexmind.data.schema import CLOSE, HIGH, LOW, OPEN, TIMESTAMP

N_OBSERVATIONS = "n_observations"
IS_COMPLETE = "is_complete"


class CompletenessPolicy(enum.Enum):
    STRICT = "strict"
    PARTIAL = "partial"


@dataclass(frozen=True)
class ResampleConfig:
    """Configuration for :func:`resample_m1_to_m5`."""

    bucket_minutes: int = 5
    expected_bars: int = 5
    completeness: CompletenessPolicy = CompletenessPolicy.STRICT

    def __post_init__(self) -> None:
        if self.bucket_minutes <= 0:
            raise ValueError("bucket_minutes must be > 0")
        if self.expected_bars <= 0:
            raise ValueError("expected_bars must be > 0")
        if self.expected_bars != self.bucket_minutes:
            # For M1 source data the expected number of observations equals the
            # bucket size in minutes; other source intervals are not supported
            # yet (they would need their own expected-bars value).
            pass


def _bucket_keys(timestamps: pd.Series, bucket_minutes: int) -> pd.Series:
    """Floor timestamps to the bucket start (naive datetime64[ns]).

    pandas >= 2.2 may hold non-nanosecond resolutions, so normalise to
    ``datetime64[ns]`` explicitly before integer arithmetic.
    """
    ns = timestamps.astype("datetime64[ns]").astype("int64")
    bucket_ns = bucket_minutes * 60 * 1_000_000_000
    floored = (ns // bucket_ns) * bucket_ns
    return pd.Series(pd.to_datetime(floored, unit="ns", utc=False), index=timestamps.index)


def resample_m1_to_m5(
    frame: pd.DataFrame,
    config: ResampleConfig | None = None,
) -> pd.DataFrame:
    """Resample a canonical M1 frame into M5 bars.

    Returns a DataFrame with canonical OHLC columns plus ``n_observations``
    and ``is_complete`` metadata columns.  The frame must already be
    time-sorted (the loader guarantees this).
    """
    cfg = config or ResampleConfig()
    if not len(frame):
        return pd.DataFrame(
            columns=[TIMESTAMP, OPEN, HIGH, LOW, CLOSE, N_OBSERVATIONS, IS_COMPLETE]
        )

    keys = _bucket_keys(frame[TIMESTAMP], cfg.bucket_minutes)
    work = frame.copy()
    work["_bucket"] = keys

    agg = (
        work.groupby("_bucket", sort=True)
        .agg(
            open=(OPEN, "first"),
            high=(HIGH, "max"),
            low=(LOW, "min"),
            close=(CLOSE, "last"),
            n_observations=(OPEN, "count"),
        )
        .reset_index()
    )
    agg.rename(columns={"_bucket": TIMESTAMP}, inplace=True)

    n_obs = agg[N_OBSERVATIONS].to_numpy(dtype="int64")
    is_complete = n_obs == cfg.expected_bars

    if cfg.completeness is CompletenessPolicy.STRICT:
        # Additionally require contiguity: the M1 timestamps inside each
        # accepted bucket must be exactly consecutive minutes.  A bucket with
        # k rows is contiguous iff all k-1 internal 1-minute gaps are exactly
        # 60 seconds.  Vectorised: count "is_contig" flags per bucket,
        # excluding each bucket's first row (whose diff is against the
        # *previous* bucket and therefore not an internal gap).
        contig = work[TIMESTAMP].diff().dt.total_seconds().eq(60.0)
        first_in_bucket = ~work["_bucket"].duplicated(keep="first")
        contig = contig & ~first_in_bucket
        # Sum per bucket (aligned with ``agg`` which is also sorted by bucket).
        contig_sum = contig.groupby(work["_bucket"]).sum().to_numpy(dtype="int64")
        contiguous = contig_sum == (n_obs - 1)
        keep = is_complete & contiguous
        out = agg.loc[keep].copy()
        out[IS_COMPLETE] = True
    else:
        out = agg.copy()
        out[IS_COMPLETE] = is_complete

    out = out.reset_index(drop=True)
    return out[[TIMESTAMP, OPEN, HIGH, LOW, CLOSE, N_OBSERVATIONS, IS_COMPLETE]]
