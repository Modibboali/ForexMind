"""Tests for the deterministic episode sampler (Phase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from forexmind.episodes.config import EpisodeConfig, GapPolicy
from forexmind.episodes.sampler import EpisodeSampler

from tests.synthetic import (
    make_instrument,
    make_split_dataset,
    timeline_m5,
)

DATES = ["2020-01-06", "2020-06-01", "2021-03-01", "2021-09-01", "2022-03-01", "2022-09-01"]


def _dataset(instruments=("EURUSD", "GBPUSD")) -> object:
    return make_split_dataset(
        {i: make_instrument(i, timeline_m5(DATES, per_day=20)) for i in instruments}
    )


def _sampler(ds, split="train", horizon=4, context=3, seed=42, gap=None) -> EpisodeSampler:
    return EpisodeSampler(
        ds,
        EpisodeConfig(
            split=split,
            horizon=horizon,
            context_length=context,
            seed=seed,
            gap_policy=gap or GapPolicy(),
        ),
    )


def test_valid_start_bounds() -> None:
    ds = _dataset()
    s = _sampler(ds)
    starts = s.valid_starts("EURUSD", "train")
    isp = ds.train("EURUSD")
    min_start = isp.first_index + 3
    max_start = isp.last_index - 4
    assert starts.min() == min_start
    assert starts.max() == max_start
    assert len(starts) == max_start - min_start + 1


def test_reproducibility_same_seed() -> None:
    ds = _dataset()
    a = _sampler(ds, seed=42).sample(20, seed=42)
    b = _sampler(ds, seed=42).sample(20, seed=42)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]


def test_horizon_correctness() -> None:
    ds = _dataset()
    specs = _sampler(ds, horizon=6, context=3).sample(20, seed=1)
    for spec in specs:
        assert spec.end_index == spec.start_index + spec.horizon
        assert spec.horizon == 6
        assert spec.context_length == 3
        # The episode stays inside the split.
        isp = ds.split(spec.instrument, spec.split)
        assert spec.start_index >= isp.first_index
        assert spec.end_index <= isp.last_index


def test_multi_instrument_uniform_sampling() -> None:
    ds = _dataset()
    specs = _sampler(ds, split="test").sample(400, seed=7)
    instruments = [s.instrument for s in specs]
    assert set(instruments) == {"EURUSD", "GBPUSD"}
    # Uniform instrument sampling: neither dominates (equal probability).
    assert 0.3 < instruments.count("EURUSD") / len(instruments) < 0.7


def test_explicit_spec() -> None:
    ds = _dataset()
    s = _sampler(ds, split="train")
    isp = ds.train("EURUSD")
    spec = s.explicit("EURUSD", isp.first_index + 3)
    assert spec.start_index == isp.first_index + 3
    with pytest.raises(ValueError):
        s.explicit("EURUSD", isp.first_index)  # not enough context


def test_gap_policy_blocks_crossing_starts() -> None:
    # m5: 20 bars in Jan 2020, then a large 5-hour gap, then 20 bars in June 2020.
    m5 = timeline_m5(["2020-01-06", "2020-06-01"], per_day=20)
    ts = m5["timestamp"].to_numpy(dtype="datetime64[ns]")
    ts[20:] = ts[20:] + np.timedelta64(300, "m")
    m5["timestamp"] = pd.to_datetime(ts)
    ds = make_split_dataset({"EURUSD": make_instrument("EURUSD", m5)})

    permissive = _sampler(ds, gap=GapPolicy(allow_cross_weekend=True, max_bar_gap_minutes=None))
    restrictive = _sampler(ds, gap=GapPolicy(allow_cross_weekend=True, max_bar_gap_minutes=60))
    all_starts = permissive.valid_starts("EURUSD", "train")
    valid_starts = restrictive.valid_starts("EURUSD", "train")
    assert len(valid_starts) < len(all_starts)
    # No valid start's market sequence [s-context+1, s+horizon] crosses the gap.
    ts_np = m5["timestamp"].to_numpy(dtype="datetime64[ns]")
    for s in valid_starts:
        a, b = s - 3 + 1, s + 4
        span = ts_np[a : b + 1]
        assert np.all(np.diff(span) <= np.timedelta64(60, "m"))


def test_allow_cross_weekend_policy() -> None:
    # Friday bars then Sunday bars -> a weekend gap in between.
    fri = timeline_m5(["2020-01-03"], per_day=20)  # Friday
    sun = timeline_m5(["2020-01-05"], per_day=20)  # Sunday
    m5 = pd.concat([fri, sun], ignore_index=True)
    ds = make_split_dataset({"EURUSD": make_instrument("EURUSD", m5)})

    with_weekend = _sampler(ds, gap=GapPolicy(allow_cross_weekend=True, max_bar_gap_minutes=None))
    no_weekend = _sampler(ds, gap=GapPolicy(allow_cross_weekend=False, max_bar_gap_minutes=None))
    n_with = len(with_weekend.valid_starts("EURUSD", "train"))
    n_without = len(no_weekend.valid_starts("EURUSD", "train"))
    assert n_with > n_without  # weekend-crossing starts are blocked when disallowed
