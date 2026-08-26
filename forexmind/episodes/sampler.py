"""Deterministic, gap-aware episode sampler (Phase 2).

The sampler only *chooses* episodes; :class:`ForexEnvironment` executes them.
Sampling is reproducible (same seed -> same specs) and:

* instruments are sampled uniformly (each pair equal probability, so EURUSD
  does not dominate by row count);
* valid starts are chosen uniformly among starts that fit the context window
  and the episode horizon inside the split, and that respect the gap policy;
* an episode never leaves its split and never observes future data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forexmind.data.splits import SplitDataset
from forexmind.episodes.config import EpisodeConfig


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """A reproducible episode specification."""

    instrument: str
    split: str
    start_index: int  # absolute M5 index where the episode begins
    end_index: int  # absolute M5 index of the final observation
    horizon: int  # number of decision steps
    context_length: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "instrument": self.instrument,
            "split": self.split,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "horizon": self.horizon,
            "context_length": self.context_length,
            "seed": self.seed,
        }


class EpisodeSampler:
    def __init__(self, dataset: SplitDataset, config: EpisodeConfig) -> None:
        self.dataset = dataset
        self.config = config
        self._valid_starts: dict[tuple[str, str], np.ndarray] = {}

    # -- gap awareness --------------------------------------------------------

    def _disallowed_pairs(self, instrument: str) -> np.ndarray:
        """Boolean array ``disallowed[j]`` for the gap between M5 bars j-1 and j."""
        ts = self.dataset.m5_timestamps(instrument)
        n = len(ts)
        disallowed = np.zeros(n, dtype=bool)
        if n <= 1:
            return disallowed
        minutes = (ts[1:] - ts[:-1]) / np.timedelta64(1, "m")
        days = ts.astype("datetime64[D]").astype("int64")
        dow = (days + 3) % 7  # 1970-01-01 was Thursday (dayofweek 3)
        prev_dow, cur_dow = dow[:-1], dow[1:]
        weekend = ((prev_dow >= 4) & (prev_dow <= 5)) & ((cur_dow == 6) | (cur_dow == 0))
        bad = np.zeros(n - 1, dtype=bool)
        policy = self.config.gap_policy
        if not policy.allow_cross_weekend:
            bad |= weekend
        if policy.max_bar_gap_minutes is not None:
            too_long = minutes > policy.max_bar_gap_minutes
            bad |= too_long & ~weekend
        disallowed[1:] = bad
        return disallowed

    # -- valid starts ---------------------------------------------------------

    def valid_starts(self, instrument: str, split: str | None = None) -> np.ndarray:
        """Absolute M5 start indices that are valid for episodes.

        A start is valid when the context window (including ``prior_close``)
        and the full episode horizon fit inside the split, and no disallowed
        gap lies within the episode's market sequence.
        """
        split = split or self.config.split
        key = (instrument.upper(), split)
        if key in self._valid_starts:
            return self._valid_starts[key]

        isp = self.dataset.split(key[0], split)
        first, last = isp.first_index, isp.last_index
        horizon = self.config.horizon
        context = self.config.context_length

        min_start = first + context  # prior_close and window inside the split
        max_start = last - horizon
        if min_start > max_start:
            self._valid_starts[key] = np.empty(0, dtype="int64")
            return self._valid_starts[key]

        starts = np.arange(min_start, max_start + 1, dtype="int64")
        pref = np.concatenate([[0], np.cumsum(self._disallowed_pairs(key[0]))])
        a = starts - context + 1
        b = starts + horizon
        gap_count = pref[b + 1] - pref[a]
        valid = starts[gap_count == 0]
        self._valid_starts[key] = valid
        return valid

    def n_valid_starts(self, instrument: str, split: str | None = None) -> int:
        return len(self.valid_starts(instrument, split))

    # -- sampling -------------------------------------------------------------

    def sample(
        self,
        n: int,
        *,
        seed: int | None = None,
        instruments: list[str] | tuple[str, ...] | None = None,
        split: str | None = None,
    ) -> list[EpisodeSpec]:
        """Sample ``n`` reproducible episode specs.

        Instruments are sampled uniformly; within an instrument, valid starts
        are sampled uniformly.  Each spec carries a unique derived seed.
        """
        split = split or self.config.split
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)
        instrs = list(instruments) if instruments is not None else list(self.dataset.instruments)
        if not instrs:
            raise ValueError("no instruments available for sampling")
        chosen = [str(rng.choice(instrs)) for _ in range(n)]
        specs: list[EpisodeSpec] = []
        for k, instr in enumerate(chosen):
            starts = self.valid_starts(instr, split)
            if len(starts) == 0:
                raise RuntimeError(f"no valid episode starts for {instr} in split {split!r}")
            idx = int(rng.integers(0, len(starts)))
            s = int(starts[idx])
            specs.append(
                EpisodeSpec(
                    instrument=instr,
                    split=split,
                    start_index=s,
                    end_index=s + self.config.horizon,
                    horizon=self.config.horizon,
                    context_length=self.config.context_length,
                    seed=(seed if seed is not None else self.config.seed) + k,
                )
            )
        return specs

    def explicit(
        self,
        instrument: str,
        start_index: int,
        *,
        horizon: int | None = None,
        split: str | None = None,
    ) -> EpisodeSpec:
        """Build a spec for an explicit (instrument, start) pair.

        Raises when the start cannot fit inside the split (context + horizon).
        """
        split = split or self.config.split
        horizon = horizon or self.config.horizon
        instr = instrument.upper()
        isp = self.dataset.split(instr, split)
        min_start = isp.first_index + self.config.context_length
        max_start = isp.last_index - horizon
        if not (min_start <= start_index <= max_start):
            raise ValueError(
                f"start {start_index} not valid for {instr}/{split}: "
                f"eligible range [{min_start}, {max_start}]"
            )
        return EpisodeSpec(
            instrument=instr,
            split=split,
            start_index=start_index,
            end_index=start_index + horizon,
            horizon=horizon,
            context_length=self.config.context_length,
            seed=self.config.seed,
        )
