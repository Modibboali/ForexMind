"""Market context window (Phase 2).

The window builder returns exactly ``[current_index - context_length + 1,
current_index]`` bars with **no future rows**.  It also provides
``prior_close`` (the close immediately before the window) so the first bar's
return features can be computed, and per-bar gap metadata so temporal gaps are
never silently treated as normal M5 progression.

Context policy:

* ``strict_split`` (default): the entire window AND ``prior_close`` must come
  from the same split.  This prevents cross-split contamination.
* ``historical_warmup``: the window may use bars before the split start (e.g.
  to give a model historical context at a split boundary).  The episode itself
  still stays inside the split.

An invalid context raises :class:`WindowError` instead of silently padding
with future data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forexmind.data.schema import CLOSE, HIGH, LOW, OPEN, TIMESTAMP

MINUTES_PER_DAY = 1440.0


class WindowError(ValueError):
    """Raised when a context window cannot be constructed validly."""


@dataclass(frozen=True)
class WindowConfig:
    context_length: int = 64
    context_policy: str = "strict_split"  # "strict_split" | "historical_warmup"

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be > 0")
        if self.context_policy not in ("strict_split", "historical_warmup"):
            raise ValueError(
                f"unsupported context_policy {self.context_policy!r}; "
                "use 'strict_split' or 'historical_warmup'"
            )


@dataclass(frozen=True, slots=True)
class MarketWindow:
    """A causal window of ``context_length`` M5 bars ending at ``current_index``."""

    instrument: str
    current_index: int
    timestamps: np.ndarray  # (context_length,) datetime64[ns]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    prior_close: float  # close of the bar immediately before the window
    minutes_since_previous: np.ndarray  # (context_length,) incl. prior bar
    is_weekend_gap: np.ndarray  # (context_length,) bool

    @property
    def closes(self) -> np.ndarray:
        return self.close

    @property
    def ohlc(self) -> np.ndarray:
        return np.stack([self.open, self.high, self.low, self.close], axis=1)


class MarketWindowBuilder:
    """Builds causal market windows for one instrument within one split."""

    def __init__(
        self,
        instrument: str,
        m5: pd.DataFrame,
        split_start: pd.Timestamp,
        split_end: pd.Timestamp,
        config: WindowConfig | None = None,
    ) -> None:
        self.instrument = instrument.upper()
        self._cfg = config or WindowConfig()
        self._ts = m5[TIMESTAMP].to_numpy(dtype="datetime64[ns]")
        self._open = m5[OPEN].to_numpy(dtype="float64")
        self._high = m5[HIGH].to_numpy(dtype="float64")
        self._low = m5[LOW].to_numpy(dtype="float64")
        self._close = m5[CLOSE].to_numpy(dtype="float64")
        self._split_start = np.datetime64(split_start)
        self._split_end = np.datetime64(split_end)
        self._n = len(self._ts)

        if self._cfg.context_policy == "strict_split":
            # Window start AND prior_close must be >= split start.
            first_idx = int(np.searchsorted(self._ts, self._split_start, side="left"))
            self._min_valid = first_idx + self._cfg.context_length
        else:
            # Historical warmup: just need enough real bars before the index.
            self._min_valid = self._cfg.context_length
        if self._min_valid > max(self._n - 1, 0):
            raise WindowError(
                f"{self.instrument}: cannot fit {self._cfg.context_length}-bar context "
                f"within split starting at {split_start}"
            )

    def min_valid_index(self) -> int:
        """Smallest observation index that can build a valid window."""
        return self._min_valid

    def max_valid_index(self) -> int:
        """Largest observation index that can build a valid window."""
        return self._n - 1

    def is_eligible(self, current_index: int) -> bool:
        return self._min_valid <= current_index < self._n

    def build(self, current_index: int) -> MarketWindow:
        if not (self._min_valid <= current_index < self._n):
            raise WindowError(
                f"{self.instrument}: index {current_index} cannot build a valid "
                f"{self._cfg.context_length}-bar window "
                f"(eligible range [{self._min_valid}, {self._n - 1}])"
            )
        start = current_index - self._cfg.context_length + 1
        prior = start - 1
        idx = np.arange(start, current_index + 1)

        # Per-bar gap metadata (minutes since previous bar, weekend flag).
        all_idx = np.concatenate(([prior], idx))
        all_ts = self._ts[all_idx]
        minutes = np.diff(all_ts) / np.timedelta64(1, "m")
        days = all_ts.astype("datetime64[D]").astype("int64")
        dow = (days + 3) % 7  # 1970-01-01 was Thursday (dayofweek 3)
        prev_dow, cur_dow = dow[:-1], dow[1:]
        weekend = ((prev_dow >= 4) & (prev_dow <= 5)) & ((cur_dow == 6) | (cur_dow == 0))

        return MarketWindow(
            instrument=self.instrument,
            current_index=current_index,
            timestamps=self._ts[idx].copy(),
            open=self._open[idx].copy(),
            high=self._high[idx].copy(),
            low=self._low[idx].copy(),
            close=self._close[idx].copy(),
            prior_close=float(self._close[prior]),
            minutes_since_previous=minutes,
            is_weekend_gap=weekend,
        )
