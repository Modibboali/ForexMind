"""Episode configuration (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forexmind.data.splits import SPLIT_NAMES


@dataclass(frozen=True)
class GapPolicy:
    """Controls how the episode sampler treats temporal data gaps.

    * ``allow_cross_weekend`` (default True): episodes may span the weekend
      closure; the gap remains explicit in the observation time features.
    * ``max_bar_gap_minutes`` (default None = unlimited): if set, any
      non-weekend gap between consecutive M5 bars larger than this invalidates
      an episode start (the temporal relationship would be ambiguous).

    Gaps are never fabricated or silently discarded.
    """

    allow_cross_weekend: bool = True
    max_bar_gap_minutes: int | None = None

    def __post_init__(self) -> None:
        if self.max_bar_gap_minutes is not None and self.max_bar_gap_minutes <= 0:
            raise ValueError("max_bar_gap_minutes must be > 0 or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_cross_weekend": self.allow_cross_weekend,
            "max_bar_gap_minutes": self.max_bar_gap_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GapPolicy:
        return cls(
            allow_cross_weekend=bool(data.get("allow_cross_weekend", True)),
            max_bar_gap_minutes=data.get("max_bar_gap_minutes"),
        )


@dataclass(frozen=True)
class EpisodeConfig:
    """Episode-level configuration (serializable)."""

    split: str = "train"
    horizon: int = 512  # number of decision steps per episode
    context_length: int = 64
    seed: int = 42
    instrument_sampling: str = "uniform"  # Phase 2: "uniform" only
    gap_policy: GapPolicy = field(default_factory=GapPolicy)

    def __post_init__(self) -> None:
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {self.split!r}; expected {list(SPLIT_NAMES)}")
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if self.context_length <= 0:
            raise ValueError("context_length must be > 0")
        if self.instrument_sampling != "uniform":
            raise ValueError(
                f"unsupported instrument_sampling {self.instrument_sampling!r}; "
                "Phase 2 supports 'uniform' only (stratification is a future "
                "extension point)"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "horizon": self.horizon,
            "context_length": self.context_length,
            "seed": self.seed,
            "instrument_sampling": self.instrument_sampling,
            "gap_policy": self.gap_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeConfig:
        gap = GapPolicy.from_dict(dict(data.get("gap_policy", {})))
        return cls(
            split=str(data["split"]),
            horizon=int(data["horizon"]),
            context_length=int(data["context_length"]),
            seed=int(data["seed"]),
            instrument_sampling=str(data.get("instrument_sampling", "uniform")),
            gap_policy=gap,
        )
