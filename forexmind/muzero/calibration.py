"""Target-scale calibration for the MuZero support heads (Stage 4.4 §10).

Forex log-equity rewards are tiny (``~1e-6`` to ``1e-3``), so a board-game-scale
support range would waste every bin.  ``scale`` controls how much of the
``[-3*scale, +3*scale]`` representable window (with ``epsilon = 0``) the targets
actually occupy.

The rule used here is explicit and inspectable::

    scale = p(|target|) / h^-1(target_u)

so that the chosen percentile of ``|target|`` lands at ``u = target_u`` in the
``[-1, 1]`` support window, leaving ``1 - target_u`` of headroom for rare moves.
With the default ``target_u = 0.6`` that is ``z/scale = 1.56`` at the percentile
and roughly a 2x margin before saturation.

Nothing is clipped silently: :func:`saturation_fraction` reports how much of the
observed data would land on the edge bins, and the report includes it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from forexmind.muzero.support import (
    inverse_transform_to_scalar,
    saturation_fraction,
)

__all__ = [
    "TargetStatistics",
    "calibration_report",
    "describe_targets",
    "headroom_factor",
    "propose_scale",
    "reward_targets",
    "value_targets",
]

_PERCENTILES = (1.0, 5.0, 50.0, 95.0, 99.0)


@dataclass(frozen=True, slots=True)
class TargetStatistics:
    """Distribution of a scalar target under its validity mask."""

    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    p01: float
    p05: float
    p50: float
    p95: float
    p99: float
    abs_p50: float
    abs_p95: float
    abs_p99: float
    abs_max: float

    @property
    def absolute_scale(self) -> float:
        """A robust characteristic magnitude: ``abs_p99`` with fallbacks."""
        for candidate in (self.abs_p99, self.abs_p95, self.abs_p50, abs(self.mean)):
            if candidate > 0.0:
                return float(candidate)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reward_targets(batch: Any) -> np.ndarray:
    """Valid real environment reward targets from a ``MuZeroBatch``."""
    return _masked_values(batch.target_rewards, batch.reward_masks)


def value_targets(batch: Any) -> np.ndarray:
    """Valid n-step value targets from a ``MuZeroBatch``."""
    return _masked_values(batch.target_values, batch.value_masks)


def _masked_values(values: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
    values_np = values.detach().cpu().numpy().astype(np.float64).reshape(-1)
    masks_np = masks.detach().cpu().numpy().reshape(-1) > 0.5
    if values_np.shape != masks_np.shape:
        raise ValueError(f"target/mask shape mismatch: {values_np.shape} vs {masks_np.shape}")
    return values_np[masks_np]


def describe_targets(values: np.ndarray | list[float]) -> TargetStatistics:
    """Summarize a target array (empty input yields an all-zero record)."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return TargetStatistics(
            count=0,
            mean=0.0,
            std=0.0,
            minimum=0.0,
            maximum=0.0,
            p01=0.0,
            p05=0.0,
            p50=0.0,
            p95=0.0,
            p99=0.0,
            abs_p50=0.0,
            abs_p95=0.0,
            abs_p99=0.0,
            abs_max=0.0,
        )
    absolute = np.abs(finite)
    percentiles = np.percentile(finite, _PERCENTILES)
    abs_percentiles = np.percentile(absolute, _PERCENTILES)
    return TargetStatistics(
        count=int(finite.size),
        mean=float(finite.mean()),
        std=float(finite.std()),
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        p01=float(percentiles[0]),
        p05=float(percentiles[1]),
        p50=float(percentiles[2]),
        p95=float(percentiles[3]),
        p99=float(percentiles[4]),
        abs_p50=float(abs_percentiles[2]),
        abs_p95=float(abs_percentiles[3]),
        abs_p99=float(abs_percentiles[4]),
        abs_max=float(absolute.max()),
    )


def headroom_factor(target_u: float = 0.6) -> float:
    """``h^-1(target_u)``: the ``|z / scale|`` that lands at ``u = target_u``."""
    if not 0.0 < target_u < 1.0:
        raise ValueError(f"target_u must be in (0, 1), got {target_u}")
    return float(inverse_transform_to_scalar(torch.tensor([float(target_u)]))[0].item())


def propose_scale(
    stats: TargetStatistics,
    *,
    target_u: float = 0.6,
    floor: float = 1e-9,
) -> float:
    """Scale placing ``abs_p99`` at ``u = target_u`` in the support window."""
    factor = headroom_factor(target_u)
    scale = stats.absolute_scale / factor
    if not np.isfinite(scale) or scale <= 0.0:
        return float(floor)
    return float(max(scale, floor))


def calibration_report(batch: Any, *, target_u: float = 0.6) -> dict[str, Any]:
    """Reward/value statistics, proposed scales, and resulting saturation."""
    reward_stats = describe_targets(reward_targets(batch))
    value_stats = describe_targets(value_targets(batch))
    reward_scale = propose_scale(reward_stats, target_u=target_u)
    value_scale = propose_scale(value_stats, target_u=target_u)
    return {
        "target_u": float(target_u),
        "headroom_factor": headroom_factor(target_u),
        "reward": {
            "statistics": reward_stats.to_dict(),
            "proposed_scale": reward_scale,
            "saturation_fraction": saturation_fraction(
                batch.target_rewards.detach().reshape(-1).float(), scale=reward_scale
            ),
        },
        "value": {
            "statistics": value_stats.to_dict(),
            "proposed_scale": value_scale,
            "saturation_fraction": saturation_fraction(
                batch.target_values.detach().reshape(-1).float(), scale=value_scale
            ),
        },
    }
