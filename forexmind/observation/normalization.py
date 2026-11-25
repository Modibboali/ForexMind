"""Leakage-free normalization (Phase 2).

The Phase-2 encoder is deliberately built from *local, scale-independent*
transformations (returns, relative prices, ratios) so that no fitted
statistics are required.  For any future feature that needs fitted statistics,
the :class:`Normalizer` abstraction enforces::

    fit(train)   -> transform(train) -> transform(validation) -> transform(test)

Never fit on validation or test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Normalizer(ABC):
    """A normalizer that must be fit on training data only."""

    @abstractmethod
    def fit(self, values: np.ndarray) -> Normalizer:
        """Compute statistics from ``values`` (train split only)."""

    @abstractmethod
    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the fitted transform (idempotent-safe on new data)."""

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)


class IdentityNormalizer(Normalizer):
    """No-op normalizer (default for Phase 2 analytical features)."""

    def fit(self, values: np.ndarray) -> IdentityNormalizer:
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)


class StandardNormalizer(Normalizer):
    """Standard score ``(x - mean) / std`` fit on training data only."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self._mean: float | None = None
        self._std: float | None = None

    def fit(self, values: np.ndarray) -> StandardNormalizer:
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        self._mean = float(np.mean(finite)) if finite.size else 0.0
        self._std = float(np.std(finite)) if finite.size else 1.0
        if self._std < self.epsilon:
            self._std = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("StandardNormalizer.transform called before fit()")
        arr = np.asarray(values, dtype=np.float32)
        return (arr - np.float32(self._mean)) / np.float32(self._std)

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None and self._std is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self._mean,
            "std": self._std,
            "epsilon": self.epsilon,
        }


def make_normalizer(name: str) -> Normalizer:
    """Registry: ``"identity"`` or ``"standard"``."""
    if name == "identity":
        return IdentityNormalizer()
    if name == "standard":
        return StandardNormalizer()
    raise ValueError(f"unknown normalizer {name!r}; use 'identity' or 'standard'")


@dataclass(frozen=True)
class NormalizerConfig:
    """Configuration describing which normalizers to apply per feature group.

    Phase 2 defaults to identity (analytical features).  ``standard`` is
    available for future fitted features; it MUST be fit on train only.
    """

    market: str = "identity"
    account: str = "identity"
    time: str = "identity"

    def __post_init__(self) -> None:
        for name in (self.market, self.account, self.time):
            if name not in ("identity", "standard"):
                raise ValueError(f"unsupported normalizer {name!r}; use 'identity' or 'standard'")
