"""Min-max value normalization for MuZero search (Stage 4.2).

PUCT compares a prior-exploration term (``O(1)``) against a value term, so the
backed-up values must be put on a comparable scale.  MuZero tracks the minimum
and maximum value observed during a single search and normalizes with them.

The normalizer is deliberately defensive:

* before any observation ``normalize`` returns ``0.0``;
* when ``maximum == minimum`` (a single distinct value) it returns ``0.0``
  instead of dividing by zero or returning an unbounded raw value;
* results are always finite.

Values are **not** clamped to ``[0, 1]``: a value outside the currently observed
range is a legitimate signal and MuZero's reference implementation does not
clamp it.
"""

from __future__ import annotations

from dataclasses import dataclass

_EPS = 1e-8


@dataclass(slots=True)
class MinMaxStats:
    """Tracks ``[minimum, maximum]`` of observed values for normalization."""

    minimum: float | None = None
    maximum: float | None = None
    count: int = 0

    def update(self, value: float) -> None:
        """Include ``value`` in the observed range."""
        value = float(value)
        self.count += 1
        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.maximum is None or value > self.maximum:
            self.maximum = value

    @property
    def initialized(self) -> bool:
        return self.minimum is not None and self.maximum is not None

    @property
    def spread(self) -> float:
        if not self.initialized:
            return 0.0
        assert self.minimum is not None and self.maximum is not None
        return float(self.maximum - self.minimum)

    def normalize(self, value: float) -> float:
        """Map ``value`` into the observed range; safe when uninitialized/equal."""
        if not self.initialized:
            return 0.0
        spread = self.spread
        if spread <= _EPS:
            return 0.0
        assert self.minimum is not None
        return float((float(value) - self.minimum) / spread)

    def reset(self) -> None:
        self.minimum = None
        self.maximum = None
        self.count = 0
