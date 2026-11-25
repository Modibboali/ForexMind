"""Training progress bar (tqdm) with a graceful no-op fallback.

``tqdm`` is an optional dependency: if it is not installed the trainer falls
back to a lightweight no-op bar so training still runs (the periodic
``progress_block`` logs still print at the configured intervals).
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - simple availability probe
    from tqdm.auto import tqdm as _tqdm  # type: ignore[import-untyped]

    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _tqdm = None
    _TQDM_AVAILABLE = False


class _NoOpBar:
    """Stand-in for tqdm when the package is not installed."""

    def __init__(self, **_: Any) -> None:
        pass

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix(self, **_: Any) -> None:
        pass

    def set_description(self, *_: Any, **__: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> _NoOpBar:
        return self

    def __exit__(self, *_: Any) -> None:
        pass


def make_progress_bar(
    total: int,
    desc: str = "",
    unit: str = " steps",
    **kwargs: Any,
) -> Any:
    """Return a tqdm progress bar over ``total`` units (or a no-op fallback)."""
    if _tqdm is not None:
        return _tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, **kwargs)
    return _NoOpBar()


def tqdm_available() -> bool:
    """True when tqdm is installed and the bar will actually render."""
    return _TQDM_AVAILABLE
