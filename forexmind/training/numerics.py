"""Numerical-stability utilities for RL training (Phase 3).

These helpers exist to find and diagnose the *first* non-finite value in the
training pipeline instead of only catching the final ``torch.distributions.
Normal(...)`` error.  They never silently replace NaN/Inf with zeros in
production code.

* :class:`FiniteError`      - descriptive exception for the first non-finite value
* :func:`tensor_stats`      - cheap min/max/mean/std + finite/nan/inf counts
* :func:`first_nonfinite_index` - flat index of the first non-finite element
* :func:`assert_finite`     - log or raise (``strict=True``) when non-finite
* :func:`parameter_stats`   - per-parameter finite check + max |value|
* :func:`grad_norm_stats`   - total/max/mean grad norms + nan/inf counts
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class FiniteError(RuntimeError):
    """Raised on the first detected non-finite value (debug mode)."""

    def __init__(
        self,
        message: str,
        *,
        component: str = "",
        context: dict[str, object] | None = None,
    ) -> None:
        self.component = component
        self.context = dict(context or {})
        super().__init__(message)


def _as_array(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def tensor_stats(value: Any) -> dict[str, Any]:
    """Descriptive stats for a tensor/ndarray, robust to non-finite values.

    min/max/mean/std are computed over the *finite* subset so a single
    ``nan`` does not hide the finite range of the rest.
    """
    arr = _as_array(value)
    total = int(arr.size)
    if total == 0:
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "total_count": 0,
            "finite_count": 0,
            "nan_count": 0,
            "inf_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    finite = np.isfinite(arr)
    n_finite = int(np.count_nonzero(finite))
    n_nan = int(np.count_nonzero(np.isnan(arr)))
    n_inf = total - n_finite - n_nan
    finite_vals = arr[finite]
    min_v: float | None
    max_v: float | None
    mean_v: float | None
    std_v: float | None
    if n_finite:
        f = finite_vals.astype(np.float64)
        min_v = float(np.min(f))
        max_v = float(np.max(f))
        mean_v = float(np.mean(f))
        std_v = float(np.std(f))
    else:
        min_v = max_v = mean_v = std_v = None
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "total_count": total,
        "finite_count": n_finite,
        "nan_count": n_nan,
        "inf_count": n_inf,
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "std": std_v,
    }


def first_nonfinite_index(value: Any) -> int | None:
    """Flat index of the first non-finite element, or ``None`` if all finite."""
    arr = _as_array(value)
    finite = np.isfinite(arr)
    if finite.all():
        return None
    return int(np.flatnonzero(~finite)[0])


def _sample_values(value: Any, index: int, radius: int = 3) -> list[float]:
    arr = _as_array(value).ravel()
    lo = max(0, index - radius)
    hi = min(len(arr), index + radius + 1)
    return [float(v) for v in arr[lo:hi]]


def format_finite_message(
    name: str,
    stats: dict[str, object],
    *,
    index: int | None,
    context: dict[str, object] | None,
    value: Any = None,
) -> str:
    """Build a human-readable first-non-finite report."""
    ctx = context or {}
    lines = [
        "=" * 64,
        f"FIRST NON-FINITE VALUE: {name}",
        "=" * 64,
        f"  component:      {name}",
        f"  training step:  {ctx.get('gradient_updates', '?')}",
        f"  env step:       {ctx.get('env_steps', '?')}",
        f"  instrument:     {ctx.get('instrument', '?')}",
        f"  timestamp:      {ctx.get('timestamp', '?')}",
        f"  shape:          {stats['shape']}",
        f"  dtype:          {stats['dtype']}",
        f"  min (finite):   {stats['min']}",
        f"  max (finite):   {stats['max']}",
        f"  mean (finite):  {stats['mean']}",
        f"  finite_count:   {stats['finite_count']}",
        f"  total_count:    {stats['total_count']}",
        f"  nan_count:      {stats['nan_count']}",
        f"  inf_count:      {stats['inf_count']}",
    ]
    if index is not None:
        lines.append(f"  first bad index: {index}")
        if value is not None:
            lines.append(f"  surrounding values: {_sample_values(value, index)}")
    for k, v in ctx.items():
        if k not in ("gradient_updates", "env_steps", "instrument", "timestamp"):
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def assert_finite(
    name: str,
    value: Any,
    *,
    context: dict[str, object] | None = None,
    strict: bool = False,
) -> dict[str, object]:
    """Check ``value`` for non-finite entries.

    Returns the stats dict always.  When a non-finite value is found it is
    logged immediately; when ``strict=True`` (debug/validation mode) it raises
    :class:`FiniteError` so training stops at the *first* offending tensor.
    """
    stats = tensor_stats(value)
    n_bad = int(stats["total_count"]) - int(stats["finite_count"])
    if n_bad == 0:
        return stats
    index = first_nonfinite_index(value)
    message = format_finite_message(name, stats, index=index, context=context, value=value)
    if strict:
        raise FiniteError(message, component=name, context=context)
    print(message, flush=True)
    return stats


def parameter_stats(module: torch.nn.Module) -> dict[str, float]:
    """Finite check + max |param| for every parameter of a module."""
    max_abs = 0.0
    n_nan = 0
    n_inf = 0
    n_total = 0
    for p in module.parameters():
        arr = p.detach().cpu().numpy()
        n_total += int(arr.size)
        n_nan += int(np.count_nonzero(np.isnan(arr)))
        n_inf += int(np.count_nonzero(~np.isfinite(arr))) - int(np.count_nonzero(np.isnan(arr)))
        if arr.size:
            max_abs = max(max_abs, float(np.max(np.abs(arr[np.isfinite(arr)]))))
    return {
        "max_abs_param": max_abs,
        "nan_params": n_nan,
        "inf_params": n_inf,
        "total_params": n_total,
    }


def grad_norm_stats(module: torch.nn.Module) -> dict[str, float]:
    """Total/mean grad norms and nan/inf grad counts for a module."""
    total_sq = 0.0
    total_abs = 0.0
    n_elems = 0
    n_nan = 0
    n_inf = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().cpu().numpy()
        n_elems += int(g.size)
        finite = np.isfinite(g)
        total_sq += float(np.sum(np.square(g[finite], dtype=np.float64)))
        total_abs += float(np.sum(np.abs(g[finite], dtype=np.float64)))
        n_nan += int(np.count_nonzero(np.isnan(g)))
        n_inf += int(np.count_nonzero(~finite)) - int(np.count_nonzero(np.isnan(g)))
    total_norm = float(np.sqrt(total_sq))
    return {
        "grad_norm": total_norm,
        "grad_mean_abs": (total_abs / n_elems) if n_elems else 0.0,
        "grad_nan": n_nan,
        "grad_inf": n_inf,
    }


def first_bad_tensor(values: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Return ``(name, stats)`` of the first non-finite tensor in an ordered dict.

    Used to scan the pipeline in the documented order (obs -> reward -> value
    -> returns -> advantages -> actor output -> ...).
    """
    for name, value in values.items():
        if value is None:
            continue
        stats = tensor_stats(value)
        if int(stats["finite_count"]) < int(stats["total_count"]):
            return name, stats
    return None
