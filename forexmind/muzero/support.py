"""MuZero scalar/support transforms (Stages 4.1-4.4).

MuZero predicts scalars (value, reward) as a **distribution over a discrete
support** rather than unrestricted regression.  Two things are needed:

1. an invertible scalar transform that allocates resolution intelligently, and
2. a discretisation onto ``support_size`` bins spanning ``[-1, 1]``.

Scalar transform (the MuZero paper's Appendix F form, as specified in the
Stage 4.4 brief)::

    h(x)    = sign(x) * (sqrt(|x| + 1) - 1) + epsilon * x
    h^-1(y) = sign(y) * (s^2 - 1),
              s = (-1 + sqrt(1 + 4*epsilon*(1 + epsilon + |y|))) / (2*epsilon)

``h`` is monotone and odd, with ``h(x) ~ x`` near zero, so the support spends
its bins where the targets actually are.  With ``epsilon = 0`` the inverse has
the exact closed form ``h^-1(y) = sign(y) * |y| * (|y| + 2)``.

Support convention
------------------
A raw economic target ``z`` is scaled before being transformed::

    u = h(z / scale)                     # |u| <= 1 for |z/scale| <= 3
    target_distribution = two_point(u, support_size)

and predictions decode back to economic units::

    prediction = scale * h^-1(expected_support(logits))

The representable range is ``|z| <= 3 * scale`` when ``epsilon = 0`` (since
``h(+-3) = +-1``); larger magnitudes **saturate** onto the edge bins.  That is
deliberate headroom control: ``scale`` is calibrated from observed target
statistics (see :mod:`forexmind.muzero.calibration`) so meaningful targets sit
well inside the range.  Nothing is clipped silently -- :func:`saturation_fraction`
measures how much signal lands on the edges.

:func:`scalar_to_support` and :func:`support_to_scalar` take the **same**
``scale`` and ``epsilon``; mixing them silently breaks the round trip, so both
default to ``scale=1.0, epsilon=0.0`` and the loss layer passes one shared
configuration.

No softmax is applied by the networks; it is applied here for decoding only.
"""

from __future__ import annotations

import torch

__all__ = [
    "SUPPORT_RANGE",
    "inverse_transform_to_scalar",
    "saturation_fraction",
    "scalar_to_support",
    "support_expected_value",
    "support_to_scalar",
    "transform_to_scalar",
]

#: Largest ``|z / scale|`` representable by the support when ``epsilon = 0``.
SUPPORT_RANGE = 3.0

_EPS_FLOOR = 1e-12


def _check_support_size(support_size: int) -> None:
    if support_size < 3 or support_size % 2 == 0:
        raise ValueError(f"support_size must be an odd integer >= 3, got {support_size}")


def transform_to_scalar(x: torch.Tensor, *, epsilon: float = 0.0) -> torch.Tensor:
    """``h(x) = sign(x) * (sqrt(|x| + 1) - 1) + epsilon * x``.

    Evaluated as the algebraically identical ``x / (sqrt(|x| + 1) + 1) + eps * x``
    so that tiny Forex-scale magnitudes (``~1e-6``) are not destroyed by
    catastrophic cancellation in ``sqrt(1 + x) - 1`` at float32.
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    return x / (torch.sqrt(torch.abs(x) + 1.0) + 1.0) + epsilon * x


def inverse_transform_to_scalar(y: torch.Tensor, *, epsilon: float = 0.0) -> torch.Tensor:
    """Exact inverse of :func:`transform_to_scalar`.

    With ``epsilon = 0`` this is ``y * (|y| + 2)``, which is cancellation-free.
    With ``epsilon > 0`` the inverse solves ``eps*d^2 + (1 + 2*eps)*d - |y| = 0``
    for ``d = sqrt(|x| + 1) - 1``.  ``epsilon = 0`` (the default) is recommended:
    the linear term is not needed once ``scale`` is calibrated, and it degrades
    float32 precision for very small magnitudes.
    """
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    magnitude = torch.abs(y)
    if epsilon <= _EPS_FLOOR:
        expanded = magnitude * (magnitude + 2.0)
    else:
        a = 1.0 + 2.0 * epsilon
        d = (torch.sqrt(a * a + 4.0 * epsilon * magnitude) - a) / (2.0 * epsilon)
        expanded = d * (d + 2.0)
    return torch.sign(y) * torch.clamp_min(expanded, 0.0)


def _two_point(u: torch.Tensor, support_size: int) -> torch.Tensor:
    """Spread mass linearly between the two bins adjacent to ``u`` in [-1, 1]."""
    x = torch.clamp(u, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x * (support_size - 1)

    lower = torch.floor(x)
    upper = torch.clamp(lower + 1.0, max=float(support_size - 1))
    p_upper = x - lower
    p_lower = 1.0 - p_upper

    out = torch.zeros((*x.shape, support_size), dtype=x.dtype, device=x.device)
    out.scatter_add_(-1, lower.long().unsqueeze(-1), p_lower.unsqueeze(-1))
    out.scatter_add_(-1, upper.long().unsqueeze(-1), p_upper.unsqueeze(-1))
    return out


def support_expected_value(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    """Expected support value of ``logits`` in ``[-1, 1]`` (no inverse transform)."""
    _check_support_size(support_size)
    if logits.shape[-1] != support_size:
        raise ValueError(f"expected last dimension {support_size}, got {tuple(logits.shape)}")
    prob = torch.softmax(logits.float(), dim=-1)
    support = torch.arange(support_size, dtype=prob.dtype, device=prob.device)
    value = (prob * support).sum(dim=-1)
    return (value / (support_size - 1) * 2.0 - 1.0).to(dtype=logits.dtype)


def scalar_to_support(
    value: torch.Tensor,
    support_size: int,
    *,
    scale: float = 1.0,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Encode raw economic scalar(s) as a support distribution.

    Args:
        value: Any-shaped tensor of raw targets (``reward_t`` or ``z_t``).
        support_size: Odd number of bins ``>= 3``.
        scale: Characteristic magnitude; ``value / scale`` is transformed into
            the ``[-1, 1]`` support window.  Must match the scale used when
            decoding with :func:`support_to_scalar`.
        epsilon: Linear term of the scalar transform (must match decoding).

    Returns:
        Tensor of shape ``value.shape + (support_size,)`` summing to 1 over the
        last dimension.
    """
    _check_support_size(support_size)
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if value.ndim == 0:
        raise ValueError("scalar_to_support expects at least a 1-D tensor")
    u = transform_to_scalar(value.float() / scale, epsilon=epsilon)
    return _two_point(u, support_size)


def support_to_scalar(
    logits: torch.Tensor,
    support_size: int,
    *,
    scale: float = 1.0,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Decode support logits into a raw economic scalar.

    ``scale`` and ``epsilon`` must be the same values used to build the target
    with :func:`scalar_to_support`.
    """
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    expected = support_expected_value(logits, support_size)
    decoded = inverse_transform_to_scalar(expected, epsilon=epsilon)
    return (decoded * scale).to(dtype=logits.dtype)


def saturation_fraction(
    value: torch.Tensor,
    *,
    scale: float = 1.0,
    epsilon: float = 0.0,
) -> float:
    """Fraction of targets outside the representable support range.

    A non-zero value means the support is clipping real signal; calibrate a
    larger ``scale`` (or a larger ``support_size``) instead of accepting it.
    """
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if value.numel() == 0:
        return 0.0
    u = transform_to_scalar(value.float().reshape(-1) / scale, epsilon=epsilon)
    return float((u.abs() > 1.0).to(torch.float32).mean().item())
