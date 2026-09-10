"""Categorical support transforms for MuZero scalar prediction (Stage 4.1).

MuZero represents scalar targets (value and reward) as a *distribution over a
discrete support* rather than unrestricted regression.  The pair of transforms
below implements the standard MuZero conversion:

* :func:`scalar_to_support` maps a scalar in ``[-1, 1]`` onto a two-point
  (piecewise-linear) probability distribution over ``support_size`` bins.
* :func:`support_to_scalar` maps logits back to a scalar via the expected
  support value and an invertible ``epsilon`` expansion.

Properties relied on elsewhere:

* ``support_size`` is odd, so bin ``(support_size - 1) / 2`` is the centre.
* The support spans ``[-1, 1]``; natural target scales are recovered by the
  ``value_scale`` / ``reward_scale`` factors in :class:`MuZeroConfig`.
* With ``epsilon == 0`` the transforms are exact inverses (up to support
  discretisation); with ``epsilon > 0`` they match the canonical MuZero
  definition and expand the representable range slightly beyond ``[-1, 1]``.

No softmax is applied by the networks; it is applied here for decoding only.
"""

from __future__ import annotations

import math

import torch


def scalar_to_support(
    x: torch.Tensor,
    support_size: int,
    *,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Encode scalar(s) ``x`` in ``[-1, 1]`` as a support distribution.

    Args:
        x: Any-shaped tensor of scaled scalars (values are clamped to
            ``[-1, 1]`` first).
        support_size: Odd number of bins ``>= 3``.
        epsilon: When ``> 0`` the inverse of the :func:`support_to_scalar`
            expansion is applied first so the pair round-trips through the
            expanded range.

    Returns:
        Tensor of shape ``x.shape + (support_size,)`` whose last dimension is a
        valid probability distribution summing to 1.
    """
    if support_size < 3 or support_size % 2 == 0:
        raise ValueError(f"support_size must be an odd integer >= 3, got {support_size}")
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    if x.ndim == 0:
        raise ValueError("scalar_to_support expects at least a 1-D tensor")

    x = x.float()
    if epsilon > 0.0:
        # Inverse of ((1 + eps) ** |x| - 1) / eps, so that decoding recovers x.
        scale = math.log1p(epsilon)
        magnitude = torch.log1p(epsilon * x.abs()) / scale
        x = torch.sign(x) * magnitude
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x * (support_size - 1)

    lower = torch.floor(x)
    upper = torch.clamp(lower + 1.0, max=float(support_size - 1))
    p_upper = x - lower
    p_lower = 1.0 - p_upper

    out = torch.zeros(
        (*x.shape, support_size),
        dtype=x.dtype,
        device=x.device,
    )
    out.scatter_add_(-1, lower.long().unsqueeze(-1), p_lower.unsqueeze(-1))
    out.scatter_add_(-1, upper.long().unsqueeze(-1), p_upper.unsqueeze(-1))
    return out


def support_to_scalar(
    logits: torch.Tensor,
    support_size: int,
    *,
    epsilon: float = 0.001,
) -> torch.Tensor:
    """Decode support logits into a scalar expected value in ``[-1, 1]``-ish.

    Args:
        logits: Tensor of shape ``[..., support_size]`` (raw network outputs).
        support_size: Odd number of bins ``>= 3``.
        epsilon: Expansion factor (0.001 matches MuZero; ``0.0`` disables it).

    Returns:
        Tensor of shape ``logits.shape[:-1]``.
    """
    if support_size < 3 or support_size % 2 == 0:
        raise ValueError(f"support_size must be an odd integer >= 3, got {support_size}")
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    if logits.shape[-1] != support_size:
        raise ValueError(f"expected last dimension {support_size}, got {tuple(logits.shape)}")

    prob = torch.softmax(logits.float(), dim=-1)
    support = torch.arange(support_size, dtype=prob.dtype, device=prob.device)
    value = (prob * support).sum(dim=-1)
    value = value / (support_size - 1)
    value = value * 2.0 - 1.0
    if epsilon > 0.0:
        value = torch.sign(value) * (((1.0 + epsilon) ** value.abs() - 1.0) / epsilon)
    return value.to(dtype=logits.dtype)
