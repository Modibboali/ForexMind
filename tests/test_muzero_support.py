"""Support-transform tests (Stage 4.1, refreshed for the Stage 4.4 transform).

The canonical transform is the MuZero Appendix F form documented in
``forexmind/muzero/support.py``::

    h(x) = sign(x) * (sqrt(|x| + 1) - 1) + epsilon * x

so the support covers raw scaled targets in ``[-3, +3]`` with ``epsilon = 0``.
"""

from __future__ import annotations

import pytest
import torch
from forexmind.muzero.support import (
    SUPPORT_RANGE,
    inverse_transform_to_scalar,
    saturation_fraction,
    scalar_to_support,
    support_expected_value,
    support_to_scalar,
    transform_to_scalar,
)

SUPPORT_SIZE = 21


def _u_grid(size: int) -> torch.Tensor:
    """The ``[-1, 1]`` support grid."""
    return -1.0 + 2.0 * torch.arange(size, dtype=torch.float32) / (size - 1)


def _value_grid(size: int, *, epsilon: float = 0.0) -> torch.Tensor:
    """Raw values whose transform lands exactly on the support grid."""
    return inverse_transform_to_scalar(_u_grid(size), epsilon=epsilon)


# --------------------------------------------------------------------------- #
# Transform pair
# --------------------------------------------------------------------------- #


def test_transform_is_odd_and_monotone() -> None:
    x = torch.linspace(-5.0, 5.0, 101)
    y = transform_to_scalar(x)
    assert torch.allclose(y, -transform_to_scalar(-x), atol=1e-6)
    assert bool((y[1:] > y[:-1]).all())


def test_transform_is_half_slope_near_zero() -> None:
    """``h'(0) = 1/2``: the transform halves small magnitudes (and expands them back)."""
    x = torch.tensor([1e-6, 1e-4, 1e-3])
    assert torch.allclose(transform_to_scalar(x), x * 0.5, rtol=1e-3)


def test_transform_maps_plus_minus_three_onto_the_support_edges() -> None:
    assert transform_to_scalar(torch.tensor([3.0])).item() == pytest.approx(1.0)
    assert transform_to_scalar(torch.tensor([-3.0])).item() == pytest.approx(-1.0)
    assert SUPPORT_RANGE == 3.0


def test_transform_round_trips_exactly() -> None:
    x = torch.linspace(-50.0, 50.0, 401)
    assert torch.allclose(inverse_transform_to_scalar(transform_to_scalar(x)), x, atol=1e-2)


@pytest.mark.parametrize("epsilon", [0.0, 0.001, 0.1])
def test_transform_round_trips_with_epsilon(epsilon: float) -> None:
    x = torch.linspace(-10.0, 10.0, 201)
    y = transform_to_scalar(x, epsilon=epsilon)
    assert torch.allclose(inverse_transform_to_scalar(y, epsilon=epsilon), x, atol=1e-2)


def test_negative_epsilon_is_rejected() -> None:
    with pytest.raises(ValueError):
        transform_to_scalar(torch.zeros(1), epsilon=-0.1)
    with pytest.raises(ValueError):
        inverse_transform_to_scalar(torch.zeros(1), epsilon=-0.1)


def test_scalar_to_support_is_a_valid_distribution() -> None:
    x = torch.linspace(-1.0, 1.0, 41)
    probs = scalar_to_support(x, SUPPORT_SIZE)
    assert probs.shape == (41, SUPPORT_SIZE)
    assert torch.all(probs >= 0.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(41), atol=1e-5)


def test_scalar_to_support_endpoints_are_one_hot() -> None:
    probs = scalar_to_support(torch.tensor([-SUPPORT_RANGE, SUPPORT_RANGE]), SUPPORT_SIZE)
    assert probs[0, 0].item() == pytest.approx(1.0)
    assert probs[0, 1:].sum().item() == pytest.approx(0.0)
    assert probs[1, -1].item() == pytest.approx(1.0)
    assert probs[1, :-1].sum().item() == pytest.approx(0.0)


def test_scalar_to_support_zero_lands_on_centre_bin() -> None:
    probs = scalar_to_support(torch.tensor([0.0]), SUPPORT_SIZE)
    assert probs[0, (SUPPORT_SIZE - 1) // 2].item() == pytest.approx(1.0)


def test_scalar_to_support_saturates_beyond_the_range() -> None:
    probs = scalar_to_support(torch.tensor([-50.0, 50.0]), SUPPORT_SIZE)
    assert probs[0, 0].item() == pytest.approx(1.0)
    assert probs[1, -1].item() == pytest.approx(1.0)


def test_scale_moves_the_representable_range() -> None:
    inside = scalar_to_support(torch.tensor([3e-3]), SUPPORT_SIZE, scale=1e-3)
    outside = scalar_to_support(torch.tensor([9e-3]), SUPPORT_SIZE, scale=1e-3)
    assert inside[0, -1].item() == pytest.approx(1.0)
    assert outside[0, -1].item() == pytest.approx(1.0)  # saturates
    assert saturation_fraction(torch.tensor([3e-3]), scale=1e-3) == pytest.approx(0.0)
    assert saturation_fraction(torch.tensor([9e-3]), scale=1e-3) == pytest.approx(1.0)


def test_saturation_fraction_is_zero_for_small_targets() -> None:
    tiny = torch.full((64,), 1e-5)
    assert saturation_fraction(tiny, scale=1e-3) == pytest.approx(0.0)


def test_support_to_scalar_centre_logits_give_zero() -> None:
    logits = torch.zeros(3, SUPPORT_SIZE)  # uniform -> symmetric centre
    value = support_to_scalar(logits, SUPPORT_SIZE)
    assert value.shape == (3,)
    assert torch.allclose(value, torch.zeros(3), atol=1e-6)


def test_round_trip_is_exact_on_support_grid() -> None:
    values = _value_grid(SUPPORT_SIZE)
    logits = torch.log(scalar_to_support(values, SUPPORT_SIZE).clamp_min(1e-30))
    recovered = support_to_scalar(logits, SUPPORT_SIZE)
    assert torch.allclose(recovered, values, atol=1e-4)


def test_round_trip_with_epsilon() -> None:
    values = _value_grid(SUPPORT_SIZE, epsilon=0.001)
    probs = scalar_to_support(values, SUPPORT_SIZE, epsilon=0.001)
    recovered = support_to_scalar(torch.log(probs.clamp_min(1e-30)), SUPPORT_SIZE, epsilon=0.001)
    assert torch.allclose(recovered, values, atol=2e-3)


def test_round_trip_generic_values_with_scaled_units() -> None:
    scale = 1e-3
    values = torch.tensor([-2.4e-3, -9e-4, 0.0, 3e-4, 1.1e-3, 2.9e-3])
    probs = scalar_to_support(values, SUPPORT_SIZE, scale=scale)
    recovered = support_to_scalar(torch.log(probs.clamp_min(1e-30)), SUPPORT_SIZE, scale=scale)
    assert torch.allclose(recovered, values, atol=scale * 0.4)


def test_expected_support_value_matches_decoded_direction() -> None:
    values = _value_grid(SUPPORT_SIZE)
    logits = torch.log(scalar_to_support(values, SUPPORT_SIZE).clamp_min(1e-30))
    expected = support_expected_value(logits, SUPPORT_SIZE)
    assert torch.allclose(expected, _u_grid(SUPPORT_SIZE), atol=1e-4)
    assert expected.min() >= -1.0 and expected.max() <= 1.0


@pytest.mark.parametrize("bad_size", [0, 2, 4, 20, -3])
def test_support_size_must_be_odd_and_ge_three(bad_size: int) -> None:
    with pytest.raises(ValueError):
        scalar_to_support(torch.zeros(1), bad_size)
    with pytest.raises(ValueError):
        support_to_scalar(torch.zeros(1, max(bad_size, 1)), bad_size)
    with pytest.raises(ValueError):
        support_expected_value(torch.zeros(1, max(bad_size, 1)), bad_size)


def test_support_to_scalar_validates_last_dimension() -> None:
    with pytest.raises(ValueError):
        support_to_scalar(torch.zeros(2, SUPPORT_SIZE + 1), SUPPORT_SIZE)


def test_scale_must_be_positive() -> None:
    with pytest.raises(ValueError):
        scalar_to_support(torch.zeros(1), SUPPORT_SIZE, scale=0.0)
    with pytest.raises(ValueError):
        support_to_scalar(torch.zeros(1, SUPPORT_SIZE), SUPPORT_SIZE, scale=-1.0)
    with pytest.raises(ValueError):
        saturation_fraction(torch.zeros(1), scale=0.0)


def test_negative_epsilon_rejected() -> None:
    with pytest.raises(ValueError):
        scalar_to_support(torch.zeros(1), SUPPORT_SIZE, epsilon=-0.1)
    with pytest.raises(ValueError):
        support_to_scalar(torch.zeros(1, SUPPORT_SIZE), SUPPORT_SIZE, epsilon=-0.1)


def test_scalar_to_support_scalar_input_rejected() -> None:
    with pytest.raises(ValueError):
        scalar_to_support(torch.tensor(0.5), SUPPORT_SIZE)
