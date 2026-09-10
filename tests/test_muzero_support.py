"""Stage 4.1 tests for the MuZero categorical support transforms.

Covers ``scalar_to_support`` / ``support_to_scalar`` shape, normalisation,
endpoints, validation, and (approximate) round-trip invertibility.
"""

from __future__ import annotations

import pytest
import torch
from forexmind.muzero.support import scalar_to_support, support_to_scalar

SUPPORT_SIZE = 21


def _grid(size: int) -> torch.Tensor:
    return -1.0 + 2.0 * torch.arange(size, dtype=torch.float32) / (size - 1)


def test_scalar_to_support_is_a_valid_distribution() -> None:
    x = torch.linspace(-1.0, 1.0, 41)
    probs = scalar_to_support(x, SUPPORT_SIZE)
    assert probs.shape == (41, SUPPORT_SIZE)
    assert torch.all(probs >= 0.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(41), atol=1e-5)


def test_scalar_to_support_endpoints_are_one_hot() -> None:
    probs = scalar_to_support(torch.tensor([-1.0, 1.0]), SUPPORT_SIZE)
    assert probs[0, 0].item() == pytest.approx(1.0)
    assert probs[0, 1:].sum().item() == pytest.approx(0.0)
    assert probs[1, -1].item() == pytest.approx(1.0)
    assert probs[1, :-1].sum().item() == pytest.approx(0.0)


def test_scalar_to_support_zero_lands_on_centre_bin() -> None:
    probs = scalar_to_support(torch.tensor([0.0]), SUPPORT_SIZE)
    assert probs[0, (SUPPORT_SIZE - 1) // 2].item() == pytest.approx(1.0)


def test_scalar_to_support_clamps_out_of_range() -> None:
    probs = scalar_to_support(torch.tensor([-5.0, 5.0]), SUPPORT_SIZE)
    assert probs[0, 0].item() == pytest.approx(1.0)
    assert probs[1, -1].item() == pytest.approx(1.0)


def test_support_to_scalar_centre_logits_give_zero() -> None:
    logits = torch.zeros(3, SUPPORT_SIZE)  # uniform -> symmetric centre
    value = support_to_scalar(logits, SUPPORT_SIZE)
    assert value.shape == (3,)
    assert torch.allclose(value, torch.zeros(3), atol=1e-6)


def test_round_trip_is_exact_on_support_grid() -> None:
    grid = _grid(SUPPORT_SIZE)
    logits = torch.log(scalar_to_support(grid, SUPPORT_SIZE).clamp_min(1e-30))
    recovered = support_to_scalar(logits, SUPPORT_SIZE, epsilon=0.0)
    assert torch.allclose(recovered, grid, atol=1e-4)


def test_round_trip_with_epsilon_matches_muzero() -> None:
    grid = _grid(SUPPORT_SIZE)
    probs = scalar_to_support(grid, SUPPORT_SIZE, epsilon=0.001)
    logits = torch.log(probs.clamp_min(1e-30))
    recovered = support_to_scalar(logits, SUPPORT_SIZE, epsilon=0.001)
    assert torch.allclose(recovered, grid, atol=2e-3)


def test_round_trip_generic_values() -> None:
    x = torch.tensor([-0.83, -0.37, 0.0, 0.11, 0.62, 0.99])
    probs = scalar_to_support(x, SUPPORT_SIZE)
    recovered = support_to_scalar(torch.log(probs.clamp_min(1e-30)), SUPPORT_SIZE, epsilon=0.0)
    assert torch.allclose(recovered, x, atol=0.02)


@pytest.mark.parametrize("bad_size", [0, 2, 4, 20, -3])
def test_support_size_must_be_odd_and_ge_three(bad_size: int) -> None:
    with pytest.raises(ValueError):
        scalar_to_support(torch.zeros(1), bad_size)
    with pytest.raises(ValueError):
        support_to_scalar(torch.zeros(1, max(bad_size, 1)), bad_size)


def test_support_to_scalar_validates_last_dimension() -> None:
    with pytest.raises(ValueError):
        support_to_scalar(torch.zeros(2, SUPPORT_SIZE + 1), SUPPORT_SIZE)


def test_negative_epsilon_rejected() -> None:
    with pytest.raises(ValueError):
        scalar_to_support(torch.zeros(1), SUPPORT_SIZE, epsilon=-0.1)
    with pytest.raises(ValueError):
        support_to_scalar(torch.zeros(1, SUPPORT_SIZE), SUPPORT_SIZE, epsilon=-0.1)


def test_scalar_to_support_scalar_input_rejected() -> None:
    with pytest.raises(ValueError):
        scalar_to_support(torch.tensor(0.5), SUPPORT_SIZE)
