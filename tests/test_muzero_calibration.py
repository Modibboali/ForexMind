"""MuZero target-scale calibration tests (Stage 4.4 §10).

The calibration helpers exist so the support heads can be *sized to the data*
instead of hard-coded.  These tests pin the arithmetic down:

* the statistics are the ones documented (plain percentiles of the masked data),
* ``propose_scale`` really places ``abs_p99`` at ``u = target_u``,
* empty / degenerate inputs fall back instead of dividing by zero,
* the report exposes saturation so nothing is clipped silently.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from forexmind.muzero.calibration import (
    TargetStatistics,
    calibration_report,
    describe_targets,
    headroom_factor,
    propose_scale,
    reward_targets,
    value_targets,
)
from forexmind.muzero.support import (
    inverse_transform_to_scalar,
    saturation_fraction,
    transform_to_scalar,
)
from forexmind.muzero.targets import TargetConfig, build_unroll_sample, collate_samples

from tests.muzero_synthetic import make_trajectory

OBS_DIM = 6
UNROLL = 3
TD_STEPS = 3


def _batch(rewards: list[float], *, steps: int | None = None):
    """A single-row batch whose reward targets are exactly ``rewards``."""
    steps = len(rewards) if steps is None else steps
    trajectory = make_trajectory(
        actions=[0] * steps,
        rewards=list(rewards),
        obs_dim=OBS_DIM,
        trajectory_id=0,
    )
    config = TargetConfig(num_unroll_steps=UNROLL, td_steps=TD_STEPS, discount=0.99)
    return collate_samples([build_unroll_sample(trajectory, 0, config)])


# --------------------------------------------------------------------------- #
# describe_targets
# --------------------------------------------------------------------------- #


def test_describe_targets_matches_plain_percentiles() -> None:
    values = np.arange(101, dtype=np.float64)  # 0 .. 100
    stats = describe_targets(values)
    assert stats.count == 101
    assert stats.mean == pytest.approx(50.0)
    assert stats.std == pytest.approx(float(np.std(values)))
    assert stats.minimum == 0.0
    assert stats.maximum == 100.0
    assert stats.p01 == pytest.approx(1.0)
    assert stats.p05 == pytest.approx(5.0)
    assert stats.p50 == pytest.approx(50.0)
    assert stats.p95 == pytest.approx(95.0)
    assert stats.p99 == pytest.approx(99.0)
    # absolute percentiles are the percentiles of |x|, not |percentile of x|
    assert stats.abs_p50 == pytest.approx(50.0)
    assert stats.abs_p99 == pytest.approx(99.0)
    assert stats.abs_max == pytest.approx(100.0)


def test_describe_targets_is_symmetric_for_a_mirrored_signal() -> None:
    values = np.concatenate([np.linspace(-1.0, -1e-3, 50), np.linspace(1e-3, 1.0, 50)])
    stats = describe_targets(values)
    assert stats.mean == pytest.approx(0.0, abs=1e-12)
    # 99th percentile of x is positive, 99th percentile of |x| sits near the max
    assert stats.p99 == pytest.approx(float(np.percentile(values, 99.0)))
    assert stats.abs_p99 > stats.p99
    assert stats.abs_max == pytest.approx(1.0)


def test_describe_targets_handles_empty_input() -> None:
    stats = describe_targets([])
    assert stats.count == 0
    assert stats.mean == 0.0
    assert stats.std == 0.0
    assert stats.minimum == 0.0
    assert stats.maximum == 0.0
    assert stats.p99 == 0.0
    assert stats.abs_max == 0.0
    assert stats.to_dict()["count"] == 0


def test_describe_targets_handles_all_zero_input() -> None:
    stats = describe_targets([0.0] * 16)
    assert stats.count == 16
    assert stats.abs_max == 0.0
    assert stats.absolute_scale == 0.0


def test_describe_targets_drops_non_finite_values() -> None:
    stats = describe_targets([np.nan, np.inf, -np.inf, 1.0, 3.0])
    assert stats.count == 2
    assert stats.mean == pytest.approx(2.0)
    assert stats.maximum == pytest.approx(3.0)


def test_absolute_scale_falls_back_when_the_tail_is_zero() -> None:
    zeros = TargetStatistics(
        count=10,
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
    assert zeros.absolute_scale == 0.0
    # abs_p99 wins when available
    assert TargetStatistics(**{**zeros.to_dict(), "abs_p99": 2.0}).absolute_scale == 2.0
    # then abs_p95, then abs_p50, then |mean|
    assert TargetStatistics(**{**zeros.to_dict(), "abs_p95": 1.5}).absolute_scale == 1.5
    assert TargetStatistics(**{**zeros.to_dict(), "abs_p50": 0.5}).absolute_scale == 0.5
    assert TargetStatistics(**{**zeros.to_dict(), "mean": -0.25}).absolute_scale == 0.25


# --------------------------------------------------------------------------- #
# headroom_factor / propose_scale
# --------------------------------------------------------------------------- #


def test_headroom_factor_is_the_inverse_transform() -> None:
    for target_u in (0.1, 0.5, 0.6, 0.9):
        expected = float(inverse_transform_to_scalar(torch.tensor([target_u]))[0].item())
        assert headroom_factor(target_u) == pytest.approx(expected)
    # h^-1(u) = u * (|u| + 2) at epsilon = 0
    assert headroom_factor(0.6) == pytest.approx(0.6 * 2.6)


@pytest.mark.parametrize("target_u", [0.0, 1.0, -0.5, 1.5])
def test_headroom_factor_rejects_out_of_range_fractions(target_u: float) -> None:
    with pytest.raises(ValueError):
        headroom_factor(target_u)


def test_more_headroom_means_a_smaller_scale() -> None:
    stats = describe_targets(np.linspace(-1.0, 1.0, 200))
    tight = propose_scale(stats, target_u=0.9)
    roomy = propose_scale(stats, target_u=0.3)
    # a *smaller* target_u places the percentile nearer the edge, so less room
    # is left above it and the scale has to grow
    assert roomy > tight
    assert tight == pytest.approx(stats.absolute_scale / headroom_factor(0.9))
    assert roomy == pytest.approx(stats.absolute_scale / headroom_factor(0.3))


def test_propose_scale_places_abs_p99_at_target_u() -> None:
    stats = describe_targets(np.linspace(-4.0, 4.0, 401))
    scale = propose_scale(stats, target_u=0.6)
    assert scale > 0.0
    encoded = float(transform_to_scalar(torch.tensor([stats.abs_p99 / scale]))[0].item())
    assert encoded == pytest.approx(0.6, abs=1e-6)
    # the percentile therefore sits inside the representable window with margin
    assert abs(stats.abs_p99 / scale) < 3.0


def test_propose_scale_falls_back_for_degenerate_statistics() -> None:
    empty = describe_targets([])
    assert propose_scale(empty, floor=1e-9) == pytest.approx(1e-9)
    assert propose_scale(empty, floor=2.5) == pytest.approx(2.5)
    zeros = describe_targets([0.0] * 8)
    assert propose_scale(zeros, floor=1e-6) == pytest.approx(1e-6)


def test_propose_scale_never_returns_a_non_finite_scale() -> None:
    stats = describe_targets([1e-30] * 32)
    scale = propose_scale(stats)
    assert np.isfinite(scale) and scale > 0.0


# --------------------------------------------------------------------------- #
# target extraction
# --------------------------------------------------------------------------- #


def test_reward_and_value_targets_respect_the_validity_masks() -> None:
    rewards = [0.0012, -0.0008, 0.0004, -0.0002]
    batch = _batch(rewards)
    reward_values = reward_targets(batch)
    value_values = value_targets(batch)
    assert reward_values.size == int(batch.reward_masks.sum().item())
    assert value_values.size == int(batch.value_masks.sum().item())
    assert reward_values.size > 0
    # every extracted reward is one of the stored real rewards (within float32)
    for value in reward_values:
        assert min(abs(float(value) - r) for r in rewards) < 1e-7
    assert np.isfinite(value_values).all()


def test_masked_out_entries_are_excluded() -> None:
    batch = _batch([0.001, -0.002, 0.003])
    batch = replace(batch, reward_masks=torch.zeros_like(batch.reward_masks))
    assert reward_targets(batch).size == 0
    assert describe_targets(reward_targets(batch)).count == 0


def test_target_shape_mismatch_is_rejected() -> None:
    batch = _batch([0.001, -0.002, 0.003])
    bad = replace(batch, reward_masks=batch.reward_masks[:, :-1].contiguous())
    with pytest.raises(ValueError, match="shape mismatch"):
        reward_targets(bad)


# --------------------------------------------------------------------------- #
# calibration_report
# --------------------------------------------------------------------------- #


def test_calibration_report_exposes_statistics_scales_and_saturation() -> None:
    rewards = [0.0012, -0.0008, 0.0004, -0.0002]
    batch = _batch(rewards)
    report = calibration_report(batch)
    assert set(report) == {"target_u", "headroom_factor", "reward", "value"}
    assert report["headroom_factor"] == pytest.approx(headroom_factor(report["target_u"]))
    for head in ("reward", "value"):
        block = report[head]
        assert set(block) == {"statistics", "proposed_scale", "saturation_fraction"}
        assert block["statistics"]["count"] > 0
        assert block["proposed_scale"] > 0.0
        assert 0.0 <= block["saturation_fraction"] <= 1.0
    # Forex-scale rewards must not saturate the window proposed for them
    assert report["reward"]["saturation_fraction"] < 0.1
    assert report["value"]["saturation_fraction"] < 0.1


def test_calibration_report_is_json_serializable() -> None:
    import json

    report = calibration_report(_batch([0.001, -0.002, 0.003]))
    assert json.loads(json.dumps(report))["reward"]["proposed_scale"] > 0.0


def test_calibration_report_flags_a_deliberately_too_small_scale() -> None:
    batch = _batch([0.001, -0.002, 0.003, -0.004])
    tiny = calibration_report(batch)  # proposed scale is data-driven and safe
    assert tiny["reward"]["saturation_fraction"] < 0.5
    # shrinking to ~1e-9 pushes every non-zero target onto the edge bins
    assert saturation_fraction(batch.target_rewards.reshape(-1), scale=1e-9) > 0.9
    # and an enormous scale saturates nothing
    assert saturation_fraction(batch.target_rewards.reshape(-1), scale=1e9) == 0.0


def test_repeated_calibration_is_deterministic() -> None:
    batch = _batch([0.001, -0.002, 0.003, -0.004, 0.0005])
    first = calibration_report(batch)
    second = calibration_report(batch)
    assert first == second


def test_proposed_scale_round_trips_the_observed_range() -> None:
    rng = np.random.default_rng(0)
    # log-return-sized rewards spanning roughly five orders of magnitude
    rewards = (rng.standard_normal(64) * 10.0 ** rng.integers(-5, -1, 64)).astype(float).tolist()
    batch = _batch(rewards)
    report = calibration_report(batch)
    scale = report["reward"]["proposed_scale"]
    values = torch.tensor(rewards, dtype=torch.float32)
    encoded = transform_to_scalar(values / scale)
    decoded = inverse_transform_to_scalar(encoded) * scale
    assert torch.allclose(decoded, values, rtol=1e-4, atol=1e-9)
