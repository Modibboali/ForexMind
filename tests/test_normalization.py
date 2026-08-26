"""Tests for leakage-free normalization (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest
from forexmind.observation.normalization import (
    IdentityNormalizer,
    NormalizerConfig,
    StandardNormalizer,
    make_normalizer,
)


def test_identity_normalizer() -> None:
    n = IdentityNormalizer()
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(n.fit_transform(x), x)


def test_standard_normalizer_fit_train_only() -> None:
    train = np.array([0.0, 10.0, 20.0])  # mean 10, std 8.16
    normalizer = StandardNormalizer().fit(train)
    assert normalizer.is_fitted
    transformed_train = normalizer.transform(train)
    assert abs(float(np.mean(transformed_train))) < 1e-6
    assert abs(float(np.std(transformed_train)) - 1.0) < 1e-6

    # Transform on validation/test uses train statistics only.
    val = np.array([10.0, 30.0])
    out = normalizer.transform(val)
    assert np.allclose(out, (val - 10.0) / float(np.std(train)))


def test_standard_normalizer_requires_fit_first() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        StandardNormalizer().transform(np.array([1.0, 2.0]))


def test_standard_normalizer_zero_variance() -> None:
    n = StandardNormalizer().fit(np.array([5.0, 5.0, 5.0]))
    # std clamped to 1.0; transform is identity relative to mean.
    assert np.allclose(n.transform(np.array([5.0, 5.0])), 0.0)


def test_make_normalizer_registry() -> None:
    assert isinstance(make_normalizer("identity"), IdentityNormalizer)
    assert isinstance(make_normalizer("standard"), StandardNormalizer)
    with pytest.raises(ValueError):
        make_normalizer("zscore")


def test_normalizer_config() -> None:
    cfg = NormalizerConfig()
    assert cfg.market == "identity"
    with pytest.raises(ValueError):
        NormalizerConfig(market="quantile")


def test_normalizer_deterministic() -> None:
    train = np.random.default_rng(0).normal(size=1000)
    a = StandardNormalizer().fit(train).transform(train)
    b = StandardNormalizer().fit(train).transform(train)
    assert np.array_equal(a, b)
