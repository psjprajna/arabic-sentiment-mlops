"""Tests for the domain drift module (PSI + confidence bucketing).

Step 1 covers the pure-math helpers and the `DriftLevel` band classifier;
adapter behavior is added in Step 2.
"""

from __future__ import annotations

import math

import pytest

from sentiment.domain.drift import (
    DriftLevel,
    classify_psi,
    confidence_bucket,
    population_stability_index,
)

_REFERENCE_SHAPES = [
    pytest.param({"positive": 0.33, "negative": 0.33, "neutral": 0.34}, id="balanced"),
    pytest.param({"positive": 0.70, "negative": 0.20, "neutral": 0.10}, id="skewed-positive"),
    pytest.param({"positive": 0.95, "negative": 0.04, "neutral": 0.01}, id="extreme"),
]


@pytest.mark.parametrize("reference", _REFERENCE_SHAPES)
def test_psi_equals_zero_when_observed_equals_reference(
    reference: dict[str, float],
) -> None:
    assert population_stability_index(reference, reference) == 0.0


def test_psi_monotonic_under_progressive_drift() -> None:
    reference = {"positive": 0.60, "negative": 0.20, "neutral": 0.20}
    drifted = [
        {"positive": 0.55, "negative": 0.225, "neutral": 0.225},
        {"positive": 0.50, "negative": 0.250, "neutral": 0.250},
        {"positive": 0.45, "negative": 0.275, "neutral": 0.275},
        {"positive": 0.40, "negative": 0.300, "neutral": 0.300},
        {"positive": 0.30, "negative": 0.350, "neutral": 0.350},
    ]
    psi_values = [population_stability_index(observed, reference) for observed in drifted]
    for prev, curr in zip(psi_values, psi_values[1:]):
        assert prev < curr, f"PSI not strictly increasing: {psi_values}"


def test_psi_finite_when_observed_class_absent() -> None:
    reference = {"positive": 0.60, "negative": 0.20, "neutral": 0.20}
    observed = {"positive": 0.70, "negative": 0.30, "neutral": 0.0}
    psi = population_stability_index(observed, reference)
    assert math.isfinite(psi)
    assert psi > 0.0


def test_psi_finite_when_reference_class_absent() -> None:
    reference = {"positive": 0.70, "negative": 0.30, "neutral": 0.0}
    observed = {"positive": 0.60, "negative": 0.20, "neutral": 0.20}
    psi = population_stability_index(observed, reference)
    assert math.isfinite(psi)
    assert psi > 0.0


@pytest.mark.parametrize(
    "confidence,expected",
    [
        (0.0, "low"),
        (0.5999, "low"),
        (0.6, "medium"),
        (0.7999, "medium"),
        (0.8, "high"),
        (1.0, "high"),
    ],
)
def test_confidence_bucket_boundaries(confidence: float, expected: str) -> None:
    assert confidence_bucket(confidence) == expected


@pytest.mark.parametrize("invalid", [-0.1, 1.01])
def test_confidence_bucket_rejects_out_of_range(invalid: float) -> None:
    with pytest.raises(ValueError):
        confidence_bucket(invalid)


@pytest.mark.parametrize(
    "psi,expected",
    [
        (0.099, DriftLevel.STABLE),
        (0.10, DriftLevel.MODERATE),
        (0.249, DriftLevel.MODERATE),
        (0.25, DriftLevel.MODERATE),
        (0.26, DriftLevel.SIGNIFICANT),
    ],
)
def test_classify_psi_bands(psi: float, expected: DriftLevel) -> None:
    assert classify_psi(psi) == expected
