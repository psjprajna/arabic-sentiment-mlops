"""Tests for the domain drift module (PSI + confidence bucketing) and the
in-memory drift monitor adapter.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor

import pytest

from sentiment.adapters.in_memory_drift_monitor import InMemoryDriftMonitor
from sentiment.domain.drift import (
    DriftLevel,
    classify_psi,
    confidence_bucket,
    population_stability_index,
)
from sentiment.domain.models import Sentiment

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


# ---------------------------------------------------------------------------
# Adapter behavior
# ---------------------------------------------------------------------------


_CLASS_REF: dict[Sentiment, float] = {
    Sentiment.POSITIVE: 0.60,
    Sentiment.NEGATIVE: 0.20,
    Sentiment.NEUTRAL: 0.20,
}
_BUCKET_REF: dict[str, float] = {"low": 0.10, "medium": 0.25, "high": 0.65}


def _make_monitor(
    *,
    predicted_class_reference: dict[Sentiment, float] | None = None,
    confidence_bucket_reference: dict[str, float] | None = None,
    buffer_size: int = 100,
    minimum_count: int = 10,
) -> InMemoryDriftMonitor:
    if predicted_class_reference is None and confidence_bucket_reference is None:
        # Default both populated unless a test wants to ablate one.
        predicted_class_reference = dict(_CLASS_REF)
        confidence_bucket_reference = dict(_BUCKET_REF)
    return InMemoryDriftMonitor(
        backend_name="test-backend-v1",
        predicted_class_reference=predicted_class_reference,
        confidence_bucket_reference=confidence_bucket_reference,
        buffer_size=buffer_size,
        minimum_count=minimum_count,
    )


@pytest.mark.parametrize(
    "buffer_size,minimum_count",
    [(0, 1), (10, 0), (5, 6)],
    ids=["buffer-zero", "min-zero", "min-exceeds-buffer"],
)
def test_constructor_rejects_invalid_sizes(buffer_size: int, minimum_count: int) -> None:
    with pytest.raises(ValueError):
        InMemoryDriftMonitor(
            backend_name="x",
            predicted_class_reference=dict(_CLASS_REF),
            confidence_bucket_reference=dict(_BUCKET_REF),
            buffer_size=buffer_size,
            minimum_count=minimum_count,
        )


def test_record_then_report_below_minimum_returns_insufficient_data() -> None:
    monitor = _make_monitor(minimum_count=10)
    for _ in range(5):
        monitor.record(Sentiment.POSITIVE, 0.9)
    report = monitor.report()
    assert report.insufficient_data is True
    assert report.observed_count == 5
    assert report.predicted_class.psi is None
    assert report.predicted_class.drift_level is None
    assert report.predicted_class.reference is not None  # baseline still rendered
    assert report.confidence_bucket.psi is None
    assert report.confidence_bucket.drift_level is None
    assert report.confidence_bucket.reference is not None


def test_record_then_report_at_minimum_returns_both_signal_psi() -> None:
    monitor = _make_monitor(minimum_count=10)
    for _ in range(6):
        monitor.record(Sentiment.POSITIVE, 0.9)
    for _ in range(2):
        monitor.record(Sentiment.NEGATIVE, 0.7)
    for _ in range(2):
        monitor.record(Sentiment.NEUTRAL, 0.5)
    report = monitor.report()
    assert report.insufficient_data is False
    assert report.observed_count == 10
    assert isinstance(report.predicted_class.psi, float)
    assert report.predicted_class.drift_level in {
        DriftLevel.STABLE,
        DriftLevel.MODERATE,
        DriftLevel.SIGNIFICANT,
    }
    assert isinstance(report.confidence_bucket.psi, float)
    assert report.confidence_bucket.drift_level in {
        DriftLevel.STABLE,
        DriftLevel.MODERATE,
        DriftLevel.SIGNIFICANT,
    }


def test_report_observed_proportions_sum_to_one() -> None:
    monitor = _make_monitor(minimum_count=4)
    monitor.record(Sentiment.POSITIVE, 0.9)
    monitor.record(Sentiment.POSITIVE, 0.7)
    monitor.record(Sentiment.NEGATIVE, 0.55)
    monitor.record(Sentiment.NEUTRAL, 0.5)
    report = monitor.report()
    assert math.isclose(sum(report.predicted_class.observed.values()), 1.0)
    assert math.isclose(sum(report.confidence_bucket.observed.values()), 1.0)


def test_buffer_wraps_after_capacity() -> None:
    monitor = _make_monitor(buffer_size=50, minimum_count=10)
    # First batch — should be evicted.
    for _ in range(50):
        monitor.record(Sentiment.POSITIVE, 0.9)
    # Second batch — keeps the buffer pinned at NEGATIVE only.
    for _ in range(150):
        monitor.record(Sentiment.NEGATIVE, 0.55)
    report = monitor.report()
    assert report.observed_count == 50
    assert report.predicted_class.observed[Sentiment.POSITIVE.value] == 0.0
    assert report.predicted_class.observed[Sentiment.NEGATIVE.value] == 1.0
    assert report.confidence_bucket.observed["low"] == 1.0
    assert report.confidence_bucket.observed["high"] == 0.0


def test_report_handles_missing_predicted_class_reference() -> None:
    monitor = _make_monitor(
        predicted_class_reference=None,
        confidence_bucket_reference=dict(_BUCKET_REF),
        minimum_count=4,
    )
    for _ in range(4):
        monitor.record(Sentiment.POSITIVE, 0.9)
    report = monitor.report()
    assert report.predicted_class.reference_missing is True
    assert report.predicted_class.psi is None
    assert report.predicted_class.drift_level is None
    assert report.predicted_class.reference is None
    assert report.predicted_class.observed  # still populated
    # Other signal still reports normally.
    assert report.confidence_bucket.reference_missing is False
    assert isinstance(report.confidence_bucket.psi, float)


def test_report_handles_missing_confidence_reference() -> None:
    monitor = _make_monitor(
        predicted_class_reference=dict(_CLASS_REF),
        confidence_bucket_reference=None,
        minimum_count=4,
    )
    for _ in range(4):
        monitor.record(Sentiment.POSITIVE, 0.9)
    report = monitor.report()
    assert report.confidence_bucket.reference_missing is True
    assert report.confidence_bucket.psi is None
    assert report.confidence_bucket.drift_level is None
    assert report.confidence_bucket.reference is None
    assert report.confidence_bucket.observed
    assert report.predicted_class.reference_missing is False
    assert isinstance(report.predicted_class.psi, float)


def test_concurrent_records_do_not_lose_writes() -> None:
    monitor = _make_monitor(buffer_size=200, minimum_count=10)

    def _do_record(_i: int) -> None:
        monitor.record(Sentiment.POSITIVE, 0.9)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_do_record, range(100)))

    report = monitor.report()
    assert report.observed_count == 100


_ARABIC_CHAR_RE = re.compile(r"[؀-ۿݐ-ݿ]")


def test_buffer_contents_contain_no_text() -> None:
    monitor = _make_monitor(minimum_count=4)
    arabic_inputs = [
        ("هذا المنتج رائع جداً", Sentiment.POSITIVE, 0.93),
        ("خدمة سيئة للغاية", Sentiment.NEGATIVE, 0.71),
        ("الطعام مقبول", Sentiment.NEUTRAL, 0.52),
        ("السرّيّة محفوظة", Sentiment.POSITIVE, 0.81),
    ]
    for _text, label, conf in arabic_inputs:
        # Adapter must accept only (label, confidence) — no text channel exists.
        monitor.record(label, conf)
    buffer_repr = repr(monitor._buffer)  # type: ignore[attr-defined]
    assert _ARABIC_CHAR_RE.search(buffer_repr) is None, buffer_repr
    for text, _label, _conf in arabic_inputs:
        assert text not in buffer_repr
