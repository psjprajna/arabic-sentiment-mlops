"""In-memory implementation of DriftMonitorPort.

Thread-safe ring buffer of (Sentiment, confidence_bucket) tuples — the
record signature carries no text, so the buffer can never hold PII (D31).
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping

from sentiment.domain.drift import (
    DriftMonitorPort,
    DriftReport,
    SignalReport,
    classify_psi,
    confidence_bucket,
    population_stability_index,
)
from sentiment.domain.models import Sentiment

_BUCKET_KEYS: tuple[str, ...] = ("low", "medium", "high")


class InMemoryDriftMonitor(DriftMonitorPort):
    """Accumulates served-prediction tuples in a bounded ring buffer."""

    def __init__(
        self,
        backend_name: str,
        predicted_class_reference: Mapping[Sentiment, float] | None,
        confidence_bucket_reference: Mapping[str, float] | None,
        buffer_size: int = 1000,
        minimum_count: int = 50,
    ) -> None:
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
        if minimum_count < 1:
            raise ValueError(f"minimum_count must be >= 1, got {minimum_count}")
        if minimum_count > buffer_size:
            raise ValueError(
                f"minimum_count ({minimum_count}) must not exceed buffer_size ({buffer_size})"
            )

        self._backend_name = backend_name
        self._buffer_size = buffer_size
        self._minimum_count = minimum_count
        self._predicted_class_reference: dict[str, float] | None = (
            {label.value: prop for label, prop in predicted_class_reference.items()}
            if predicted_class_reference is not None
            else None
        )
        self._confidence_bucket_reference: dict[str, float] | None = (
            dict(confidence_bucket_reference) if confidence_bucket_reference is not None else None
        )
        self._buffer: deque[tuple[Sentiment, str]] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()

    def record(self, label: Sentiment, confidence: float) -> None:
        bucket = confidence_bucket(confidence)
        with self._lock:
            self._buffer.append((label, bucket))

    def report(self) -> DriftReport:
        with self._lock:
            snapshot = list(self._buffer)
        observed_count = len(snapshot)
        insufficient_data = observed_count < self._minimum_count

        class_observed = _class_proportions(snapshot)
        bucket_observed = _bucket_proportions(snapshot)

        return DriftReport(
            backend=self._backend_name,
            observed_count=observed_count,
            buffer_size=self._buffer_size,
            minimum_count=self._minimum_count,
            insufficient_data=insufficient_data,
            predicted_class=_build_signal(
                observed=class_observed,
                reference=self._predicted_class_reference,
                insufficient_data=insufficient_data,
            ),
            confidence_bucket=_build_signal(
                observed=bucket_observed,
                reference=self._confidence_bucket_reference,
                insufficient_data=insufficient_data,
            ),
        )


def _class_proportions(snapshot: list[tuple[Sentiment, str]]) -> dict[str, float]:
    counts: dict[str, int] = {label.value: 0 for label in Sentiment}
    for sentiment, _bucket in snapshot:
        counts[sentiment.value] += 1
    total = len(snapshot)
    if total == 0:
        return {key: 0.0 for key in counts}
    return {key: count / total for key, count in counts.items()}


def _bucket_proportions(snapshot: list[tuple[Sentiment, str]]) -> dict[str, float]:
    counts: dict[str, int] = {key: 0 for key in _BUCKET_KEYS}
    for _sentiment, bucket in snapshot:
        counts[bucket] += 1
    total = len(snapshot)
    if total == 0:
        return {key: 0.0 for key in counts}
    return {key: count / total for key, count in counts.items()}


def _build_signal(
    observed: dict[str, float],
    reference: dict[str, float] | None,
    insufficient_data: bool,
) -> SignalReport:
    if reference is None:
        return SignalReport(
            psi=None,
            drift_level=None,
            reference=None,
            observed=observed,
            reference_missing=True,
        )
    reference_copy = dict(reference)
    if insufficient_data:
        return SignalReport(
            psi=None,
            drift_level=None,
            reference=reference_copy,
            observed=observed,
            reference_missing=False,
        )
    psi = population_stability_index(observed, reference_copy)
    return SignalReport(
        psi=psi,
        drift_level=classify_psi(psi),
        reference=reference_copy,
        observed=observed,
        reference_missing=False,
    )
