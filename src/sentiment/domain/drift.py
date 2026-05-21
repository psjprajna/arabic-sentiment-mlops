"""Drift-monitoring domain: PSI math, confidence bucketing, and the
DriftMonitorPort that adapters implement.

No infrastructure imports allowed (ADR-0002 — enforced by tests/test_fitness.py).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from sentiment.domain.models import Sentiment

_LOW_UPPER = 0.6
_MEDIUM_UPPER = 0.8

_STABLE_UPPER = 0.10
_MODERATE_UPPER = 0.25


class DriftLevel(StrEnum):
    STABLE = "stable"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"


@dataclass(frozen=True)
class SignalReport:
    psi: float | None
    drift_level: DriftLevel | None
    reference: dict[str, float] | None
    observed: dict[str, float]
    reference_missing: bool


@dataclass(frozen=True)
class DriftReport:
    backend: str
    observed_count: int
    buffer_size: int
    minimum_count: int
    insufficient_data: bool
    predicted_class: SignalReport
    confidence_bucket: SignalReport


class DriftMonitorPort(ABC):
    """Records served predictions for distribution-drift reporting.

    The record signature carries no text — PII-safety is enforced at the
    port type (D31).
    """

    @abstractmethod
    def record(self, label: Sentiment, confidence: float) -> None: ...

    @abstractmethod
    def report(self) -> DriftReport: ...


def confidence_bucket(c: float) -> str:
    """Map a confidence value into `low` / `medium` / `high`.

    Half-open intervals (D34): low [0, 0.6), medium [0.6, 0.8), high [0.8, 1.0].
    Raises ValueError outside [0.0, 1.0].
    """
    if not 0.0 <= c <= 1.0:
        raise ValueError(f"confidence must be in [0.0, 1.0], got {c}")
    if c < _LOW_UPPER:
        return "low"
    if c < _MEDIUM_UPPER:
        return "medium"
    return "high"


def population_stability_index(
    observed: Mapping[str, float],
    expected: Mapping[str, float],
    epsilon: float = 1e-4,
) -> float:
    """Compute PSI between two proportion maps.

    Sums over the union of bucket names; adds `epsilon` to every bucket
    before computing the log ratio (D25 — prevents log(0) when a class is
    absent on either side).
    """
    psi = 0.0
    for key in set(observed) | set(expected):
        o = observed.get(key, 0.0) + epsilon
        e = expected.get(key, 0.0) + epsilon
        psi += (o - e) * math.log(o / e)
    return psi


def classify_psi(psi: float) -> DriftLevel:
    """Map a PSI value to a drift band per D26.

    `<0.10` STABLE, `[0.10, 0.25]` MODERATE, `>0.25` SIGNIFICANT.
    """
    if psi < _STABLE_UPPER:
        return DriftLevel.STABLE
    if psi <= _MODERATE_UPPER:
        return DriftLevel.MODERATE
    return DriftLevel.SIGNIFICANT
