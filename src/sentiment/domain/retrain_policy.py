"""Pure decision: given a /metrics/drift response, should we retrain?

No infrastructure imports allowed (ADR-0002 — enforced by tests/test_fitness.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from sentiment.domain.drift import DriftLevel

_DEFAULT_MIN_OBSERVED: Final[int] = 200
_DEFAULT_SIGNALS: Final[frozenset[str]] = frozenset({"predicted_class", "confidence_bucket"})


@dataclass(frozen=True, slots=True)
class RetrainGate:
    required_level: DriftLevel = DriftLevel.SIGNIFICANT
    minimum_observed_count: int = _DEFAULT_MIN_OBSERVED
    signals_to_check: frozenset[str] = field(default_factory=lambda: _DEFAULT_SIGNALS)


@dataclass(frozen=True, slots=True)
class RetrainDecision:
    should_retrain: bool
    reason: str
    observed_count: int
    triggering_signals: tuple[tuple[str, float], ...] = ()


def should_retrain(drift_report: dict[str, object], *, gate: RetrainGate) -> RetrainDecision:
    """Apply the gate to a `/metrics/drift` response dict."""
    insufficient = bool(drift_report.get("insufficient_data", True))
    observed = int(drift_report.get("observed_count", 0))
    if insufficient:
        return RetrainDecision(False, "insufficient_data", observed)
    if observed < gate.minimum_observed_count:
        return RetrainDecision(
            False,
            f"observed_count {observed} below threshold {gate.minimum_observed_count}",
            observed,
        )
    signals = drift_report.get("signals", {})
    if not isinstance(signals, dict):
        return RetrainDecision(False, "signals not a dict", observed)
    triggering: list[tuple[str, float]] = []
    for key in gate.signals_to_check:
        sig = signals.get(key)
        if not isinstance(sig, dict):
            continue
        level = sig.get("drift_level")
        if level == gate.required_level.value:
            psi = float(sig.get("psi", 0.0))
            triggering.append((key, psi))
    if not triggering:
        return RetrainDecision(
            False,
            f"no signal at required level {gate.required_level.value}",
            observed,
        )
    reason = "; ".join(f"{name} psi={psi:.3f}" for name, psi in triggering)
    return RetrainDecision(
        True,
        f"{len(triggering)} signal(s) at required level: {reason}",
        observed,
        tuple(triggering),
    )
