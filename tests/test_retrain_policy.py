"""Truth-table tests for the pure retrain-on-drift gate (spec 009 §D29)."""

from __future__ import annotations

from sentiment.domain.retrain_policy import RetrainGate, should_retrain


def _make_report(
    *,
    observed: int = 500,
    insufficient: bool = False,
    signals: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "backend": "catboost-baseline-v1",
        "observed_count": observed,
        "buffer_size": 1000,
        "minimum_count": 50,
        "insufficient_data": insufficient,
        "signals": signals or {},
    }


def test_should_retrain_fires_when_one_signal_significant_and_enough_observations() -> None:
    report = _make_report(
        signals={"predicted_class": {"psi": 0.31, "drift_level": "significant"}},
    )
    decision = should_retrain(report, gate=RetrainGate())
    assert decision.should_retrain is True
    assert decision.triggering_signals == (("predicted_class", 0.31),)
    assert decision.observed_count == 500


def test_should_retrain_fires_with_both_signals_significant() -> None:
    report = _make_report(
        signals={
            "predicted_class": {"psi": 0.31, "drift_level": "significant"},
            "confidence_bucket": {"psi": 0.42, "drift_level": "significant"},
        },
    )
    decision = should_retrain(report, gate=RetrainGate())
    assert decision.should_retrain is True
    triggering = dict(decision.triggering_signals)
    assert triggering == {"predicted_class": 0.31, "confidence_bucket": 0.42}


def test_should_retrain_does_not_fire_when_insufficient_data() -> None:
    report = _make_report(insufficient=True)
    decision = should_retrain(report, gate=RetrainGate())
    assert decision.should_retrain is False
    assert "insufficient_data" in decision.reason


def test_should_retrain_does_not_fire_below_observed_count_threshold() -> None:
    report = _make_report(
        observed=50,
        signals={"predicted_class": {"psi": 0.31, "drift_level": "significant"}},
    )
    decision = should_retrain(report, gate=RetrainGate())
    assert decision.should_retrain is False
    assert "50" in decision.reason
    assert "200" in decision.reason


def test_should_retrain_does_not_fire_when_all_drift_levels_null() -> None:
    report = _make_report(
        signals={
            "predicted_class": {"psi": None, "drift_level": None},
            "confidence_bucket": {"psi": None, "drift_level": None},
        },
    )
    decision = should_retrain(report, gate=RetrainGate())
    assert decision.should_retrain is False


def test_should_retrain_does_not_fire_on_moderate_only() -> None:
    report = _make_report(
        signals={"predicted_class": {"psi": 0.15, "drift_level": "moderate"}},
    )
    decision = should_retrain(report, gate=RetrainGate())
    assert decision.should_retrain is False
    assert "significant" in decision.reason


def test_should_retrain_does_not_fire_on_empty_signals_dict() -> None:
    report = _make_report(signals={})
    decision = should_retrain(report, gate=RetrainGate())
    assert decision.should_retrain is False
