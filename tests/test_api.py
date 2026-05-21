"""API integration tests — Phase 3 backend dispatch + Phase 4 drift monitoring."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app, get_classifier, get_drift_monitor
from sentiment.domain.classifier import SentimentClassifierPort
from sentiment.domain.drift import (
    DriftLevel,
    DriftMonitorPort,
    DriftReport,
    SignalReport,
)
from sentiment.domain.models import Sentiment, SentimentResult


@pytest.fixture
def stub_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """App in default stub mode (no env vars set)."""
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    with TestClient(create_app()) as client:
        yield client


def test_health_returns_200(stub_client: TestClient) -> None:
    response = stub_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "stub"


def test_predict_arabic_text_returns_stub_shape(stub_client: TestClient) -> None:
    response = stub_client.post("/predict", json={"text": "مرحبا بالعالم"})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] in ("positive", "negative", "neutral")
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["text"] == "مرحبا بالعالم"


def test_predict_empty_text_returns_422(stub_client: TestClient) -> None:
    response = stub_client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_whitespace_only_returns_422(stub_client: TestClient) -> None:
    response = stub_client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_lifespan_loads_stub_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    app = create_app()
    with TestClient(app) as client:
        assert app.state.backend_name == "stub"
        body = client.get("/health").json()
        assert body["model"] == "stub"


def test_lifespan_loads_lora_when_env_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    init_calls: list[Path] = []

    class FakeLoRA(SentimentClassifierPort):
        def __init__(self, model_dir: Path, base_model: str = "ignored") -> None:
            init_calls.append(Path(model_dir))

        def predict(self, text: str) -> SentimentResult:
            return SentimentResult(text=text, sentiment=Sentiment.POSITIVE, confidence=0.9)

    monkeypatch.setattr("api.main.AraBERTLoRAAdapter", FakeLoRA)
    monkeypatch.setenv("SENTIMENT_BACKEND", "lora")
    monkeypatch.setenv("LORA_MODEL_DIR", str(tmp_path))

    app = create_app()
    with TestClient(app) as client:
        assert app.state.backend_name == "arabert-lora-v1"
        body = client.get("/health").json()
        assert body["model"] == "arabert-lora-v1"

    assert init_calls == [tmp_path.resolve()]


def test_lifespan_loads_catboost_when_env_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_calls: list[Path] = []

    class FakeCatBoost(SentimentClassifierPort):
        def __init__(self, model_dir: Path) -> None:
            init_calls.append(Path(model_dir))

        def predict(self, text: str) -> SentimentResult:
            return SentimentResult(text=text, sentiment=Sentiment.NEGATIVE, confidence=0.7)

    monkeypatch.setattr("api.main.CatBoostAdapter", FakeCatBoost)
    monkeypatch.setenv("SENTIMENT_BACKEND", "catboost")
    monkeypatch.setenv("CATBOOST_MODEL_DIR", str(tmp_path))

    app = create_app()
    with TestClient(app) as client:
        assert app.state.backend_name == "catboost-baseline-v1"
        body = client.get("/health").json()
        assert body["model"] == "catboost-baseline-v1"

    assert init_calls == [tmp_path.resolve()]


def test_lifespan_fails_fast_on_unknown_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTIMENT_BACKEND", "bogus")
    app = create_app()
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        with TestClient(app):
            pass
    msg = str(excinfo.value)
    assert "bogus" in msg
    for name in ("stub", "catboost", "lora"):
        assert name in msg


def test_lifespan_fails_fast_when_lora_model_dir_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "definitely-not-here"
    monkeypatch.setenv("SENTIMENT_BACKEND", "lora")
    monkeypatch.setenv("LORA_MODEL_DIR", str(missing))
    app = create_app()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        with TestClient(app):
            pass


class _RecordingFakeClassifier(SentimentClassifierPort):
    def __init__(self, result: SentimentResult) -> None:
        self._result = result
        self.calls: list[str] = []

    def predict(self, text: str) -> SentimentResult:
        self.calls.append(text)
        return self._result


def test_predict_uses_overridden_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    fake = _RecordingFakeClassifier(
        SentimentResult(text="recorded", sentiment=Sentiment.POSITIVE, confidence=0.93)
    )
    app = create_app()
    app.dependency_overrides[get_classifier] = lambda: fake
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "الفندق ممتاز"})
    assert response.status_code == 200
    body = response.json()
    assert body == {"text": "recorded", "sentiment": "positive", "confidence": 0.93}
    assert fake.calls == ["الفندق ممتاز"]


class _RaisingClassifier(SentimentClassifierPort):
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def predict(self, text: str) -> SentimentResult:
        raise self._exc


def test_predict_returns_500_when_classifier_raises_unexpected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    user_text = "نص سري لا يجب تسريبه"
    app = create_app()
    app.dependency_overrides[get_classifier] = lambda: _RaisingClassifier(
        RuntimeError("boom internal trace 0xdeadbeef")
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"text": user_text})
    assert response.status_code == 500
    body = response.json()
    assert body == {"detail": "internal inference error"}
    raw = response.text
    for leak in ("boom", "0xdeadbeef", "Traceback", "RuntimeError", user_text):
        assert leak not in raw


def test_predict_returns_422_when_classifier_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    app = create_app()
    app.dependency_overrides[get_classifier] = lambda: _RaisingClassifier(
        ValueError("text must not be empty")
    )
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "non-empty"})
    assert response.status_code == 422
    assert response.json()["detail"] == "text must not be empty"


# ---------------------------------------------------------------------------
# Phase 4 — /metrics/drift + recording instrumentation
# ---------------------------------------------------------------------------


class _RecordingMonitor(DriftMonitorPort):
    def __init__(self, raise_on_record: bool = False) -> None:
        self.records: list[tuple[Sentiment, float]] = []
        self._raise_on_record = raise_on_record

    def record(self, label: Sentiment, confidence: float) -> None:
        if self._raise_on_record:
            raise RuntimeError("drift trace 0xfeedface boom")
        self.records.append((label, confidence))

    def report(self) -> DriftReport:
        raise NotImplementedError


class _FixedReportMonitor(DriftMonitorPort):
    def __init__(self, report: DriftReport) -> None:
        self._report = report

    def record(self, label: Sentiment, confidence: float) -> None:
        pass

    def report(self) -> DriftReport:
        return self._report


def _full_report() -> DriftReport:
    return DriftReport(
        backend="arabert-lora-v1",
        observed_count=50,
        buffer_size=1000,
        minimum_count=50,
        insufficient_data=False,
        predicted_class=SignalReport(
            psi=0.087,
            drift_level=DriftLevel.STABLE,
            reference={"positive": 0.6, "negative": 0.2, "neutral": 0.2},
            observed={"positive": 0.65, "negative": 0.17, "neutral": 0.18},
            reference_missing=False,
        ),
        confidence_bucket=SignalReport(
            psi=0.124,
            drift_level=DriftLevel.MODERATE,
            reference={"low": 0.10, "medium": 0.25, "high": 0.65},
            observed={"low": 0.18, "medium": 0.30, "high": 0.52},
            reference_missing=False,
        ),
    )


def _insufficient_report() -> DriftReport:
    return DriftReport(
        backend="arabert-lora-v1",
        observed_count=12,
        buffer_size=1000,
        minimum_count=50,
        insufficient_data=True,
        predicted_class=SignalReport(
            psi=None,
            drift_level=None,
            reference={"positive": 0.6, "negative": 0.2, "neutral": 0.2},
            observed={"positive": 0.83, "negative": 0.08, "neutral": 0.08},
            reference_missing=False,
        ),
        confidence_bucket=SignalReport(
            psi=None,
            drift_level=None,
            reference={"low": 0.10, "medium": 0.25, "high": 0.65},
            observed={"low": 0.17, "medium": 0.25, "high": 0.58},
            reference_missing=False,
        ),
    )


def test_predict_records_to_drift_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    fake_classifier = _RecordingFakeClassifier(
        SentimentResult(text="recorded", sentiment=Sentiment.POSITIVE, confidence=0.93)
    )
    monitor = _RecordingMonitor()
    app = create_app()
    app.dependency_overrides[get_classifier] = lambda: fake_classifier
    app.dependency_overrides[get_drift_monitor] = lambda: monitor
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "الفندق ممتاز"})
    assert response.status_code == 200
    assert monitor.records == [(Sentiment.POSITIVE, 0.93)]


def test_metrics_drift_returns_both_psi_signals_after_enough_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    app = create_app()
    app.dependency_overrides[get_drift_monitor] = lambda: _FixedReportMonitor(_full_report())
    with TestClient(app) as client:
        response = client.get("/metrics/drift")
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "arabert-lora-v1"
    assert body["observed_count"] == 50
    assert body["insufficient_data"] is False
    pc = body["signals"]["predicted_class"]
    cb = body["signals"]["confidence_bucket"]
    assert isinstance(pc["psi"], float)
    assert pc["drift_level"] == "stable"
    assert pc["reference"] == {"positive": 0.6, "negative": 0.2, "neutral": 0.2}
    assert pc.get("reference_missing") is None  # field omitted when reference present
    assert isinstance(cb["psi"], float)
    assert cb["drift_level"] == "moderate"


def test_metrics_drift_returns_insufficient_data_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    app = create_app()
    app.dependency_overrides[get_drift_monitor] = lambda: _FixedReportMonitor(
        _insufficient_report()
    )
    with TestClient(app) as client:
        response = client.get("/metrics/drift")
    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_data"] is True
    pc = body["signals"]["predicted_class"]
    cb = body["signals"]["confidence_bucket"]
    assert pc["psi"] is None
    assert pc["drift_level"] is None
    assert pc["reference"] is not None  # baseline still rendered for the dashboard
    assert pc["observed"] is not None
    assert cb["psi"] is None
    assert cb["drift_level"] is None
    assert cb["reference"] is not None


def test_metrics_drift_returns_503_for_stub_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    with TestClient(create_app()) as client:
        response = client.get("/metrics/drift")
    assert response.status_code == 503
    assert response.json() == {"detail": "drift monitoring disabled for stub backend"}


def test_metrics_drift_marks_signal_reference_missing_when_field_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeLoRA(SentimentClassifierPort):
        def __init__(self, model_dir: Path, base_model: str = "ignored") -> None:
            pass

        def predict(self, text: str) -> SentimentResult:
            return SentimentResult(text=text, sentiment=Sentiment.POSITIVE, confidence=0.9)

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    # Report has confidence_histogram but NO confusion_matrix.
    (reports_dir / "arabert-lora-v1.json").write_text(
        json.dumps({"confidence_histogram": {"low": 0.1, "medium": 0.25, "high": 0.65}}),
        encoding="utf-8",
    )

    monkeypatch.setattr("api.main.AraBERTLoRAAdapter", FakeLoRA)
    monkeypatch.setenv("SENTIMENT_BACKEND", "lora")
    monkeypatch.setenv("LORA_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("DRIFT_REPORTS_DIR", str(reports_dir))

    with TestClient(create_app()) as client:
        response = client.get("/metrics/drift")
    assert response.status_code == 200
    body = response.json()
    pc = body["signals"]["predicted_class"]
    cb = body["signals"]["confidence_bucket"]
    assert pc["reference_missing"] is True
    assert pc["psi"] is None
    assert pc["drift_level"] is None
    assert "reference" not in pc  # omitted when reference_missing
    assert pc["observed"] is not None
    # Confidence-bucket reference IS loaded; the only reason psi is null here
    # is insufficient_data (no predictions served yet) — exercised numerically in step-5 smoke.
    assert cb.get("reference_missing") is None
    assert cb["reference"] == {"low": 0.1, "medium": 0.25, "high": 0.65}


def test_predict_succeeds_when_drift_monitor_record_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    user_text = "نص حساس لا يجب تسريبه أبدًا"
    fake_classifier = _RecordingFakeClassifier(
        SentimentResult(text="recorded", sentiment=Sentiment.POSITIVE, confidence=0.93)
    )
    raising_monitor = _RecordingMonitor(raise_on_record=True)
    app = create_app()
    app.dependency_overrides[get_classifier] = lambda: fake_classifier
    app.dependency_overrides[get_drift_monitor] = lambda: raising_monitor
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": user_text})
    assert response.status_code == 200
    assert response.json() == {
        "text": "recorded",
        "sentiment": "positive",
        "confidence": 0.93,
    }
    raw = response.text
    for leak in ("0xfeedface", "boom", "Traceback", "RuntimeError", user_text):
        assert leak not in raw


def test_metrics_drift_response_omits_text_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    arabic_text = "السرّيّة محفوظة"
    fake_classifier = _RecordingFakeClassifier(
        SentimentResult(text=arabic_text, sentiment=Sentiment.POSITIVE, confidence=0.93)
    )
    # Real InMemoryDriftMonitor via override — proves the buffer's record()
    # cannot accept text, and the serializer never reaches text.
    from sentiment.adapters.in_memory_drift_monitor import InMemoryDriftMonitor

    monitor = InMemoryDriftMonitor(
        backend_name="arabert-lora-v1",
        predicted_class_reference={
            Sentiment.POSITIVE: 0.6,
            Sentiment.NEGATIVE: 0.2,
            Sentiment.NEUTRAL: 0.2,
        },
        confidence_bucket_reference={"low": 0.1, "medium": 0.25, "high": 0.65},
        buffer_size=10,
        minimum_count=1,
    )
    app = create_app()
    app.dependency_overrides[get_classifier] = lambda: fake_classifier
    app.dependency_overrides[get_drift_monitor] = lambda: monitor
    with TestClient(app) as client:
        client.post("/predict", json={"text": arabic_text})
        response = client.get("/metrics/drift")
    assert response.status_code == 200
    assert arabic_text not in response.text
    # No Arabic chars at all in the body.
    import re

    assert re.search(r"[؀-ۿݐ-ݿ]", response.text) is None


def test_lifespan_rejects_invalid_drift_buffer_size(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLoRA(SentimentClassifierPort):
        def __init__(self, model_dir: Path, base_model: str = "ignored") -> None:
            pass

        def predict(self, text: str) -> SentimentResult:
            return SentimentResult(text=text, sentiment=Sentiment.POSITIVE, confidence=0.9)

    monkeypatch.setattr("api.main.AraBERTLoRAAdapter", FakeLoRA)
    monkeypatch.setenv("SENTIMENT_BACKEND", "lora")
    monkeypatch.setenv("LORA_MODEL_DIR", "/tmp")
    monkeypatch.setenv("DRIFT_BUFFER_SIZE", "0")
    app = create_app()
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        with TestClient(app):
            pass
    assert "DRIFT_BUFFER_SIZE" in str(excinfo.value)
