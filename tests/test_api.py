"""API integration tests — Phase 3 (backend dispatch + DI seam)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from sentiment.domain.classifier import SentimentClassifierPort
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
