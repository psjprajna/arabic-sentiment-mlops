"""Smoke tests for the Phase 0 walking skeleton."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_predict_arabic_text_returns_stub_shape() -> None:
    response = client.post("/predict", json={"text": "مرحبا بالعالم"})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] in ("positive", "negative", "neutral")
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["text"] == "مرحبا بالعالم"


def test_predict_empty_text_returns_422() -> None:
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_whitespace_only_returns_422() -> None:
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422
