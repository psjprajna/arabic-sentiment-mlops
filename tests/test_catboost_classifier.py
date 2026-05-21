"""Tests for CatBoostAdapter (mocked — no real model on disk)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from sentiment.adapters.catboost_classifier import CatBoostAdapter, _normalize
from sentiment.domain.models import Sentiment


def _build_adapter(
    tmp_path: Path,
    *,
    proba: list[float],
    labels: list[str] | None = None,
) -> CatBoostAdapter:
    labels = labels or ["positive", "negative", "neutral"]
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.cbm").write_bytes(b"stub")
    (model_dir / "vectorizer.joblib").write_bytes(b"stub")
    (model_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    adapter = CatBoostAdapter.__new__(CatBoostAdapter)
    adapter._model = MagicMock()
    adapter._model.predict_proba.return_value = np.array([proba])
    adapter._vectorizer = MagicMock()
    adapter._vectorizer.transform.return_value = "VECTORIZED"
    adapter._labels = [Sentiment(lbl) for lbl in labels]
    return adapter


def test_predict_raises_on_empty_text(tmp_path: Path) -> None:
    adapter = _build_adapter(tmp_path, proba=[0.5, 0.3, 0.2])
    with pytest.raises(ValueError):
        adapter.predict("")


def test_predict_raises_on_whitespace(tmp_path: Path) -> None:
    adapter = _build_adapter(tmp_path, proba=[0.5, 0.3, 0.2])
    with pytest.raises(ValueError):
        adapter.predict("   ")


def test_predict_returns_original_text(tmp_path: Path) -> None:
    adapter = _build_adapter(tmp_path, proba=[0.8, 0.1, 0.1])
    raw = "مرحبا  بالعالم"  # deliberate double space
    result = adapter.predict(raw)
    assert result.text == raw


def test_predict_maps_label_index_correctly(tmp_path: Path) -> None:
    adapter = _build_adapter(
        tmp_path,
        proba=[0.1, 0.7, 0.2],
        labels=["positive", "negative", "neutral"],
    )
    result = adapter.predict("نص ما")
    assert result.sentiment is Sentiment.NEGATIVE
    assert result.confidence == pytest.approx(0.7)


def test_init_raises_when_model_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        CatBoostAdapter(missing)


def test_normalization_strips_urls_and_mentions() -> None:
    out = _normalize("رائع http://x.com @user #المنتج")
    assert "http" not in out
    assert "@user" not in out
    assert "#" not in out
    assert "المنتج" in out
    assert "رائع" in out
