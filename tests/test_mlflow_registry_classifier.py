"""Unit tests for MLflowRegistryAdapter + load_from_registry_or_fallback.

Tests stub `mlflow.pyfunc.load_model` and `MlflowClient` — no real `mlruns/`
or filesystem registry is touched. Drives Phase 7 (ADR-0004) registry-first
serving.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from sentiment.adapters.mlflow_registry_classifier import (
    MLflowRegistryAdapter,
    RegistryVersionInfo,
    load_from_registry_or_fallback,
)
from sentiment.domain.classifier import SentimentClassifierPort
from sentiment.domain.models import Sentiment, SentimentResult


@dataclass
class _FakeModelVersion:
    name: str
    version: str
    run_id: str


class _FakePyfunc:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert list(df.columns) == ["text"]
        assert len(df) == len(self._rows)
        return pd.DataFrame(self._rows)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip env that could leak between tests."""
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    yield


def _install_fake_load_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: list[dict[str, object]],
) -> list[str]:
    captured_uris: list[str] = []

    def _fake_load_model(model_uri: str) -> _FakePyfunc:
        captured_uris.append(model_uri)
        return _FakePyfunc(rows)

    monkeypatch.setattr(
        "sentiment.adapters.mlflow_registry_classifier.mlflow.pyfunc.load_model",
        _fake_load_model,
    )
    return captured_uris


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    get_model_version_returns: _FakeModelVersion | Exception | None = None,
    search_returns: list[_FakeModelVersion] | Exception | None = None,
) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    class _FakeClient:
        def __init__(self, tracking_uri: str | None = None) -> None:
            captured.append({"init_tracking_uri": tracking_uri})

        def get_model_version(self, name: str, version: str) -> _FakeModelVersion:
            captured.append({"get_model_version": (name, version)})
            if isinstance(get_model_version_returns, Exception):
                raise get_model_version_returns
            assert isinstance(get_model_version_returns, _FakeModelVersion)
            return get_model_version_returns

        def search_model_versions(
            self,
            filter_string: str | None = None,
            max_results: int = 100,
            order_by: list[str] | None = None,
            page_token: str | None = None,
        ) -> list[_FakeModelVersion]:
            captured.append(
                {
                    "search": {
                        "filter_string": filter_string,
                        "order_by": order_by,
                        "max_results": max_results,
                    }
                }
            )
            if isinstance(search_returns, Exception):
                raise search_returns
            assert search_returns is not None
            return list(search_returns)

    monkeypatch.setattr(
        "sentiment.adapters.mlflow_registry_classifier.MlflowClient",
        _FakeClient,
    )
    return captured


def test_adapter_loads_specific_version(monkeypatch: pytest.MonkeyPatch) -> None:
    uris = _install_fake_load_model(
        monkeypatch,
        rows=[{"text": "x", "sentiment": "positive", "confidence": 0.9}],
    )

    adapter = MLflowRegistryAdapter(
        registry_name="catboost-baseline",
        version="3",
        run_id="run-3",
        tracking_uri="sqlite:///mlflow.db",
    )

    assert uris == ["models:/catboost-baseline/3"]
    assert isinstance(adapter, SentimentClassifierPort)


def test_predict_converts_pyfunc_df_to_sentiment_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_load_model(
        monkeypatch,
        rows=[{"text": "x", "sentiment": "positive", "confidence": 0.91}],
    )
    adapter = MLflowRegistryAdapter(registry_name="arabert-lora", version="1", run_id="run-1")

    result = adapter.predict("الفيلم رائع")

    assert result == SentimentResult(
        text="الفيلم رائع",
        sentiment=Sentiment.POSITIVE,
        confidence=0.91,
    )


def test_predict_rejects_unknown_label_with_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_load_model(
        monkeypatch,
        rows=[{"text": "x", "sentiment": "garbage", "confidence": 0.5}],
    )
    adapter = MLflowRegistryAdapter(registry_name="catboost-baseline", version="2", run_id="r")

    with pytest.raises(ValueError, match="garbage"):
        adapter.predict("نص")


def test_predict_rejects_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_load_model(monkeypatch, rows=[])
    adapter = MLflowRegistryAdapter(registry_name="catboost-baseline", version="1", run_id="r")

    with pytest.raises(ValueError, match="empty"):
        adapter.predict("   ")


def test_model_version_property_exposes_name_version_runid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_load_model(
        monkeypatch,
        rows=[{"text": "x", "sentiment": "neutral", "confidence": 0.5}],
    )

    adapter = MLflowRegistryAdapter(registry_name="arabert-lora", version="11", run_id="run-abc")

    assert adapter.model_version == RegistryVersionInfo(
        name="arabert-lora",
        version="11",
        run_id="run-abc",
        source="registry",
    )


def test_loader_resolves_latest_when_requested_version_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    uris = _install_fake_load_model(
        monkeypatch,
        rows=[{"text": "x", "sentiment": "positive", "confidence": 0.9}],
    )
    captured = _install_fake_client(
        monkeypatch,
        search_returns=[
            _FakeModelVersion(name="catboost-baseline", version="7", run_id="latest-run"),
        ],
    )

    classifier, info = load_from_registry_or_fallback(
        backend="catboost",
        fallback_dir=tmp_path,
        requested_version=None,
    )

    assert uris == ["models:/catboost-baseline/7"]
    assert info == RegistryVersionInfo(
        name="catboost-baseline", version="7", run_id="latest-run", source="registry"
    )
    search_calls = [c["search"] for c in captured if "search" in c]
    assert search_calls == [
        {
            "filter_string": "name='catboost-baseline'",
            "order_by": ["version_number DESC"],
            "max_results": 1,
        }
    ]
    assert isinstance(classifier, MLflowRegistryAdapter)


def test_loader_uses_requested_version_via_get_model_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_load_model(
        monkeypatch,
        rows=[{"text": "x", "sentiment": "negative", "confidence": 0.6}],
    )
    captured = _install_fake_client(
        monkeypatch,
        get_model_version_returns=_FakeModelVersion(
            name="arabert-lora", version="2", run_id="pinned-run"
        ),
    )

    _, info = load_from_registry_or_fallback(
        backend="lora",
        fallback_dir=tmp_path,
        requested_version="2",
    )

    assert info is not None
    assert info == RegistryVersionInfo(
        name="arabert-lora", version="2", run_id="pinned-run", source="registry"
    )
    assert any(c.get("get_model_version") == ("arabert-lora", "2") for c in captured)
    assert all("search" not in c for c in captured)


def test_loader_falls_back_on_mlflow_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fallback_calls: list[Path] = []

    class _FakeCatBoost(SentimentClassifierPort):
        def __init__(self, model_dir: Path) -> None:
            fallback_calls.append(Path(model_dir))

        def predict(self, text: str) -> SentimentResult:
            return SentimentResult(text=text, sentiment=Sentiment.POSITIVE, confidence=0.5)

    monkeypatch.setattr("sentiment.adapters.catboost_classifier.CatBoostAdapter", _FakeCatBoost)

    def _exploding_load_model(model_uri: str) -> object:
        raise mlflow.exceptions.MlflowException("registry unreachable")

    monkeypatch.setattr(
        "sentiment.adapters.mlflow_registry_classifier.mlflow.pyfunc.load_model",
        _exploding_load_model,
    )
    _install_fake_client(
        monkeypatch,
        get_model_version_returns=_FakeModelVersion(
            name="catboost-baseline", version="3", run_id="r"
        ),
    )

    with caplog.at_level(logging.WARNING):
        classifier, info = load_from_registry_or_fallback(
            backend="catboost",
            fallback_dir=tmp_path / "cb-models",
            requested_version="3",
        )

    assert info is None
    assert fallback_calls == [tmp_path / "cb-models"]
    assert isinstance(classifier, _FakeCatBoost)
    assert any("filesystem fallback" in rec.message for rec in caplog.records)


def test_loader_falls_back_on_empty_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fallback_calls: list[Path] = []

    class _FakeLora(SentimentClassifierPort):
        def __init__(self, model_dir: Path, base_model: str = "ignored") -> None:
            fallback_calls.append(Path(model_dir))

        def predict(self, text: str) -> SentimentResult:
            return SentimentResult(text=text, sentiment=Sentiment.NEUTRAL, confidence=0.5)

    monkeypatch.setattr("sentiment.adapters.arabert_lora_classifier.AraBERTLoRAAdapter", _FakeLora)
    _install_fake_load_model(
        monkeypatch,
        rows=[{"text": "x", "sentiment": "positive", "confidence": 0.9}],
    )
    _install_fake_client(monkeypatch, search_returns=[])

    with caplog.at_level(logging.WARNING):
        classifier, info = load_from_registry_or_fallback(
            backend="lora",
            fallback_dir=tmp_path / "lora-models",
            requested_version=None,
        )

    assert info is None
    assert fallback_calls == [tmp_path / "lora-models"]
    assert isinstance(classifier, _FakeLora)
    assert any("filesystem fallback" in rec.message for rec in caplog.records)


def test_loader_falls_back_on_os_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing mlruns/ dir → OSError from pyfunc.load_model."""
    fallback_calls: list[Path] = []

    class _FakeCatBoost(SentimentClassifierPort):
        def __init__(self, model_dir: Path) -> None:
            fallback_calls.append(Path(model_dir))

        def predict(self, text: str) -> SentimentResult:
            return SentimentResult(text=text, sentiment=Sentiment.POSITIVE, confidence=0.5)

    monkeypatch.setattr("sentiment.adapters.catboost_classifier.CatBoostAdapter", _FakeCatBoost)

    def _exploding_load_model(model_uri: str) -> object:
        raise FileNotFoundError("no mlruns directory")

    monkeypatch.setattr(
        "sentiment.adapters.mlflow_registry_classifier.mlflow.pyfunc.load_model",
        _exploding_load_model,
    )
    _install_fake_client(
        monkeypatch,
        get_model_version_returns=_FakeModelVersion(
            name="catboost-baseline", version="1", run_id="r"
        ),
    )

    with caplog.at_level(logging.WARNING):
        _, info = load_from_registry_or_fallback(
            backend="catboost",
            fallback_dir=tmp_path,
            requested_version="1",
        )

    assert info is None
    assert fallback_calls == [tmp_path]


def test_loader_does_not_swallow_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Programmer errors (ValueError) must surface — not silently fall back."""

    def _value_error_load_model(model_uri: str) -> object:
        raise ValueError("programmer bug: malformed config")

    monkeypatch.setattr(
        "sentiment.adapters.mlflow_registry_classifier.mlflow.pyfunc.load_model",
        _value_error_load_model,
    )
    _install_fake_client(
        monkeypatch,
        get_model_version_returns=_FakeModelVersion(
            name="catboost-baseline", version="1", run_id="r"
        ),
    )

    with pytest.raises(ValueError, match="programmer bug"):
        load_from_registry_or_fallback(
            backend="catboost",
            fallback_dir=tmp_path,
            requested_version="1",
        )
