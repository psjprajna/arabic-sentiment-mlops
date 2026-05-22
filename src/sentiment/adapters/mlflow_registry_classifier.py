"""MLflow Model Registry adapter implementing SentimentClassifierPort.

Phase 7 (ADR-0004) — makes the local file-backed MLflow Model Registry the
serving source-of-truth. ``load_from_registry_or_fallback`` resolves a model
version (pinned by ``requested_version`` or the latest non-archived), wraps the
loaded ``SentimentPyfunc`` in a port-shaped adapter, and exposes the
``RegistryVersionInfo`` that the API surfaces through ``/health``.

If the registry load fails for an *expected* reason (missing ``mlruns/``,
empty registry, MLflow error), the loader falls back to constructing the
existing filesystem adapter from ``fallback_dir`` and returns ``None`` as
``model_version`` to signal the fallback. Programmer errors (``ValueError``,
etc.) are not caught — they crash loud.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.exceptions
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

from sentiment.domain.classifier import SentimentClassifierPort
from sentiment.domain.models import Sentiment, SentimentResult

logger = logging.getLogger("sentiment.adapters.mlflow_registry_classifier")

_REGISTRY_NAMES: dict[str, str] = {
    "catboost": "catboost-baseline",
    "lora": "arabert-lora",
}


@dataclass(frozen=True)
class RegistryVersionInfo:
    name: str
    version: str
    run_id: str
    source: Literal["registry"] = "registry"


class _RegistryEmptyError(Exception):
    """Internal sentinel: search returned no versions for the requested name."""


class MLflowRegistryAdapter(SentimentClassifierPort):
    """Wraps a registry-loaded ``SentimentPyfunc`` as a SentimentClassifierPort."""

    def __init__(
        self,
        *,
        registry_name: str,
        version: str,
        run_id: str,
        tracking_uri: str = "sqlite:///mlflow.db",
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{registry_name}/{version}"
        self._pyfunc = mlflow.pyfunc.load_model(model_uri)
        self._version_info = RegistryVersionInfo(name=registry_name, version=version, run_id=run_id)

    @property
    def model_version(self) -> RegistryVersionInfo:
        return self._version_info

    def predict(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        df_out = self._pyfunc.predict(pd.DataFrame({"text": [text]}))
        row = df_out.iloc[0]
        label = str(row["sentiment"])
        try:
            sentiment = Sentiment(label)
        except ValueError as exc:
            raise ValueError(f"registry model returned unknown sentiment label: {label!r}") from exc
        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=float(row["confidence"]),
        )


def _resolve_latest_model_version(client: MlflowClient, registry_name: str) -> object:
    """Latest ModelVersion across all stages — uses ``search_model_versions``
    rather than the deprecated ``get_latest_versions`` (MLflow 2.9+, ADR-0004).
    The returned object exposes ``.version`` and ``.run_id``."""
    versions = client.search_model_versions(
        filter_string=f"name='{registry_name}'",
        order_by=["version_number DESC"],
        max_results=1,
    )
    if not versions:
        raise _RegistryEmptyError(f"no versions registered for {registry_name!r}")
    return versions[0]


_FALLBACK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    mlflow.exceptions.MlflowException,
    mlflow.exceptions.RestException,
    OSError,
    FileNotFoundError,
    _RegistryEmptyError,
)


def _build_fallback(
    backend: Literal["catboost", "lora"], fallback_dir: Path
) -> SentimentClassifierPort:
    if backend == "catboost":
        from sentiment.adapters.catboost_classifier import CatBoostAdapter

        return CatBoostAdapter(model_dir=fallback_dir)
    if backend == "lora":
        from sentiment.adapters.arabert_lora_classifier import AraBERTLoRAAdapter

        return AraBERTLoRAAdapter(model_dir=fallback_dir)
    raise ValueError(f"unknown backend: {backend!r}")


def load_from_registry_or_fallback(
    *,
    backend: Literal["catboost", "lora"],
    fallback_dir: Path,
    requested_version: str | None,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> tuple[SentimentClassifierPort, RegistryVersionInfo | None]:
    """Return ``(adapter, version_info)`` from the registry, else filesystem.

    On the registry path ``version_info`` is non-None. On the fallback path it
    is ``None`` (and a WARNING is logged).
    """
    registry_name = _REGISTRY_NAMES[backend]
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        if requested_version is None:
            mv = _resolve_latest_model_version(client, registry_name)
        else:
            mv = client.get_model_version(name=registry_name, version=requested_version)
        adapter = MLflowRegistryAdapter(
            registry_name=registry_name,
            version=str(mv.version),
            run_id=str(mv.run_id),
            tracking_uri=tracking_uri,
        )
        return adapter, adapter.model_version
    except _FALLBACK_EXCEPTIONS as exc:
        logger.warning(
            "registry load failed for backend=%s (%s: %s); using filesystem fallback at %s",
            backend,
            exc.__class__.__name__,
            exc,
            fallback_dir,
        )
        return _build_fallback(backend, fallback_dir), None
