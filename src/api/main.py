"""FastAPI composition root — wires backends + drift monitoring (ADR-0002).

`SENTIMENT_BACKEND` (`stub` | `catboost` | `lora`, default `stub`) picks
the classifier; non-stub backends are resolved through the MLflow Model
Registry (ADR-0004) with the existing filesystem adapter as fallback for
offline dev. Non-stub backends also get a `DriftMonitorPort` on
`app.state.drift_monitor`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

from sentiment.adapters.in_memory_drift_monitor import InMemoryDriftMonitor
from sentiment.adapters.mlflow_registry_classifier import (
    RegistryVersionInfo,
    load_from_registry_or_fallback,
)
from sentiment.adapters.stub_classifier import StubClassifier
from sentiment.domain.classifier import SentimentClassifierPort
from sentiment.domain.drift import DriftMonitorPort, DriftReport, SignalReport
from sentiment.domain.models import Sentiment, SentimentResult

logger = logging.getLogger("api.main")

_VALID_BACKENDS: tuple[str, ...] = ("stub", "catboost", "lora")
_DEFAULT_LORA_DIR = Path("models/arabert-lora-v1")
_DEFAULT_CATBOOST_DIR = Path("models/catboost-baseline-v1")
_BACKEND_NAMES: dict[str, str] = {
    "stub": "stub",
    "catboost": "catboost-baseline-v1",
    "lora": "arabert-lora-v1",
}

_LABEL_ORDER: tuple[Sentiment, ...] = (Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL)
_DEFAULT_BUFFER_SIZE = 1000
_MINIMUM_COUNT = 50
_DEFAULT_REPORTS_DIR = Path("reports")


class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float


def _build_classifier(
    backend: str,
) -> tuple[SentimentClassifierPort, RegistryVersionInfo | None]:
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"unknown SENTIMENT_BACKEND={backend!r}; expected one of {_VALID_BACKENDS}"
        )
    if backend == "stub":
        return StubClassifier(), None
    requested_version = os.environ.get("MODEL_VERSION")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    if backend == "catboost":
        fallback_dir = Path(os.environ.get("CATBOOST_MODEL_DIR", _DEFAULT_CATBOOST_DIR)).resolve()
        return load_from_registry_or_fallback(
            backend="catboost",
            fallback_dir=fallback_dir,
            requested_version=requested_version,
            tracking_uri=tracking_uri,
        )
    fallback_dir = Path(os.environ.get("LORA_MODEL_DIR", _DEFAULT_LORA_DIR)).resolve()
    return load_from_registry_or_fallback(
        backend="lora",
        fallback_dir=fallback_dir,
        requested_version=requested_version,
        tracking_uri=tracking_uri,
    )


def _load_reference(
    backend_name: str, reports_dir: Path
) -> tuple[
    dict[Sentiment, float] | None,
    dict[str, float] | None,
]:
    report_path = reports_dir / f"{backend_name}.json"
    if not report_path.is_file():
        logger.info("reference missing: %s (file not found)", report_path)
        return None, None
    with report_path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)

    pred_ref = _extract_predicted_class_reference(report, report_path)
    conf_ref = _extract_confidence_reference(report, report_path)
    return pred_ref, conf_ref


def _extract_predicted_class_reference(
    report: dict[str, object], report_path: Path
) -> dict[Sentiment, float] | None:
    try:
        matrix = report["confusion_matrix"]
        dim = len(_LABEL_ORDER)
        column_totals = [sum(matrix[r][c] for r in range(dim)) for c in range(dim)]  # type: ignore[index]
        total = sum(column_totals)
        return {
            label: count / total for label, count in zip(_LABEL_ORDER, column_totals, strict=True)
        }
    except (KeyError, TypeError, IndexError, ZeroDivisionError):
        logger.info("reference missing: confusion_matrix in %s", report_path)
        return None


def _extract_confidence_reference(
    report: dict[str, object], report_path: Path
) -> dict[str, float] | None:
    histogram = report.get("confidence_histogram")
    if isinstance(histogram, dict) and all(isinstance(v, int | float) for v in histogram.values()):
        return {str(k): float(v) for k, v in histogram.items()}
    logger.info("reference missing: confidence_histogram in %s", report_path)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    backend = os.environ.get("SENTIMENT_BACKEND", "stub")
    classifier, version_info = _build_classifier(backend)
    app.state.classifier = classifier
    app.state.model_version = version_info
    app.state.backend_name = _BACKEND_NAMES[backend]

    raw_size = os.environ.get("DRIFT_BUFFER_SIZE")
    try:
        buffer_size = int(raw_size) if raw_size is not None else _DEFAULT_BUFFER_SIZE
    except ValueError as exc:
        raise ValueError(f"DRIFT_BUFFER_SIZE must be an integer >= 1, got {raw_size!r}") from exc
    if buffer_size < 1:
        raise ValueError(f"DRIFT_BUFFER_SIZE must be >= 1, got {buffer_size}")
    reports_dir = Path(os.environ.get("DRIFT_REPORTS_DIR", _DEFAULT_REPORTS_DIR))

    if backend == "stub":
        app.state.drift_monitor = None
        logger.info("drift backend=stub — monitor disabled")
    else:
        pred_ref, conf_ref = _load_reference(app.state.backend_name, reports_dir)
        app.state.drift_monitor = InMemoryDriftMonitor(
            backend_name=app.state.backend_name,
            predicted_class_reference=pred_ref,
            confidence_bucket_reference=conf_ref,
            buffer_size=buffer_size,
            minimum_count=_MINIMUM_COUNT,
        )
        mv_label = (
            f"{version_info.name}/{version_info.version}" if version_info else "filesystem-fallback"
        )
        logger.info(
            "drift backend=%s buffer_size=%d minimum_count=%d pred_ref=%s conf_ref=%s mv=%s",
            app.state.backend_name,
            buffer_size,
            _MINIMUM_COUNT,
            "loaded" if pred_ref else "missing",
            "loaded" if conf_ref else "missing",
            mv_label,
        )
    yield


def get_classifier(request: Request) -> SentimentClassifierPort:
    return request.app.state.classifier


def get_drift_monitor(request: Request) -> DriftMonitorPort | None:
    return request.app.state.drift_monitor


def _record_safely(monitor: DriftMonitorPort | None, result: SentimentResult) -> None:
    if monitor is None:
        return
    try:
        monitor.record(result.sentiment, result.confidence)
    except Exception:
        logger.warning("drift recording failed", exc_info=True)


def _serialize_signal(signal: SignalReport) -> dict[str, object]:
    payload: dict[str, object] = {
        "psi": signal.psi,
        "drift_level": signal.drift_level.value if signal.drift_level is not None else None,
        "observed": signal.observed,
    }
    if signal.reference_missing:
        payload["reference_missing"] = True
    else:
        payload["reference"] = signal.reference
    return payload


def _drift_report_to_dict(report: DriftReport) -> dict[str, object]:
    return {
        "backend": report.backend,
        "observed_count": report.observed_count,
        "buffer_size": report.buffer_size,
        "minimum_count": report.minimum_count,
        "insufficient_data": report.insufficient_data,
        "signals": {
            "predicted_class": _serialize_signal(report.predicted_class),
            "confidence_bucket": _serialize_signal(report.confidence_bucket),
        },
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="Arabic Sentiment MLOps",
        version="0.5.0",
        description="Sentiment analysis for Arabic text (UAE dialect + MSA).",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        version_info: RegistryVersionInfo | None = request.app.state.model_version
        return {
            "status": "ok",
            "model": request.app.state.backend_name,
            "model_version": dataclasses.asdict(version_info) if version_info else None,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        req: PredictRequest,
        classifier: SentimentClassifierPort = Depends(get_classifier),
        drift_monitor: DriftMonitorPort | None = Depends(get_drift_monitor),
    ) -> PredictResponse:
        try:
            result = classifier.predict(req.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            logger.exception("inference failed")
            raise HTTPException(status_code=500, detail="internal inference error") from None
        _record_safely(drift_monitor, result)
        return PredictResponse(
            text=result.text,
            sentiment=result.sentiment.value,
            confidence=result.confidence,
        )

    @app.get("/metrics/drift")
    def metrics_drift(
        monitor: DriftMonitorPort | None = Depends(get_drift_monitor),
    ) -> dict[str, object]:
        if monitor is None:
            raise HTTPException(
                status_code=503,
                detail="drift monitoring disabled for stub backend",
            )
        return _drift_report_to_dict(monitor.report())

    return app


app = create_app()
