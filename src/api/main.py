"""FastAPI composition root for Arabic Sentiment MLOps.

Selects the sentiment backend at startup via ``SENTIMENT_BACKEND``
(``stub`` | ``catboost`` | ``lora``; default ``stub``) and stores the
constructed adapter on ``app.state`` for the lifetime of the process.
Adapters live behind the ``SentimentClassifierPort`` — this module is the
only place where concrete adapter classes are named (ADR-0002).
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

from sentiment.adapters.arabert_lora_classifier import AraBERTLoRAAdapter
from sentiment.adapters.catboost_classifier import CatBoostAdapter
from sentiment.adapters.stub_classifier import StubClassifier
from sentiment.domain.classifier import SentimentClassifierPort

logger = logging.getLogger("api.main")

_VALID_BACKENDS: tuple[str, ...] = ("stub", "catboost", "lora")
_DEFAULT_LORA_DIR = Path("models/arabert-lora-v1")
_DEFAULT_CATBOOST_DIR = Path("models/catboost-baseline-v1")
_BACKEND_NAMES: dict[str, str] = {
    "stub": "stub",
    "catboost": "catboost-baseline-v1",
    "lora": "arabert-lora-v1",
}


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


def _build_classifier(backend: str) -> SentimentClassifierPort:
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"unknown SENTIMENT_BACKEND={backend!r}; expected one of {_VALID_BACKENDS}"
        )
    if backend == "stub":
        return StubClassifier()
    if backend == "catboost":
        model_dir = Path(os.environ.get("CATBOOST_MODEL_DIR", _DEFAULT_CATBOOST_DIR)).resolve()
        return CatBoostAdapter(model_dir=model_dir)
    model_dir = Path(os.environ.get("LORA_MODEL_DIR", _DEFAULT_LORA_DIR)).resolve()
    return AraBERTLoRAAdapter(model_dir=model_dir)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    backend = os.environ.get("SENTIMENT_BACKEND", "stub")
    app.state.classifier = _build_classifier(backend)
    app.state.backend_name = _BACKEND_NAMES[backend]
    logger.info("loaded backend=%s model=%s", backend, app.state.backend_name)
    yield


def get_classifier(request: Request) -> SentimentClassifierPort:
    return request.app.state.classifier


def create_app() -> FastAPI:
    app = FastAPI(
        title="Arabic Sentiment MLOps",
        version="0.3.0",
        description="Sentiment analysis for Arabic text (UAE dialect + MSA).",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "model": request.app.state.backend_name}

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        req: PredictRequest,
        classifier: SentimentClassifierPort = Depends(get_classifier),
    ) -> PredictResponse:
        try:
            result = classifier.predict(req.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            logger.exception("inference failed")
            raise HTTPException(status_code=500, detail="internal inference error") from None
        return PredictResponse(
            text=result.text,
            sentiment=result.sentiment.value,
            confidence=result.confidence,
        )

    return app


app = create_app()
