"""FastAPI application — Arabic Sentiment MLOps (Phase 0 walking skeleton)."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from sentiment.adapters.stub_classifier import StubClassifier
from sentiment.domain.models import SentimentResult

app = FastAPI(
    title="Arabic Sentiment MLOps",
    version="0.1.0",
    description="Sentiment analysis for Arabic text (UAE dialect + MSA).",
)

_classifier = StubClassifier()


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "stub"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    result: SentimentResult = _classifier.predict(req.text)
    return PredictResponse(
        text=result.text,
        sentiment=result.sentiment.value,
        confidence=result.confidence,
    )
