"""Stub adapter: always returns neutral with zero confidence.

Phase 0 only. Replace with CatBoostAdapter in Phase 1.
"""

from __future__ import annotations

from sentiment.domain.classifier import SentimentClassifierPort
from sentiment.domain.models import Sentiment, SentimentResult


class StubClassifier(SentimentClassifierPort):
    """Phase-0 stub — wires the architecture without a real model."""

    def predict(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        return SentimentResult(
            text=text,
            sentiment=Sentiment.NEUTRAL,
            confidence=0.0,
        )
