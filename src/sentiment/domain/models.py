"""Domain models for Arabic sentiment analysis.

No infrastructure imports allowed in this module (ADR-0002).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class SentimentResult:
    text: str
    sentiment: Sentiment
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not self.text or not self.text.strip():
            raise ValueError("text must not be empty")
