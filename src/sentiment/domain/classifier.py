"""Domain port: SentimentClassifierPort.

Adapters live in sentiment/adapters/ and implement this port.
No infrastructure imports allowed here (ADR-0002).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sentiment.domain.models import SentimentResult


class SentimentClassifierPort(ABC):
    """Classify Arabic text into a sentiment category.

    Implementations must never be imported inside this module.
    Wire adapters in the composition root (api/main.py).
    """

    @abstractmethod
    def predict(self, text: str) -> SentimentResult:
        """Classify text. Raises ValueError on empty or whitespace-only input."""
        ...
