"""CatBoost + TF-IDF adapter implementing SentimentClassifierPort.

The adapter loads a fitted CatBoost model, a fitted TfidfVectorizer, and the
index→Sentiment label mapping from a model directory at construction. Text
normalization happens inside the adapter — the `SentimentResult.text` keeps the
caller's original input verbatim (preserves tashkeel, domain rule from AGENTS.md).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
from catboost import CatBoostClassifier

from sentiment.domain.classifier import SentimentClassifierPort
from sentiment.domain.models import Sentiment, SentimentResult

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#")
_WS_RE = re.compile(r"\s+")

_MODEL_FILE = "model.cbm"
_VECTORIZER_FILE = "vectorizer.joblib"
_LABELS_FILE = "labels.json"


def _normalize(text: str) -> str:
    out = _URL_RE.sub(" ", text)
    out = _MENTION_RE.sub(" ", out)
    out = _HASHTAG_RE.sub("", out)
    out = _WS_RE.sub(" ", out).strip()
    return out


class CatBoostAdapter(SentimentClassifierPort):
    """Phase-1 adapter — CatBoost classifier over TF-IDF features."""

    def __init__(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_path = model_dir / _MODEL_FILE
        vec_path = model_dir / _VECTORIZER_FILE
        labels_path = model_dir / _LABELS_FILE
        for p in (model_path, vec_path, labels_path):
            if not p.exists():
                raise FileNotFoundError(f"missing adapter artifact: {p}")

        model = CatBoostClassifier()
        model.load_model(str(model_path))
        self._model = model
        self._vectorizer = joblib.load(vec_path)
        raw_labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self._labels: list[Sentiment] = [Sentiment(lbl) for lbl in raw_labels]

    def predict(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        normalized = _normalize(text)
        features = self._vectorizer.transform([normalized])
        proba = self._model.predict_proba(features)[0]
        idx = int(proba.argmax())
        sentiment = self._labels[idx]
        confidence = float(proba[idx])
        return SentimentResult(text=text, sentiment=sentiment, confidence=confidence)
