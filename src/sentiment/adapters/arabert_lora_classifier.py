"""AraBERT + LoRA adapter implementing SentimentClassifierPort.

Loads a base BERT classifier head plus PEFT LoRA weights from disk and the
index→Sentiment label mapping. Tokenizer comes from the same model_dir (PEFT
saves it alongside the adapter). The `SentimentResult.text` keeps the caller's
original input verbatim — tashkeel preserved, no normalization (domain rule
from AGENTS.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment.domain.classifier import SentimentClassifierPort
from sentiment.domain.models import Sentiment, SentimentResult

_ADAPTER_CONFIG_FILE = "adapter_config.json"
_LABELS_FILE = "labels.json"
_MAX_LENGTH = 128
_DEFAULT_BASE_MODEL = "aubmindlab/bert-base-arabertv2"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AraBERTLoRAAdapter(SentimentClassifierPort):
    """Phase-2 adapter — AraBERTv2 base + PEFT LoRA classifier."""

    def __init__(
        self,
        model_dir: Path,
        base_model: str = _DEFAULT_BASE_MODEL,
    ) -> None:
        model_dir = Path(model_dir)
        adapter_config_path = model_dir / _ADAPTER_CONFIG_FILE
        labels_path = model_dir / _LABELS_FILE
        if not adapter_config_path.exists():
            raise FileNotFoundError(f"missing LoRA marker: {adapter_config_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"missing labels.json: {labels_path}")

        self._device = _pick_device()
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)
        model = PeftModel.from_pretrained(base, str(model_dir))
        model.eval()
        model.to(self._device)
        self._model = model

        raw_labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self._labels: list[Sentiment] = [Sentiment(lbl) for lbl in raw_labels]

    def predict(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=_MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        idx = int(probs.argmax())
        sentiment = self._labels[idx]
        confidence = float(probs[idx])
        return SentimentResult(text=text, sentiment=sentiment, confidence=confidence)
