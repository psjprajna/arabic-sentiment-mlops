"""Tests for AraBERTLoRAAdapter (mocked — no real weights, no HF download)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from sentiment.adapters import arabert_lora_classifier as mod
from sentiment.adapters.arabert_lora_classifier import AraBERTLoRAAdapter
from sentiment.domain.models import Sentiment


def _make_model_dir(tmp_path: Path, *, labels: list[str] | None = None) -> Path:
    labels = labels or ["positive", "negative", "neutral"]
    model_dir = tmp_path / "lora"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "adapter_model.safetensors").write_bytes(b"stub")
    (model_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    return model_dir


def _build_adapter(
    tmp_path: Path,
    *,
    logits: list[float],
    labels: list[str] | None = None,
    monkeypatch: pytest.MonkeyPatch,
) -> AraBERTLoRAAdapter:
    model_dir = _make_model_dir(tmp_path, labels=labels)
    stub_tokenizer = MagicMock()
    stub_tokenizer.return_value = {
        "input_ids": torch.zeros((1, 128), dtype=torch.long),
        "attention_mask": torch.zeros((1, 128), dtype=torch.long),
        "token_type_ids": torch.zeros((1, 128), dtype=torch.long),
    }
    monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", lambda *_a, **_k: stub_tokenizer)

    stub_base = MagicMock()
    monkeypatch.setattr(
        mod.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *_a, **_k: stub_base,
    )

    stub_model = MagicMock()
    stub_model.to.return_value = stub_model
    stub_output = MagicMock()
    stub_output.logits = torch.tensor([logits])
    stub_model.return_value = stub_output
    monkeypatch.setattr(mod.PeftModel, "from_pretrained", lambda *_a, **_k: stub_model)

    return AraBERTLoRAAdapter(model_dir)


def test_predict_raises_on_empty_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter(tmp_path, logits=[1.0, 0.0, 0.0], monkeypatch=monkeypatch)
    with pytest.raises(ValueError):
        adapter.predict("")


def test_predict_raises_on_whitespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter(tmp_path, logits=[1.0, 0.0, 0.0], monkeypatch=monkeypatch)
    with pytest.raises(ValueError):
        adapter.predict("   ")


def test_predict_returns_original_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _build_adapter(tmp_path, logits=[3.0, 0.1, 0.5], monkeypatch=monkeypatch)
    diacritized = "رَائِع"  # tashkeel preserved end-to-end
    result = adapter.predict(diacritized)
    assert result.text == diacritized


def test_predict_maps_label_index_correctly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logits = [2.0, 0.1, 0.5]
    adapter = _build_adapter(
        tmp_path,
        logits=logits,
        labels=["positive", "negative", "neutral"],
        monkeypatch=monkeypatch,
    )
    result = adapter.predict("نص ما")
    assert result.sentiment is Sentiment.POSITIVE

    exps = [math.exp(x) for x in logits]
    expected_conf = exps[0] / sum(exps)
    assert result.confidence == pytest.approx(expected_conf, abs=1e-6)


def test_init_raises_when_adapter_config_missing(tmp_path: Path) -> None:
    model_dir = tmp_path / "lora"
    model_dir.mkdir()
    (model_dir / "labels.json").write_text("[]", encoding="utf-8")
    with pytest.raises(FileNotFoundError) as excinfo:
        AraBERTLoRAAdapter(model_dir)
    assert "adapter_config.json" in str(excinfo.value)


def test_init_raises_when_labels_missing(tmp_path: Path) -> None:
    model_dir = tmp_path / "lora"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError) as excinfo:
        AraBERTLoRAAdapter(model_dir)
    assert "labels.json" in str(excinfo.value)
