"""Smoke + orchestration tests for the AraBERT LoRA training pipeline.

All HF / PEFT / Trainer call surfaces monkey-patched. No model weights
downloaded, no real forward/backward pass. Asserts orchestration: stratified
subsampling, artifact persistence, report schema, MLflow params/metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sentiment.adapters.hard_dataset import Example, HARDDataset
from sentiment.domain.models import Sentiment
from sentiment.domain.normalizer import TextNormalizerPort


class _StubTokenizer:
    """Minimal tokenizer stub — emits BERT-style outputs of fixed length."""

    model_input_names = ("input_ids", "attention_mask", "token_type_ids")
    name_or_path = "stub-tokenizer"

    def __call__(self, texts: list[str] | str, **_kwargs: Any) -> dict[str, list[Any]]:
        if isinstance(texts, str):
            n = 1
        else:
            n = len(texts)
        seq = [0] * 8  # arbitrary short padded length
        return {
            "input_ids": [seq for _ in range(n)],
            "attention_mask": [seq for _ in range(n)],
            "token_type_ids": [seq for _ in range(n)],
        }

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")


class _StubModel:
    """Minimal model stub — fake save_pretrained writes the LoRA markers."""

    def to(self, _device: str) -> _StubModel:
        return self

    def eval(self) -> _StubModel:  # noqa: A003 — name fixed by HF convention
        return self

    def save_pretrained(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text("{}", encoding="utf-8")
        (p / "adapter_model.safetensors").write_bytes(b"stub")


class _StubTrainer:
    """Captures construction kwargs; predict returns oracle logits."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs
        self.model = kwargs["model"]

    def train(self) -> None:
        return None

    def predict(self, dataset: Any) -> SimpleNamespace:
        labels = list(dataset["label"])
        n_classes = 3
        logits = np.zeros((len(labels), n_classes), dtype=np.float32)
        for i, lbl in enumerate(labels):
            logits[i, lbl] = 10.0  # perfect prediction → F1 = 1.0
        return SimpleNamespace(predictions=logits, label_ids=np.array(labels), metrics={})


class _StubTrainingArguments:
    def __init__(self, **kwargs: Any) -> None:
        self.output_dir = kwargs.get("output_dir", "/tmp/lora-staging")
        for key, value in kwargs.items():
            setattr(self, key, value)


def _stub_get_peft_model(model: _StubModel, _config: Any) -> _StubModel:
    return model


def _stub_data_collator(_tokenizer: _StubTokenizer) -> object:
    return object()


def _stub_lora_config(**_kwargs: Any) -> object:
    return object()


@pytest.fixture
def _patched(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from sentiment.training import lora as mod

    captured_log: list[dict[str, Any]] = []
    captured_attach: list[dict[str, Any]] = []

    def _fake_log_run(**kwargs: Any) -> str | dict[str, str]:
        captured_log.append(kwargs)
        if kwargs.get("register_as") is None:
            return "stub-run-id"
        return {
            "name": str(kwargs["register_as"]),
            "version": str(len(captured_log)),
            "run_id": f"stub-run-{len(captured_log)}",
            "model_uri": f"runs:/stub-run-{len(captured_log)}/model",
            "registered_at": "2026-05-22T00:00:00+00:00",
        }

    def _fake_log_artifact_to_run(**kwargs: Any) -> None:
        captured_attach.append(kwargs)

    monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", lambda *_a, **_k: _StubTokenizer())
    monkeypatch.setattr(
        mod.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *_a, **_k: _StubModel(),
    )
    monkeypatch.setattr(mod, "get_peft_model", _stub_get_peft_model)
    monkeypatch.setattr(mod, "LoraConfig", _stub_lora_config)
    monkeypatch.setattr(mod, "Trainer", _StubTrainer)
    monkeypatch.setattr(mod, "TrainingArguments", _StubTrainingArguments)
    monkeypatch.setattr(mod, "DataCollatorWithPadding", _stub_data_collator)
    monkeypatch.setattr(mod, "log_run", _fake_log_run)
    monkeypatch.setattr(mod, "log_artifact_to_run", _fake_log_artifact_to_run)
    return {"captured_log": captured_log, "captured_attach": captured_attach}


def _synthetic_rows(n_per_class: int = 20) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for i in range(n_per_class):
        rows.append((f"رائع جدا الفندق نظيف {i}", 5))
        rows.append((f"سيء جدا قذر مزعج {i}", 1))
        rows.append((f"عادي مقبول لا بأس {i}", 3))
    return rows


def test_run_lora_training_writes_artifacts(tmp_path: Path, _patched: dict[str, Any]) -> None:
    from sentiment.training.lora import run_lora_training

    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    model_dir = tmp_path / "lora"
    report_path = tmp_path / "lora-report.json"

    run_lora_training(
        splits=splits,
        model_dir=model_dir,
        report_path=report_path,
        n_train_subsample=30,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )

    assert (model_dir / "adapter_config.json").exists()
    assert (model_dir / "adapter_model.safetensors").exists()
    assert (model_dir / "tokenizer.json").exists()
    assert (model_dir / "labels.json").exists()
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    assert labels == ["positive", "negative", "neutral"]
    assert report_path.exists()


def test_run_lora_training_report_schema(tmp_path: Path, _patched: dict[str, Any]) -> None:
    from sentiment.training.lora import run_lora_training

    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    model_dir = tmp_path / "lora"
    report_path = tmp_path / "lora-report.json"

    report = run_lora_training(
        splits=splits,
        model_dir=model_dir,
        report_path=report_path,
        n_train_subsample=30,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )

    parsed = json.loads(report_path.read_text(encoding="utf-8"))
    assert parsed == report
    for key in (
        "model",
        "base_model",
        "dataset",
        "split_sizes",
        "lora",
        "training",
        "f1_per_class",
        "f1_macro",
        "confusion_matrix",
        "label_order",
        "baseline_f1_macro",
        "delta_vs_baseline",
        "trained_at",
    ):
        assert key in parsed, f"missing key: {key}"
    assert 0.0 <= parsed["f1_macro"] <= 1.0
    cm = parsed["confusion_matrix"]
    assert len(cm) == 3 and all(len(row) == 3 for row in cm)
    assert parsed["label_order"] == ["positive", "negative", "neutral"]
    for cls in ("positive", "negative", "neutral"):
        assert cls in parsed["f1_per_class"]
        assert parsed["f1_per_class"][cls] is not None


def test_run_lora_training_calls_mlflow(tmp_path: Path, _patched: dict[str, Any]) -> None:
    from sentiment.training.lora import run_lora_training

    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    run_lora_training(
        splits=splits,
        model_dir=tmp_path / "lora",
        report_path=tmp_path / "lora-report.json",
        n_train_subsample=30,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )

    captured = _patched["captured_log"]
    assert len(captured) == 1
    call = captured[0]
    assert call["run_name"] == "arabert-lora-v1"

    expected_param_keys = {
        "base_model",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "n_train_full",
        "n_train_subsample",
        "n_dev",
        "n_test",
        "max_length",
        "batch_size",
        "learning_rate",
        "epochs",
        "seed",
        "device",
    }
    assert expected_param_keys.issubset(call["params"].keys())

    expected_metric_keys = {"f1_macro", "f1_positive", "f1_negative", "f1_neutral"}
    assert expected_metric_keys.issubset(call["metrics"].keys())

    artifact_names = {Path(p).name for p in call["artifact_paths"]}
    assert "labels.json" in artifact_names

    # Phase 6 — registry kwargs threaded through.
    assert call["register_as"] == "arabert-lora"
    assert call["python_model"] is not None
    assert set(call["artifacts"].keys()) == {"backend_type", "model_dir"}


def test_run_lora_training_report_includes_dialect_breakdown(
    tmp_path: Path, _patched: dict[str, Any]
) -> None:
    from sentiment.training.lora import run_lora_training

    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    report = run_lora_training(
        splits=splits,
        model_dir=tmp_path / "lora",
        report_path=tmp_path / "lora-report.json",
        n_train_subsample=30,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )

    assert "dialect_breakdown" in report
    breakdown = report["dialect_breakdown"]
    assert isinstance(breakdown, dict)
    assert breakdown["tagger"] == "lexicon-v1"
    assert "gulf" in breakdown and "msa" in breakdown
    gulf_share = breakdown["gulf_share"]
    assert isinstance(gulf_share, float)
    assert 0.0 <= gulf_share <= 1.0
    # Synthetic LoRA-fixture rows carry no Gulf markers — gulf bucket empty.
    assert breakdown["gulf"] == {"insufficient_examples": True, "n": 0}


def test_run_lora_training_subsamples_train_split(tmp_path: Path, _patched: dict[str, Any]) -> None:
    from sentiment.training.lora import run_lora_training

    # Build a 270-row split deterministically so train ≈ 216 after the 80/10/10
    # split inside HARDDataset.from_rows. We want subsample (50) to be much
    # smaller than train so the test exercises the subsampling path.
    rows: list[tuple[str, int]] = []
    for i in range(90):
        rows.append((f"رائع {i}", 5))
        rows.append((f"سيء {i}", 1))
        rows.append((f"عادي {i}", 3))
    splits = HARDDataset.from_rows(rows, seed=42)
    assert len(splits.train) >= 50

    run_lora_training(
        splits=splits,
        model_dir=tmp_path / "lora",
        report_path=tmp_path / "lora-report.json",
        n_train_subsample=50,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )

    captured = _patched["captured_log"][0]
    assert captured["params"]["n_train_subsample"] == 50
    assert captured["params"]["n_train_full"] == len(splits.train)

    # Trainer captured the subsampled training dataset
    train_ds = _StubTrainer.last_kwargs["train_dataset"]
    assert len(train_ds) == 50

    # Stratified: each class is within ±1 of its proportional share
    full_counts: dict[Sentiment, int] = {s: 0 for s in Sentiment}
    for ex in splits.train:
        full_counts[ex.sentiment] += 1
    sub_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for label in train_ds["label"]:
        sub_counts[int(label)] += 1
    label_to_idx = {Sentiment.POSITIVE: 0, Sentiment.NEGATIVE: 1, Sentiment.NEUTRAL: 2}
    total = len(splits.train)
    for sentiment, idx in label_to_idx.items():
        expected = round(full_counts[sentiment] * 50 / total)
        assert abs(sub_counts[idx] - expected) <= 1, (
            f"class {sentiment} off by >1 (sub={sub_counts[idx]}, expected≈{expected})"
        )


# ---------------------------------------------------------------------------
# Phase 6 — MLflow Model Registry surfacing on every LoRA training run
# ---------------------------------------------------------------------------


def test_lora_report_includes_registry_block(tmp_path: Path, _patched: dict[str, Any]) -> None:
    from sentiment.training.lora import run_lora_training

    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    report_path = tmp_path / "lora-report.json"
    # Pre-populate the report with a Phase-4-owned confidence_histogram so
    # _preserve_existing_owned_keys carries it forward across retrains.
    report_path.write_text(
        json.dumps({"confidence_histogram": {"low": 1, "med": 2, "high": 3}}),
        encoding="utf-8",
    )

    report = run_lora_training(
        splits=splits,
        model_dir=tmp_path / "lora",
        report_path=report_path,
        n_train_subsample=30,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )

    parsed = json.loads(report_path.read_text(encoding="utf-8"))
    assert parsed == report
    assert "registry" in parsed
    assert set(parsed["registry"].keys()) == {
        "name",
        "version",
        "run_id",
        "model_uri",
        "registered_at",
    }
    assert parsed["registry"]["name"] == "arabert-lora"
    assert "dialect_breakdown" in parsed
    # confidence_histogram carried forward by _preserve_existing_owned_keys.
    assert parsed["confidence_histogram"] == {"low": 1, "med": 2, "high": 3}


def test_lora_registry_version_increments_on_second_call(
    tmp_path: Path, _patched: dict[str, Any]
) -> None:
    from sentiment.training.lora import run_lora_training

    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    common = dict(
        splits=splits,
        model_dir=tmp_path / "lora",
        report_path=tmp_path / "lora-report.json",
        n_train_subsample=30,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )

    first = run_lora_training(**common)
    second = run_lora_training(**common)

    assert first["registry"]["version"] == "1"
    assert second["registry"]["version"] == "2"


def test_lora_report_artifact_attached_to_mlflow_run(
    tmp_path: Path, _patched: dict[str, Any]
) -> None:
    from sentiment.training.lora import run_lora_training

    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    report_path = tmp_path / "lora-report.json"

    run_lora_training(
        splits=splits,
        model_dir=tmp_path / "lora",
        report_path=report_path,
        n_train_subsample=30,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )

    attached = _patched["captured_attach"]
    assert len(attached) == 1
    assert Path(attached[0]["path"]) == report_path
    # run_id comes from the fake log_run's stub return — proves the
    # attach call ran after log_run resolved and used its run_id.
    assert attached[0]["run_id"] == "stub-run-1"


# ---------------------------------------------------------------------------
# Phase 11b — Gulf->MSA train-side augmentation on the LoRA backend
# ---------------------------------------------------------------------------


class _SentinelNormalizer(TextNormalizerPort):
    """Returns a recognizable sentinel — proves augmented rows reach the trainer."""

    SENTINEL = "zzzsentinelzzz"

    def normalize(self, text: str) -> str:
        return self.SENTINEL


def _synthetic_rows_with_gulf(n_per_class: int = 20) -> list[tuple[str, int]]:
    """Same shape as _synthetic_rows but positive class carries Gulf markers."""
    rows: list[tuple[str, int]] = []
    for i in range(n_per_class):
        rows.append((f"شلون الفندق وايد زين رائع ممتاز جدا {i}", 5))
        rows.append((f"سيء جدا قذر مزعج فظيع غير محترم {i}", 1))
        rows.append((f"عادي مقبول لا بأس متوسط لا أكثر {i}", 3))
    return rows


def test_augment_train_with_normalizer_appends_only_gulf_rows() -> None:
    from sentiment.training.lora import _augment_train_with_normalizer

    examples = [
        Example(text="شلون الفندق", sentiment=Sentiment.POSITIVE),
        Example(text="الفندق نظيف", sentiment=Sentiment.POSITIVE),
        Example(text="وايد زين", sentiment=Sentiment.POSITIVE),
    ]
    out = _augment_train_with_normalizer(examples, _SentinelNormalizer())
    # Original 3 + 2 augmented copies (only the Gulf-tagged rows).
    assert len(out) == 5
    assert out[:3] == examples
    assert all(ex.text == _SentinelNormalizer.SENTINEL for ex in out[3:])
    assert all(ex.sentiment is Sentiment.POSITIVE for ex in out[3:])


def test_run_lora_augmentation_reaches_trainer_input(
    tmp_path: Path, _patched: dict[str, Any]
) -> None:
    from sentiment.training.lora import run_lora_training

    splits = HARDDataset.from_rows(_synthetic_rows_with_gulf(20), seed=42)
    n_train_pre_aug = min(len(splits.train), 30)

    run_lora_training(
        splits=splits,
        model_dir=tmp_path / "lora",
        report_path=tmp_path / "lora-report.json",
        n_train_subsample=30,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        normalizer=_SentinelNormalizer(),
    )

    train_ds = _StubTrainer.last_kwargs["train_dataset"]
    # Augmented rows must show up in the dataset handed to Trainer.
    assert _SentinelNormalizer.SENTINEL in train_ds["text"]
    # Augmentation grew the trainer's train split beyond the pre-aug subsample.
    assert len(train_ds) > n_train_pre_aug


def test_run_lora_logs_transliterate_gulf_param_when_normalizer_passed(
    tmp_path: Path, _patched: dict[str, Any]
) -> None:
    from sentiment.training.lora import run_lora_training

    splits = HARDDataset.from_rows(_synthetic_rows_with_gulf(20), seed=42)

    run_lora_training(
        splits=splits,
        model_dir=tmp_path / "lora",
        report_path=tmp_path / "lora-report.json",
        n_train_subsample=30,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        normalizer=_SentinelNormalizer(),
    )

    call = _patched["captured_log"][0]
    params = call["params"]
    assert params["transliterate_gulf"] is True
    assert "n_train_original" in params
    assert params["n_train_subsample"] > params["n_train_original"]


def test_run_lora_default_path_omits_transliterate_gulf_param(
    tmp_path: Path, _patched: dict[str, Any]
) -> None:
    from sentiment.training.lora import run_lora_training

    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)

    run_lora_training(
        splits=splits,
        model_dir=tmp_path / "lora",
        report_path=tmp_path / "lora-report.json",
        n_train_subsample=30,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )

    params = _patched["captured_log"][0]["params"]
    assert "transliterate_gulf" not in params
    assert "n_train_original" not in params
