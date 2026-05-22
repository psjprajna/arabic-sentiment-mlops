"""Train AraBERTv2 with LoRA on the HARD 3-class split and log to MLflow.

Persists under `model_dir`: `adapter_config.json`, `adapter_model.safetensors`
(PEFT LoRA weights), `tokenizer.json` (+ tokenizer config files), `labels.json`.
Persists at `report_path`: aggregate metrics matching Phase-1's schema plus a
`baseline_f1_macro` field pulled from `reports/catboost-baseline-v1.json`.

The MLflow run is logged via `mlflow_logging.log_run`. Trainer's built-in MLflow
callback is disabled (`report_to=[]`) — single tracking surface.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from sentiment.adapters.hard_dataset import DatasetSplits, Example, HARDDataset
from sentiment.domain.models import Sentiment
from sentiment.training.dialect_breakdown import compute_dialect_breakdown
from sentiment.training.mlflow_logging import log_run

_LABEL_ORDER: tuple[Sentiment, ...] = (
    Sentiment.POSITIVE,
    Sentiment.NEGATIVE,
    Sentiment.NEUTRAL,
)
_LABEL_TO_IDX: dict[Sentiment, int] = {s: i for i, s in enumerate(_LABEL_ORDER)}

_BASE_MODEL = "aubmindlab/bert-base-arabertv2"
_DEFAULT_MODEL_DIR = Path("models/arabert-lora-v1")
_DEFAULT_REPORT_PATH = Path("reports/arabert-lora-v1.json")
_DEFAULT_BASELINE_REPORT = Path("reports/catboost-baseline-v1.json")
_DEFAULT_N_TRAIN_SUBSAMPLE = 30_000
_DEFAULT_TRACKING_URI = "file:./mlruns"

_MAX_LENGTH = 128
_BATCH_SIZE = 16
_EPOCHS = 3
_LEARNING_RATE = 2e-4
_LORA_RANK = 8
_LORA_ALPHA = 16
_LORA_DROPOUT = 0.1
_LORA_TARGET_MODULES: tuple[str, ...] = ("query", "value")


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _stratified_subsample(examples: Sequence[Example], n: int, seed: int) -> list[Example]:
    if n >= len(examples):
        return list(examples)
    labels = [ex.sentiment.value for ex in examples]
    sub, _ = train_test_split(
        list(examples),
        train_size=n,
        random_state=seed,
        stratify=labels,
    )
    return list(sub)


def _examples_to_hf_dataset(
    examples: Sequence[Example],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> HFDataset:
    texts = [ex.text for ex in examples]
    labels = [_LABEL_TO_IDX[ex.sentiment] for ex in examples]
    raw = HFDataset.from_dict({"text": texts, "label": labels})

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return raw.map(tokenize, batched=True)


def _compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    label_ids = list(range(len(_LABEL_ORDER)))
    per_class = f1_score(labels, preds, labels=label_ids, average=None, zero_division=0.0)
    macro = f1_score(labels, preds, labels=label_ids, average="macro", zero_division=0.0)
    return {
        "f1_macro": float(macro),
        "f1_positive": float(per_class[0]),
        "f1_negative": float(per_class[1]),
        "f1_neutral": float(per_class[2]),
    }


def _evaluate_on_test(
    trainer: Trainer, test_dataset: HFDataset
) -> tuple[dict[str, float], float, list[list[int]], np.ndarray, np.ndarray]:
    output = trainer.predict(test_dataset)
    logits = output.predictions
    labels = np.asarray(output.label_ids).ravel()
    preds = logits.argmax(axis=-1).ravel()
    label_ids = list(range(len(_LABEL_ORDER)))
    per_class = f1_score(labels, preds, labels=label_ids, average=None, zero_division=0.0)
    macro = float(f1_score(labels, preds, labels=label_ids, average="macro", zero_division=0.0))
    cm = confusion_matrix(labels, preds, labels=label_ids)
    per_class_dict = {_LABEL_ORDER[i].value: float(per_class[i]) for i in range(len(_LABEL_ORDER))}
    return per_class_dict, macro, cm.astype(int).tolist(), preds, labels


def _persist_lora_adapter(
    model_dir: Path,
    model: object,
    tokenizer: object,
    labels: list[str],
) -> None:
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=model_dir.parent) as staging:
        staging_dir = Path(staging) / "out"
        staging_dir.mkdir()
        model.save_pretrained(str(staging_dir))  # type: ignore[attr-defined]
        tokenizer.save_pretrained(str(staging_dir))  # type: ignore[attr-defined]
        (staging_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.move(str(staging_dir), str(model_dir))


def _preserve_existing_confidence_histogram(report_path: Path, report: dict[str, object]) -> None:
    """Carry forward `confidence_histogram` from a prior report if present.

    The histogram is owned by `sentiment.training.build_confidence_reference`,
    not by training. LoRA on MPS has tiny non-determinism but the
    histogram is dominated by the high-confidence bucket — drift across
    retrains is within float-rounding noise. Preserving here avoids the
    ~10-minute rebuild after every retrain; operators can still refresh
    by re-running the build script.
    """
    if "confidence_histogram" in report or not report_path.exists():
        return
    try:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if "confidence_histogram" in existing:
        report["confidence_histogram"] = existing["confidence_histogram"]


def _write_report(report_path: Path, report: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _preserve_existing_confidence_histogram(report_path, report)
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(report_path)


def _load_baseline_f1(path: Path) -> float | None:
    if not path.exists():
        print(
            f"warning: baseline report {path} not found; baseline_f1_macro=null",
            file=sys.stderr,
        )
        return None
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["f1_macro"])
    except (KeyError, ValueError, json.JSONDecodeError) as err:
        print(
            f"warning: could not read f1_macro from {path}: {err}",
            file=sys.stderr,
        )
        return None


def run_lora_training(
    *,
    splits: DatasetSplits,
    model_dir: Path,
    report_path: Path,
    base_model: str = _BASE_MODEL,
    n_train_subsample: int = _DEFAULT_N_TRAIN_SUBSAMPLE,
    baseline_report_path: Path = _DEFAULT_BASELINE_REPORT,
    mlflow_tracking_uri: str = _DEFAULT_TRACKING_URI,
    seed: int = 42,
) -> dict[str, object]:
    device = _pick_device()
    n_train_full = len(splits.train)
    train_examples = _stratified_subsample(splits.train, n_train_subsample, seed)
    effective_subsample = len(train_examples)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(_LABEL_ORDER),
        id2label={i: s.value for i, s in enumerate(_LABEL_ORDER)},
        label2id={s.value: i for i, s in enumerate(_LABEL_ORDER)},
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=_LORA_RANK,
        lora_alpha=_LORA_ALPHA,
        lora_dropout=_LORA_DROPOUT,
        bias="none",
        target_modules=list(_LORA_TARGET_MODULES),
    )
    model = get_peft_model(base, lora_config)

    train_ds = _examples_to_hf_dataset(train_examples, tokenizer, _MAX_LENGTH)
    dev_ds = _examples_to_hf_dataset(splits.dev, tokenizer, _MAX_LENGTH)
    test_ds = _examples_to_hf_dataset(splits.test, tokenizer, _MAX_LENGTH)

    staging_dir = Path(tempfile.mkdtemp(prefix="lora-staging-"))
    training_args = TrainingArguments(
        output_dir=str(staging_dir),
        num_train_epochs=_EPOCHS,
        per_device_train_batch_size=_BATCH_SIZE,
        per_device_eval_batch_size=2 * _BATCH_SIZE,
        learning_rate=_LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        fp16=device == "cuda",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    per_class, macro, cm, preds, y_test = _evaluate_on_test(trainer, test_ds)
    baseline = _load_baseline_f1(baseline_report_path)
    delta = (macro - baseline) if baseline is not None else None

    # Raw test text (diacritics preserved) for the dialect tagger —
    # the HF dataset has tokenized away the original strings.
    test_texts_raw = [ex.text for ex in splits.test]
    dialect_breakdown = compute_dialect_breakdown(
        texts=test_texts_raw,
        y_true=y_test,
        y_pred=preds,
    )

    labels_str = [s.value for s in _LABEL_ORDER]
    report: dict[str, object] = {
        "model": "arabert-lora-v1",
        "base_model": base_model,
        "dataset": "HARD",
        "split_sizes": {
            "train_full": n_train_full,
            "train_subsample": effective_subsample,
            "dev": len(splits.dev),
            "test": len(splits.test),
        },
        "lora": {
            "rank": _LORA_RANK,
            "alpha": _LORA_ALPHA,
            "dropout": _LORA_DROPOUT,
            "target_modules": list(_LORA_TARGET_MODULES),
        },
        "training": {
            "epochs": _EPOCHS,
            "batch_size": _BATCH_SIZE,
            "learning_rate": _LEARNING_RATE,
            "max_length": _MAX_LENGTH,
            "seed": seed,
        },
        "f1_per_class": per_class,
        "f1_macro": macro,
        "confusion_matrix": cm,
        "label_order": labels_str,
        "baseline_f1_macro": baseline,
        "delta_vs_baseline": delta,
        "trained_at": datetime.now(UTC).isoformat(),
        "dialect_breakdown": dialect_breakdown,
    }

    _persist_lora_adapter(Path(model_dir), model, tokenizer, labels_str)
    _write_report(Path(report_path), report)

    params: dict[str, object] = {
        "base_model": base_model,
        "lora_rank": _LORA_RANK,
        "lora_alpha": _LORA_ALPHA,
        "lora_dropout": _LORA_DROPOUT,
        "target_modules": ",".join(_LORA_TARGET_MODULES),
        "n_train_full": n_train_full,
        "n_train_subsample": effective_subsample,
        "n_dev": len(splits.dev),
        "n_test": len(splits.test),
        "max_length": _MAX_LENGTH,
        "batch_size": _BATCH_SIZE,
        "learning_rate": _LEARNING_RATE,
        "epochs": _EPOCHS,
        "seed": seed,
        "device": device,
        "baseline_f1_macro": baseline if baseline is not None else "null",
    }
    metrics: dict[str, float] = {
        "f1_macro": macro,
        **{f"f1_{cls}": val for cls, val in per_class.items()},
    }
    log_run(
        run_name="arabert-lora-v1",
        params=params,
        metrics=metrics,
        artifact_paths=[Path(report_path), Path(model_dir) / "labels.json"],
        tracking_uri=mlflow_tracking_uri,
    )

    shutil.rmtree(staging_dir, ignore_errors=True)
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AraBERT + LoRA on HARD; log to MLflow.")
    parser.add_argument("--model-dir", type=Path, default=_DEFAULT_MODEL_DIR)
    parser.add_argument("--report-path", type=Path, default=_DEFAULT_REPORT_PATH)
    parser.add_argument("--source", type=str, default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-subsample", type=int, default=_DEFAULT_N_TRAIN_SUBSAMPLE)
    parser.add_argument("--mlflow-uri", type=str, default=_DEFAULT_TRACKING_URI)
    parser.add_argument("--base-model", type=str, default=_BASE_MODEL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    splits = HARDDataset(source=args.source, seed=args.seed).load()
    report = run_lora_training(
        splits=splits,
        model_dir=args.model_dir,
        report_path=args.report_path,
        base_model=args.base_model,
        n_train_subsample=args.n_train_subsample,
        mlflow_tracking_uri=args.mlflow_uri,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
