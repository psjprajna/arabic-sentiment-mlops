"""Train a CatBoost + TF-IDF baseline on a `DatasetSplits` and persist artifacts.

Produces under `model_dir`: `model.cbm`, `vectorizer.joblib`, `labels.json`.
Produces at `report_path`: aggregate metrics (per-class F1 + 3x3 confusion matrix).

No MLflow in Phase 1 (D2) — the JSON report is the artifact.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score

from sentiment.adapters.catboost_classifier import _normalize
from sentiment.adapters.hard_dataset import DatasetSplits, Example, HARDDataset
from sentiment.domain.models import Sentiment
from sentiment.training.dialect_breakdown import compute_dialect_breakdown

_LABEL_ORDER: tuple[Sentiment, ...] = (Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL)
_DEFAULT_MODEL_DIR = Path("models/catboost-baseline-v1")
_DEFAULT_REPORT_PATH = Path("reports/catboost-baseline-v1.json")


def _texts_and_labels(examples: Sequence[Example]) -> tuple[list[str], np.ndarray]:
    texts = [_normalize(ex.text) for ex in examples]
    label_to_idx = {s: i for i, s in enumerate(_LABEL_ORDER)}
    y = np.array([label_to_idx[ex.sentiment] for ex in examples], dtype=np.int64)
    return texts, y


def _fit_vectorizer(texts: Sequence[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20_000,
        min_df=3,
        sublinear_tf=True,
    )
    vec.fit(texts)
    return vec


def _fit_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
) -> CatBoostClassifier:
    clf = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=20,
        random_seed=42,
        verbose=50,
        task_type="CPU",
        allow_writing_files=False,
    )
    clf.fit(x_train, y_train, eval_set=(x_dev, y_dev))
    return clf


def _evaluate(
    clf: CatBoostClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], float, list[list[int]], np.ndarray]:
    preds = clf.predict(x_test).astype(int).ravel()
    label_ids = list(range(len(_LABEL_ORDER)))
    per_class = f1_score(y_test, preds, labels=label_ids, average=None, zero_division=0.0)
    macro = float(f1_score(y_test, preds, labels=label_ids, average="macro", zero_division=0.0))
    cm = confusion_matrix(y_test, preds, labels=list(range(len(_LABEL_ORDER))))
    per_class_dict = {_LABEL_ORDER[i].value: float(per_class[i]) for i in range(len(_LABEL_ORDER))}
    return per_class_dict, macro, cm.astype(int).tolist(), preds


def _persist_artifacts(
    model_dir: Path,
    clf: CatBoostClassifier,
    vec: TfidfVectorizer,
) -> None:
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=model_dir.parent) as staging:
        staging_dir = Path(staging) / "out"
        staging_dir.mkdir()
        clf.save_model(str(staging_dir / "model.cbm"))
        joblib.dump(vec, staging_dir / "vectorizer.joblib")
        labels = [s.value for s in _LABEL_ORDER]
        (staging_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.move(str(staging_dir), str(model_dir))


def _preserve_existing_confidence_histogram(report_path: Path, report: dict[str, object]) -> None:
    """Carry forward `confidence_histogram` from a prior report if present.

    The histogram is owned by `sentiment.training.build_confidence_reference`,
    not by training. With CatBoost's deterministic seed the histogram is
    stable across retrains; preserving it here saves an explicit rebuild
    step. The operator can still refresh by re-running the build script.
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


def run_baseline(
    *,
    splits: DatasetSplits,
    model_dir: Path,
    report_path: Path,
) -> dict[str, object]:
    train_texts, y_train = _texts_and_labels(splits.train)
    dev_texts, y_dev = _texts_and_labels(splits.dev)
    test_texts, y_test = _texts_and_labels(splits.test)

    vec = _fit_vectorizer(train_texts)
    x_train = vec.transform(train_texts)
    x_dev = vec.transform(dev_texts)
    x_test = vec.transform(test_texts)

    clf = _fit_classifier(x_train, y_train, x_dev, y_dev)
    per_class, macro, cm, preds = _evaluate(clf, x_test, y_test)

    # Use raw Example.text (diacritics preserved) for dialect tagging,
    # NOT the _normalize-d test_texts that the model consumed.
    test_texts_raw = [ex.text for ex in splits.test]
    dialect_breakdown = compute_dialect_breakdown(
        texts=test_texts_raw,
        y_true=y_test,
        y_pred=preds,
    )

    report: dict[str, object] = {
        "model": "catboost-baseline-v1",
        "dataset": "HARD",
        "split_sizes": {
            "train": len(splits.train),
            "dev": len(splits.dev),
            "test": len(splits.test),
        },
        "f1_per_class": per_class,
        "f1_macro": macro,
        "confusion_matrix": cm,
        "label_order": [s.value for s in _LABEL_ORDER],
        "trained_at": datetime.now(UTC).isoformat(),
        "dialect_breakdown": dialect_breakdown,
    }

    _persist_artifacts(Path(model_dir), clf, vec)
    _write_report(Path(report_path), report)
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost baseline on HARD.")
    parser.add_argument("--model-dir", type=Path, default=_DEFAULT_MODEL_DIR)
    parser.add_argument("--report-path", type=Path, default=_DEFAULT_REPORT_PATH)
    parser.add_argument("--source", type=str, default="default")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    splits = HARDDataset(source=args.source, seed=args.seed).load()
    report = run_baseline(
        splits=splits,
        model_dir=args.model_dir,
        report_path=args.report_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
