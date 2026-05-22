"""Train a CatBoost + TF-IDF baseline on a `DatasetSplits` and persist artifacts.

Produces under `model_dir`: `model.cbm`, `vectorizer.joblib`, `labels.json`,
`pyfunc_meta.json`. Produces at `report_path`: aggregate metrics (per-class
F1 + 3x3 confusion matrix + `dialect_breakdown` + `registry` block).

Phase 6 (ADR-0003): every run logs to MLflow and registers a fresh version
under name `catboost-baseline` via `SentimentPyfunc`. The `registry` block
in the report links the JSON artifact back to its registered model version.
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
from sentiment.training.mlflow_logging import (
    SentimentPyfunc,
    log_artifact_to_run,
    log_run,
)

_LABEL_ORDER: tuple[Sentiment, ...] = (Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL)
_DEFAULT_MODEL_DIR = Path("models/catboost-baseline-v1")
_DEFAULT_REPORT_PATH = Path("reports/catboost-baseline-v1.json")
_DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
_REGISTERED_MODEL_NAME = "catboost-baseline"
_MAX_FEATURES = 20_000
_ITERATIONS = 200
_DEPTH = 6
_LEARNING_RATE = 0.1
_RANDOM_SEED = 42
_KEYS_TO_PRESERVE: tuple[str, ...] = ("confidence_histogram", "dialect_breakdown")


def _texts_and_labels(examples: Sequence[Example]) -> tuple[list[str], np.ndarray]:
    texts = [_normalize(ex.text) for ex in examples]
    label_to_idx = {s: i for i, s in enumerate(_LABEL_ORDER)}
    y = np.array([label_to_idx[ex.sentiment] for ex in examples], dtype=np.int64)
    return texts, y


def _fit_vectorizer(texts: Sequence[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=_MAX_FEATURES,
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
        iterations=_ITERATIONS,
        depth=_DEPTH,
        learning_rate=_LEARNING_RATE,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=20,
        random_seed=_RANDOM_SEED,
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


def _preserve_existing_owned_keys(report_path: Path, report: dict[str, object]) -> None:
    """Carry forward fields owned by separate scripts from a prior report.

    Phase-4 owns ``confidence_histogram`` (built by
    ``sentiment.training.build_confidence_reference``); Phase-5's
    ``dialect_breakdown`` is owned by training itself. Both are tracked so a
    future training script that stops producing one of them does not silently
    wipe the prior value. The operator can still refresh ``confidence_histogram``
    by re-running the build script.
    """
    if not report_path.exists():
        return
    try:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for key in _KEYS_TO_PRESERVE:
        if key not in report and key in existing:
            report[key] = existing[key]


def _write_report(report_path: Path, report: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _preserve_existing_owned_keys(report_path, report)
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(report_path)


def run_baseline(
    *,
    splits: DatasetSplits,
    model_dir: Path,
    report_path: Path,
    mlflow_tracking_uri: str = _DEFAULT_TRACKING_URI,
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

    meta_path = Path(model_dir) / "pyfunc_meta.json"
    meta_path.write_text(json.dumps({"backend_type": "catboost"}), encoding="utf-8")

    params: dict[str, object] = {
        "backend": "catboost",
        "max_features": _MAX_FEATURES,
        "iterations": _ITERATIONS,
        "depth": _DEPTH,
        "learning_rate": _LEARNING_RATE,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "auto_class_weights": "Balanced",
        "random_seed": _RANDOM_SEED,
        "n_train": len(splits.train),
        "n_dev": len(splits.dev),
        "n_test": len(splits.test),
    }
    metrics: dict[str, float] = {
        "f1_macro": macro,
        **{f"f1_{cls}": val for cls, val in per_class.items()},
    }
    registry_info = log_run(
        run_name="catboost-baseline-v1",
        params=params,
        metrics=metrics,
        artifact_paths=[],
        tracking_uri=mlflow_tracking_uri,
        register_as=_REGISTERED_MODEL_NAME,
        python_model=SentimentPyfunc(),
        artifacts={
            "backend_type": str(meta_path),
            "model_dir": str(model_dir),
        },
    )
    assert isinstance(registry_info, dict)  # narrowed by register_as not None
    report["registry"] = registry_info

    _write_report(Path(report_path), report)
    log_artifact_to_run(
        run_id=registry_info["run_id"],
        tracking_uri=mlflow_tracking_uri,
        path=Path(report_path),
    )
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost baseline on HARD.")
    parser.add_argument("--model-dir", type=Path, default=_DEFAULT_MODEL_DIR)
    parser.add_argument("--report-path", type=Path, default=_DEFAULT_REPORT_PATH)
    parser.add_argument("--source", type=str, default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow-uri", type=str, default=_DEFAULT_TRACKING_URI)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    splits = HARDDataset(source=args.source, seed=args.seed).load()
    report = run_baseline(
        splits=splits,
        model_dir=args.model_dir,
        report_path=args.report_path,
        mlflow_tracking_uri=args.mlflow_uri,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
