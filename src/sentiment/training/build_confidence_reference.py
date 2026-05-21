"""Backfill `confidence_histogram` into reports/<backend>.json.

Runs the chosen backend over the HARD test split and persists the 3-band
confidence distribution as a new top-level field in the existing report JSON.
Re-uses `domain.drift.confidence_bucket` so the bucket boundaries cannot drift
out of sync with the live `/metrics/drift` endpoint.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from sentiment.adapters.arabert_lora_classifier import AraBERTLoRAAdapter
from sentiment.adapters.catboost_classifier import CatBoostAdapter
from sentiment.adapters.hard_dataset import HARDDataset
from sentiment.domain.classifier import SentimentClassifierPort
from sentiment.domain.drift import confidence_bucket

_BUCKET_KEYS: tuple[str, ...] = ("low", "medium", "high")
_DEFAULT_REPORTS_DIR = Path("reports")
_BACKEND_TO_REPORT_NAME: dict[str, str] = {
    "catboost": "catboost-baseline-v1",
    "lora": "arabert-lora-v1",
}
_BACKEND_TO_MODEL_DIR: dict[str, Path] = {
    "catboost": Path("models/catboost-baseline-v1"),
    "lora": Path("models/arabert-lora-v1"),
}


def _build_adapter(backend: str) -> SentimentClassifierPort:
    if backend == "catboost":
        return CatBoostAdapter(model_dir=_BACKEND_TO_MODEL_DIR[backend])
    return AraBERTLoRAAdapter(model_dir=_BACKEND_TO_MODEL_DIR[backend])


def _compute_confidence_histogram(confidences: Iterable[float]) -> dict[str, float]:
    counts: dict[str, int] = {key: 0 for key in _BUCKET_KEYS}
    total = 0
    for value in confidences:
        counts[confidence_bucket(value)] += 1
        total += 1
    if total == 0:
        raise ValueError("no confidence values provided — cannot build histogram")
    return {key: count / total for key, count in counts.items()}


def _iter_confidences(
    adapter: SentimentClassifierPort,
    texts: Sequence[str],
    limit: int,
) -> Iterable[float]:
    capped = texts if limit <= 0 else texts[:limit]
    for text in capped:
        yield adapter.predict(text).confidence


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill confidence_histogram into the backend's report JSON.",
    )
    parser.add_argument("--backend", required=True, choices=sorted(_BACKEND_TO_REPORT_NAME))
    parser.add_argument("--reports-dir", type=Path, default=_DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap predictions for a smoke run; 0 means full test split.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    adapter = _build_adapter(args.backend)
    splits = HARDDataset().load()
    texts = [ex.text for ex in splits.test]
    histogram = _compute_confidence_histogram(_iter_confidences(adapter, texts, args.limit))

    report_path = Path(args.reports_dir) / f"{_BACKEND_TO_REPORT_NAME[args.backend]}.json"
    with report_path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    report["confidence_histogram"] = histogram
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(json.dumps(histogram, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
