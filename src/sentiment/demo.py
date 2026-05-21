"""CLI showcase for the trained CatBoost adapter.

Loads `CatBoostAdapter` from a model dir, prints either a markdown table over the
curated cases or a single row for `--text`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from sentiment.adapters.arabert_lora_classifier import AraBERTLoRAAdapter
from sentiment.adapters.catboost_classifier import CatBoostAdapter
from sentiment.domain.classifier import SentimentClassifierPort
from sentiment.domain.models import SentimentResult

CURATED_CASES: list[str] = [
    "الفندق ممتاز والخدمة رائعة جدا",
    "الغرفة كانت قذرة والموظفون غير محترمين",
    "الفندق عادي، لا أكثر ولا أقل",
    "الفندق وايد زين وأهل البحرين يحبونه",
    "ما عجبني المكان أبداً",
    "الخدمة ليست سيئة",
    "🌟🌟🌟🌟🌟",
    "Great hotel, excellent service",
]

_DEFAULT_MODEL_DIRS: dict[str, Path] = {
    "catboost": Path("models/catboost-baseline-v1"),
    "lora": Path("models/arabert-lora-v1"),
}


def _format_row(idx: int, text: str, result: SentimentResult) -> str:
    return f"| {idx} | {text} | {result.sentiment.value} | {result.confidence:.2f} |"


def _render_table(rows: Iterable[tuple[int, str, SentimentResult]]) -> str:
    lines = [
        "| # | input | predicted | confidence |",
        "|---|-------|-----------|------------|",
    ]
    for idx, text, result in rows:
        lines.append(_format_row(idx, text, result))
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Showcase a trained sentiment adapter.")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--backend", type=str, choices=("catboost", "lora"), default="catboost")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.model_dir is None:
        args.model_dir = _DEFAULT_MODEL_DIRS[args.backend]
    return args


def _build_adapter(backend: str, model_dir: Path) -> SentimentClassifierPort:
    if backend == "lora":
        return AraBERTLoRAAdapter(model_dir)
    return CatBoostAdapter(model_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    adapter = _build_adapter(args.backend, args.model_dir)
    if args.text is not None:
        result = adapter.predict(args.text)
        print(_render_table([(1, args.text, result)]))
        return 0
    rows: list[tuple[int, str, SentimentResult]] = []
    for i, text in enumerate(CURATED_CASES, start=1):
        rows.append((i, text, adapter.predict(text)))
    print(_render_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
