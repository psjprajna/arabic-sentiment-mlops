"""Smoke test for the CatBoost baseline training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from sentiment.adapters.hard_dataset import HARDDataset
from sentiment.training.baseline import run_baseline


def _synthetic_rows(n_per_class: int = 20) -> list[tuple[str, int]]:
    """Build a separable synthetic Arabic-like corpus."""
    rows: list[tuple[str, int]] = []
    for i in range(n_per_class):
        rows.append((f"رائع ممتاز جدا الفندق نظيف الخدمة سريعة {i}", 5))
        rows.append((f"سيء جدا قذر مزعج فظيع غير محترم {i}", 1))
        rows.append((f"عادي مقبول لا بأس متوسط لا أكثر {i}", 3))
    return rows


def test_run_baseline_writes_artifacts_and_clears_bar(tmp_path: Path) -> None:
    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    model_dir = tmp_path / "m"
    report_path = tmp_path / "r.json"

    report = run_baseline(splits=splits, model_dir=model_dir, report_path=report_path)

    assert (model_dir / "model.cbm").exists()
    assert (model_dir / "vectorizer.joblib").exists()
    assert (model_dir / "labels.json").exists()
    assert report_path.exists()

    parsed = json.loads(report_path.read_text(encoding="utf-8"))
    assert parsed == report
    assert parsed["f1_macro"] > 0.5
    for cls in ("positive", "negative", "neutral"):
        assert cls in parsed["f1_per_class"]
        assert parsed["f1_per_class"][cls] is not None
    cm = parsed["confusion_matrix"]
    assert len(cm) == 3 and all(len(row) == 3 for row in cm)
    assert all(isinstance(v, int) for row in cm for v in row)


def test_run_baseline_report_includes_dialect_breakdown(tmp_path: Path) -> None:
    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    model_dir = tmp_path / "m"
    report_path = tmp_path / "r.json"

    report = run_baseline(splits=splits, model_dir=model_dir, report_path=report_path)

    assert "dialect_breakdown" in report
    breakdown = report["dialect_breakdown"]
    assert isinstance(breakdown, dict)
    assert breakdown["tagger"] == "lexicon-v1"
    assert "gulf" in breakdown and "msa" in breakdown
    gulf_share = breakdown["gulf_share"]
    assert isinstance(gulf_share, float)
    assert 0.0 <= gulf_share <= 1.0
    # Synthetic rows carry no Gulf markers — gulf bucket is empty.
    assert breakdown["gulf"] == {"insufficient_examples": True, "n": 0}
    msa_block = breakdown["msa"]
    assert isinstance(msa_block, dict)
    # If MSA bucket has enough rows, it carries full metrics.
    if "f1_macro" in msa_block:
        assert "f1_per_class" in msa_block
        assert "confusion_matrix" in msa_block
        assert msa_block["n"] >= 20
