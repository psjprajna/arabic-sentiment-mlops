"""Smoke test for the CatBoost baseline training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from mlflow.tracking import MlflowClient

from sentiment.adapters.hard_dataset import Example, HARDDataset
from sentiment.domain.models import Sentiment
from sentiment.domain.normalizer import TextNormalizerPort
from sentiment.training.baseline import _augment_train_with_normalizer, run_baseline


def _synthetic_rows(n_per_class: int = 20) -> list[tuple[str, int]]:
    """Build a separable synthetic Arabic-like corpus."""
    rows: list[tuple[str, int]] = []
    for i in range(n_per_class):
        rows.append((f"رائع ممتاز جدا الفندق نظيف الخدمة سريعة {i}", 5))
        rows.append((f"سيء جدا قذر مزعج فظيع غير محترم {i}", 1))
        rows.append((f"عادي مقبول لا بأس متوسط لا أكثر {i}", 3))
    return rows


def _tracking_uri(tmp_path: Path) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{tmp_path}/mlflow.db"


def test_run_baseline_writes_artifacts_and_clears_bar(tmp_path: Path) -> None:
    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    model_dir = tmp_path / "m"
    report_path = tmp_path / "r.json"

    report = run_baseline(
        splits=splits,
        model_dir=model_dir,
        report_path=report_path,
        mlflow_tracking_uri=_tracking_uri(tmp_path),
    )

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

    report = run_baseline(
        splits=splits,
        model_dir=model_dir,
        report_path=report_path,
        mlflow_tracking_uri=_tracking_uri(tmp_path),
    )

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


# ---------------------------------------------------------------------------
# Phase 6 — MLflow run + Model Registry surfacing on every baseline run
# ---------------------------------------------------------------------------


def test_baseline_logs_to_mlflow_when_tracking_uri_passed(tmp_path: Path) -> None:
    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    uri = _tracking_uri(tmp_path)

    run_baseline(
        splits=splits,
        model_dir=tmp_path / "m",
        report_path=tmp_path / "r.json",
        mlflow_tracking_uri=uri,
    )

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("arabic-sentiment")
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.tags.get("mlflow.runName") == "catboost-baseline-v1"
    assert "n_train" in run.data.params
    assert "f1_macro" in run.data.metrics


def test_baseline_report_includes_registry_block(tmp_path: Path) -> None:
    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    report_path = tmp_path / "r.json"
    # Pre-populate the report with a Phase-4-owned confidence_histogram so
    # the _preserve_existing_owned_keys helper has something to carry forward.
    report_path.write_text(
        json.dumps({"confidence_histogram": {"low": 1, "med": 2, "high": 3}}),
        encoding="utf-8",
    )

    run_baseline(
        splits=splits,
        model_dir=tmp_path / "m",
        report_path=report_path,
        mlflow_tracking_uri=_tracking_uri(tmp_path),
    )

    parsed = json.loads(report_path.read_text(encoding="utf-8"))
    assert "registry" in parsed
    assert set(parsed["registry"].keys()) == {
        "name",
        "version",
        "run_id",
        "model_uri",
        "registered_at",
    }
    assert parsed["registry"]["name"] == "catboost-baseline"
    assert parsed["registry"]["version"] == "1"
    assert "dialect_breakdown" in parsed
    # confidence_histogram was carried forward by _preserve_existing_owned_keys.
    assert parsed["confidence_histogram"] == {"low": 1, "med": 2, "high": 3}


def test_baseline_registry_version_increments_on_second_call(tmp_path: Path) -> None:
    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    uri = _tracking_uri(tmp_path)
    common = dict(
        splits=splits,
        model_dir=tmp_path / "m",
        report_path=tmp_path / "r.json",
        mlflow_tracking_uri=uri,
    )

    run_baseline(**common)
    first = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    run_baseline(**common)
    second = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))

    assert first["registry"]["version"] == "1"
    assert second["registry"]["version"] == "2"


def test_baseline_report_artifact_attached_to_mlflow_run(tmp_path: Path) -> None:
    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    uri = _tracking_uri(tmp_path)
    report_path = tmp_path / "r.json"

    run_baseline(
        splits=splits,
        model_dir=tmp_path / "m",
        report_path=report_path,
        mlflow_tracking_uri=uri,
    )

    parsed = json.loads(report_path.read_text(encoding="utf-8"))
    run_id = parsed["registry"]["run_id"]
    client = MlflowClient(tracking_uri=uri)
    artifact_names = {a.path for a in client.list_artifacts(run_id)}
    assert report_path.name in artifact_names


# ---------------------------------------------------------------------------
# Phase 11 — Gulf->MSA train-side augmentation
# ---------------------------------------------------------------------------


class _SentinelNormalizer(TextNormalizerPort):
    """Returns a recognizable sentinel — proves the augmented row reaches fit()."""

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


def test_run_baseline_augmentation_reaches_fitted_vocabulary(tmp_path: Path) -> None:
    splits = HARDDataset.from_rows(_synthetic_rows_with_gulf(20), seed=42)
    model_dir = tmp_path / "m"
    report_path = tmp_path / "r.json"

    run_baseline(
        splits=splits,
        model_dir=model_dir,
        report_path=report_path,
        mlflow_tracking_uri=_tracking_uri(tmp_path),
        normalizer=_SentinelNormalizer(),
    )

    vec = joblib.load(model_dir / "vectorizer.joblib")
    assert _SentinelNormalizer.SENTINEL in vec.vocabulary_


def test_run_baseline_logs_transliterate_gulf_param_when_normalizer_passed(
    tmp_path: Path,
) -> None:
    splits = HARDDataset.from_rows(_synthetic_rows_with_gulf(20), seed=42)
    uri = _tracking_uri(tmp_path)

    run_baseline(
        splits=splits,
        model_dir=tmp_path / "m",
        report_path=tmp_path / "r.json",
        mlflow_tracking_uri=uri,
        normalizer=_SentinelNormalizer(),
    )

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("arabic-sentiment")
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    params = runs[0].data.params
    assert params.get("transliterate_gulf") == "True"
    assert "n_train_original" in params
    assert int(params["n_train"]) > int(params["n_train_original"])


def test_run_baseline_default_path_omits_transliterate_gulf_param(tmp_path: Path) -> None:
    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    uri = _tracking_uri(tmp_path)

    run_baseline(
        splits=splits,
        model_dir=tmp_path / "m",
        report_path=tmp_path / "r.json",
        mlflow_tracking_uri=uri,
    )

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("arabic-sentiment")
    assert experiment is not None
    params = client.search_runs(experiment_ids=[experiment.experiment_id])[0].data.params
    assert "transliterate_gulf" not in params
    assert "n_train_original" not in params
