"""Tests for the mlflow_logging.log_run helper (SQLite-backed tracking)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from sentiment.adapters.catboost_classifier import CatBoostAdapter
from sentiment.adapters.hard_dataset import HARDDataset
from sentiment.training.baseline import run_baseline
from sentiment.training.mlflow_logging import (
    SentimentPyfunc,
    log_artifact_to_run,
    log_run,
)


def _tracking_uri(tmp_path: Path) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{tmp_path}/mlflow.db"


def _unique_experiment(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _synthetic_rows(n_per_class: int = 20) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for i in range(n_per_class):
        rows.append((f"رائع ممتاز جدا الفندق نظيف الخدمة سريعة {i}", 5))
        rows.append((f"سيء جدا قذر مزعج فظيع غير محترم {i}", 1))
        rows.append((f"عادي مقبول لا بأس متوسط لا أكثر {i}", 3))
    return rows


def _build_catboost_fixture(tmp_path: Path, *, tracking_uri: str) -> Path:
    """Train a tiny CatBoost model and return its model_dir.

    ``tracking_uri`` is forwarded to ``run_baseline`` so the inner
    MLflow registration lands in the caller's per-test sqlite db
    rather than the project default (Phase 9 Step 0: lessons.md
    2026-05-22 | testing).
    """
    splits = HARDDataset.from_rows(_synthetic_rows(20), seed=42)
    model_dir = tmp_path / "cb"
    report_path = tmp_path / "cb-report.json"
    run_baseline(
        splits=splits,
        model_dir=model_dir,
        report_path=report_path,
        mlflow_tracking_uri=tracking_uri,
    )
    return model_dir


def _write_meta(tmp_path: Path, backend_type: str) -> Path:
    meta = tmp_path / "pyfunc_meta.json"
    meta.write_text(json.dumps({"backend_type": backend_type}), encoding="utf-8")
    return meta


def test_log_run_creates_experiment_and_run(tmp_path: Path) -> None:
    exp = _unique_experiment("create-run")
    run_id = log_run(
        run_name="t1",
        params={"a": 1},
        metrics={"f1": 0.5},
        artifact_paths=[],
        experiment_name=exp,
        tracking_uri=_tracking_uri(tmp_path),
    )

    mlflow.set_tracking_uri(_tracking_uri(tmp_path))
    experiment = mlflow.get_experiment_by_name(exp)
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == run_id


def test_log_run_persists_params_and_metrics(tmp_path: Path) -> None:
    exp = _unique_experiment("params")
    run_id = log_run(
        run_name="t-params",
        params={"base_model": "x", "rank": 8},
        metrics={"f1_macro": 0.8},
        artifact_paths=[],
        experiment_name=exp,
        tracking_uri=_tracking_uri(tmp_path),
    )

    mlflow.set_tracking_uri(_tracking_uri(tmp_path))
    fetched = mlflow.get_run(run_id)
    assert fetched.data.params["base_model"] == "x"
    assert fetched.data.params["rank"] == "8"  # mlflow stores params as strings
    assert fetched.data.metrics["f1_macro"] == 0.8


def test_log_run_logs_artifact_files(tmp_path: Path) -> None:
    exp = _unique_experiment("artifacts")
    artifact = tmp_path / "r.json"
    artifact.write_text("{}", encoding="utf-8")
    run_id = log_run(
        run_name="t-art",
        params={},
        metrics={},
        artifact_paths=[artifact],
        experiment_name=exp,
        tracking_uri=_tracking_uri(tmp_path),
    )

    mlflow.set_tracking_uri(_tracking_uri(tmp_path))
    client = mlflow.MlflowClient(tracking_uri=_tracking_uri(tmp_path))
    listed = client.list_artifacts(run_id)
    names = {a.path for a in listed}
    assert "r.json" in names


def test_log_run_reuses_existing_experiment(tmp_path: Path) -> None:
    exp = _unique_experiment("reuse")
    log_run(
        run_name="r1",
        params={},
        metrics={},
        artifact_paths=[],
        experiment_name=exp,
        tracking_uri=_tracking_uri(tmp_path),
    )
    log_run(
        run_name="r2",
        params={},
        metrics={},
        artifact_paths=[],
        experiment_name=exp,
        tracking_uri=_tracking_uri(tmp_path),
    )

    mlflow.set_tracking_uri(_tracking_uri(tmp_path))
    experiment = mlflow.get_experiment_by_name(exp)
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 2


# ---------------------------------------------------------------------------
# Phase 6 — registry surfacing (SentimentPyfunc + register_as in log_run)
# ---------------------------------------------------------------------------


def test_log_run_registers_model_when_register_as_set(tmp_path: Path) -> None:
    exp = _unique_experiment("register")
    uri = _tracking_uri(tmp_path / "mlruns")
    model_dir = _build_catboost_fixture(tmp_path, tracking_uri=uri)
    meta = _write_meta(tmp_path, "catboost")
    name = f"catboost-baseline-{uuid.uuid4().hex[:8]}"

    result = log_run(
        run_name="t-register",
        params={"backend": "catboost"},
        metrics={"f1_macro": 0.5},
        artifact_paths=[],
        experiment_name=exp,
        tracking_uri=uri,
        register_as=name,
        python_model=SentimentPyfunc(),
        artifacts={"backend_type": str(meta), "model_dir": str(model_dir)},
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "name",
        "version",
        "run_id",
        "model_uri",
        "registered_at",
    }
    assert result["name"] == name
    assert result["version"] == "1"
    assert result["model_uri"] == f"runs:/{result['run_id']}/model"

    client = MlflowClient(tracking_uri=uri)
    assert client.get_registered_model(name) is not None
    versions = client.search_model_versions(f"name='{name}'")
    assert len(versions) == 1


def test_log_run_increments_version_on_second_call(tmp_path: Path) -> None:
    exp = _unique_experiment("incr")
    uri = _tracking_uri(tmp_path / "mlruns")
    model_dir = _build_catboost_fixture(tmp_path, tracking_uri=uri)
    meta = _write_meta(tmp_path, "catboost")
    name = f"catboost-baseline-{uuid.uuid4().hex[:8]}"

    common = dict(
        run_name="t-incr",
        params={"backend": "catboost"},
        metrics={"f1_macro": 0.5},
        artifact_paths=[],
        experiment_name=exp,
        tracking_uri=uri,
        register_as=name,
        python_model=SentimentPyfunc(),
        artifacts={"backend_type": str(meta), "model_dir": str(model_dir)},
    )

    first = log_run(**common)
    second = log_run(**common)

    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert first["version"] == "1"
    assert second["version"] == "2"

    client = MlflowClient(tracking_uri=uri)
    versions = client.search_model_versions(f"name='{name}'")
    assert len(versions) == 2


def test_log_run_backward_compatible_when_register_as_none(tmp_path: Path) -> None:
    exp = _unique_experiment("compat")
    result = log_run(
        run_name="t-compat",
        params={"a": 1},
        metrics={"f1": 0.5},
        artifact_paths=[],
        experiment_name=exp,
        tracking_uri=_tracking_uri(tmp_path),
    )
    assert isinstance(result, str)


@pytest.mark.parametrize(
    ("python_model", "artifacts"),
    [
        (None, {"backend_type": "x", "model_dir": "y"}),
        (SentimentPyfunc(), None),
        (None, None),
    ],
)
def test_log_run_validates_register_args(
    tmp_path: Path,
    python_model: mlflow.pyfunc.PythonModel | None,
    artifacts: dict[str, str] | None,
) -> None:
    exp = _unique_experiment("validate")
    uri = _tracking_uri(tmp_path / "mlruns")

    with pytest.raises(ValueError):
        log_run(
            run_name="t-validate",
            params={},
            metrics={},
            artifact_paths=[],
            experiment_name=exp,
            tracking_uri=uri,
            register_as="some-name",
            python_model=python_model,
            artifacts=artifacts,
        )

    # No MLflow side effect before the raise.
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(exp)
    if experiment is not None:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) == 0


def test_pyfunc_roundtrip_predictions_match_inprocess_classifier(
    tmp_path: Path,
) -> None:
    exp = _unique_experiment("roundtrip")
    uri = _tracking_uri(tmp_path / "mlruns")
    model_dir = _build_catboost_fixture(tmp_path, tracking_uri=uri)
    meta = _write_meta(tmp_path, "catboost")
    name = f"catboost-baseline-{uuid.uuid4().hex[:8]}"

    sentences = ["الفندق ممتاز", "الخدمة سيئة جداً", "الموقع عادي"]

    adapter = CatBoostAdapter(model_dir=model_dir)
    expected = [adapter.predict(s).sentiment.value for s in sentences]

    result = log_run(
        run_name="t-roundtrip",
        params={"backend": "catboost"},
        metrics={"f1_macro": 0.5},
        artifact_paths=[],
        experiment_name=exp,
        tracking_uri=uri,
        register_as=name,
        python_model=SentimentPyfunc(),
        artifacts={"backend_type": str(meta), "model_dir": str(model_dir)},
    )
    assert isinstance(result, dict)

    mlflow.set_tracking_uri(uri)
    loaded = mlflow.pyfunc.load_model(result["model_uri"])
    df = loaded.predict(pd.DataFrame({"text": sentences}))

    assert list(df["sentiment"]) == expected
    schema = loaded.metadata.get_input_schema()
    assert schema is not None
    input_cols = schema.input_names()
    assert input_cols == ["text"]


def test_log_artifact_to_run_attaches_file_to_existing_run(tmp_path: Path) -> None:
    exp = _unique_experiment("attach")
    uri = _tracking_uri(tmp_path / "mlruns")
    run_id = log_run(
        run_name="t-attach",
        params={},
        metrics={},
        artifact_paths=[],
        experiment_name=exp,
        tracking_uri=uri,
    )
    assert isinstance(run_id, str)

    artifact = tmp_path / "after.json"
    artifact.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    log_artifact_to_run(run_id=run_id, tracking_uri=uri, path=artifact)

    client = MlflowClient(tracking_uri=uri)
    names = {a.path for a in client.list_artifacts(run_id)}
    assert "after.json" in names


def test_default_tracking_uri_is_sqlite_across_modules() -> None:
    import inspect

    from sentiment.adapters.mlflow_registry_classifier import (
        MLflowRegistryAdapter,
        load_from_registry_or_fallback,
    )
    from sentiment.training import baseline, lora
    from sentiment.training import mlflow_logging as ml_logging

    expected = "sqlite:///mlflow.db"

    assert ml_logging._DEFAULT_TRACKING_URI == expected
    assert baseline._DEFAULT_TRACKING_URI == expected
    assert lora._DEFAULT_TRACKING_URI == expected

    adapter_sig = inspect.signature(MLflowRegistryAdapter.__init__)
    assert adapter_sig.parameters["tracking_uri"].default == expected

    loader_sig = inspect.signature(load_from_registry_or_fallback)
    assert loader_sig.parameters["tracking_uri"].default == expected

    api_main_path = Path(__file__).parent.parent / "src" / "api" / "main.py"
    api_main_src = api_main_path.read_text(encoding="utf-8")
    assert f'"{expected}"' in api_main_src
    assert '"file:./mlruns"' not in api_main_src
