"""Tests for the mlflow_logging.log_run helper (file-backed tracking)."""

from __future__ import annotations

import uuid
from pathlib import Path

import mlflow

from sentiment.training.mlflow_logging import log_run


def _tracking_uri(tmp_path: Path) -> str:
    return f"file:{tmp_path}"


def _unique_experiment(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


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
