"""File-backed MLflow run logging helper.

Opens a single MLflow run inside a named experiment, logs params + metrics +
artifact files, returns the resulting run_id. Isolates the `mlflow` import
surface so callers in `sentiment/training/` don't depend on mlflow's call
shape (set_experiment, start_run context manager, log_artifact per file).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import mlflow

_DEFAULT_EXPERIMENT = "arabic-sentiment"
_DEFAULT_TRACKING_URI = "file:./mlruns"


def log_run(
    *,
    run_name: str,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    artifact_paths: Sequence[Path],
    experiment_name: str = _DEFAULT_EXPERIMENT,
    tracking_uri: str = _DEFAULT_TRACKING_URI,
) -> str:
    """Open one MLflow run, log everything, return the run_id."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(dict(params))
        if metrics:
            mlflow.log_metrics(dict(metrics))
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))
        return run.info.run_id
