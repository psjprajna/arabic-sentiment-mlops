"""File-backed MLflow run logging + Model Registry surfacing.

Opens a single MLflow run inside a named experiment, logs params + metrics +
artifact files, returns the resulting run_id. Isolates the `mlflow` import
surface so callers in `sentiment/training/` don't depend on mlflow's call
shape.

Phase 6 (ADR-0003, decisions D1 + D7) adds polymorphic Model Registry
surfacing: one `SentimentPyfunc(mlflow.pyfunc.PythonModel)` colocated here
dispatches on a `backend_type` artifact field at load time, lazy-imports
the matching adapter (CatBoost or AraBERT-LoRA), and exposes a uniform
`predict(text) -> {text, sentiment, confidence}` shape. `log_run` gains
three optional kwargs (`register_as`, `python_model`, `artifacts`) which
together trigger `mlflow.pyfunc.log_model` + `mlflow.register_model` and
return a `{name, version, run_id, model_uri, registered_at}` dict.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

_DEFAULT_EXPERIMENT = "arabic-sentiment"
_DEFAULT_TRACKING_URI = "file:./mlruns"


class SentimentPyfunc(mlflow.pyfunc.PythonModel):
    """Polymorphic pyfunc dispatching on backend_type at load time."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        meta_path = Path(context.artifacts["backend_type"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        backend = meta["backend_type"]
        model_dir = Path(context.artifacts["model_dir"])
        if backend == "catboost":
            from sentiment.adapters.catboost_classifier import CatBoostAdapter

            self._classifier = CatBoostAdapter(model_dir=model_dir)
        elif backend == "arabert-lora":
            from sentiment.adapters.arabert_lora_classifier import (
                AraBERTLoRAAdapter,
            )

            self._classifier = AraBERTLoRAAdapter(model_dir=model_dir)
        else:
            raise ValueError(f"unknown backend_type: {backend!r}")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: Mapping[str, object] | None = None,
    ) -> pd.DataFrame:
        texts = list(model_input["text"])
        rows = [self._classifier.predict(t) for t in texts]
        return pd.DataFrame(
            [
                {
                    "text": r.text,
                    "sentiment": r.sentiment.value,
                    "confidence": r.confidence,
                }
                for r in rows
            ]
        )


def _validate_register_args(
    register_as: str | None,
    python_model: mlflow.pyfunc.PythonModel | None,
    artifacts: Mapping[str, str] | None,
) -> None:
    if register_as is None:
        return
    if python_model is None or artifacts is None:
        raise ValueError(
            "log_run(register_as=...) requires both python_model and "
            f"artifacts to be set; got python_model={python_model!r}, "
            f"artifacts={artifacts!r}"
        )


def log_run(
    *,
    run_name: str,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    artifact_paths: Sequence[Path],
    experiment_name: str = _DEFAULT_EXPERIMENT,
    tracking_uri: str = _DEFAULT_TRACKING_URI,
    register_as: str | None = None,
    python_model: mlflow.pyfunc.PythonModel | None = None,
    artifacts: Mapping[str, str] | None = None,
) -> str | dict[str, str]:
    """Open one MLflow run, log everything, optionally register a model.

    Backward-compatible: when ``register_as`` is ``None`` returns the raw
    ``run_id`` (string). When set, also logs a pyfunc model and registers
    it under ``register_as``; returns a dict with keys
    ``{name, version, run_id, model_uri, registered_at}``.
    """
    _validate_register_args(register_as, python_model, artifacts)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(dict(params))
        if metrics:
            mlflow.log_metrics(dict(metrics))
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))
        if register_as is None:
            return run.info.run_id

        assert python_model is not None  # narrowed by _validate_register_args
        assert artifacts is not None
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=python_model,
            artifacts=dict(artifacts),
            input_example=pd.DataFrame({"text": ["مرحبا"]}),
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=register_as)
        return {
            "name": result.name,
            "version": str(result.version),
            "run_id": run.info.run_id,
            "model_uri": model_uri,
            "registered_at": datetime.now(UTC).isoformat(),
        }


def log_artifact_to_run(*, run_id: str, tracking_uri: str, path: Path) -> None:
    """Attach a file to an already-closed MLflow run.

    Used when an artifact (e.g. the report JSON containing a `registry`
    block computed during `log_run`) can only be written after the run
    that produced it has returned.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    client.log_artifact(run_id=run_id, local_path=str(path))
