"""Session-scoped fixtures shared across the test suite."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True, scope="session")
def _session_mlflow_uri(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Pin ``MLFLOW_TRACKING_URI`` to a session tmp dir.

    Belt-and-braces against the fixture-pollution path documented in
    ``tasks/lessons.md`` (2026-05-22 | testing). Even if a fixture
    forgets to pass ``mlflow_tracking_uri=`` to ``run_baseline`` /
    ``run_lora_training``, any code that reads
    ``os.environ["MLFLOW_TRACKING_URI"]`` (e.g. ``api/main.py``'s
    lifespan) lands in this session-scoped sqlite db instead of the
    real ``app/mlflow.db``.
    """
    session_db = tmp_path_factory.mktemp("mlflow") / "session.db"
    previous = os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{session_db}"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = previous
