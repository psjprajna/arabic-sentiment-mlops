"""Integration tests for the retrain-on-drift CLI (spec 009 §Testing §Integration)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from sentiment.training import retrain_on_drift


def _fake_run_baseline(**_kwargs: object) -> dict[str, object]:
    return {
        "f1_macro": 0.77,
        "registry": {
            "name": "catboost-baseline",
            "version": "42",
            "run_id": "fakerun",
            "model_uri": "models:/catboost-baseline/42",
            "registered_at": "2026-05-23T00:00:00Z",
        },
    }


def _fake_run_lora(**_kwargs: object) -> dict[str, object]:
    return {
        "f1_macro": 0.84,
        "registry": {
            "name": "arabert-lora",
            "version": "7",
            "run_id": "fakerun",
            "model_uri": "models:/arabert-lora/7",
            "registered_at": "2026-05-23T00:00:00Z",
        },
    }


@pytest.fixture(autouse=True)
def _stub_load_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid the real HARDDataset.load() (network + heavy) in CLI tests."""
    monkeypatch.setattr(retrain_on_drift, "_load_splits", lambda: object())


@pytest.fixture
def chdir_to_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _fire_report() -> dict[str, object]:
    return {
        "backend": "catboost-baseline-v1",
        "observed_count": 500,
        "buffer_size": 1000,
        "minimum_count": 50,
        "insufficient_data": False,
        "signals": {
            "predicted_class": {"psi": 0.31, "drift_level": "significant"},
        },
    }


def _stable_report() -> dict[str, object]:
    return {
        "backend": "catboost-baseline-v1",
        "observed_count": 500,
        "buffer_size": 1000,
        "minimum_count": 50,
        "insufficient_data": False,
        "signals": {
            "predicted_class": {"psi": 0.05, "drift_level": "stable"},
        },
    }


def _read_log_entries(tmp_path: Path) -> list[dict[str, object]]:
    log_path = tmp_path / "reports" / "retrain-decisions.jsonl"
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def test_cli_dry_run_does_not_invoke_backend_but_writes_log(
    chdir_to_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[dict[str, object]] = []

    def trap(**kwargs: object) -> dict[str, object]:
        called.append(kwargs)
        return _fake_run_baseline(**kwargs)

    monkeypatch.setattr("sentiment.training.baseline.run_baseline", trap)
    report_path = chdir_to_tmp / "drift.json"
    report_path.write_text(json.dumps(_fire_report()))

    exit_code = retrain_on_drift.main(
        ["--backend", "catboost", "--drift-report", str(report_path), "--dry-run"]
    )
    assert exit_code == 0
    assert called == []
    entries = _read_log_entries(chdir_to_tmp)
    assert len(entries) == 1
    assert entries[0]["dry_run"] is True
    assert entries[0]["decision"] is True
    assert entries[0]["registry"] is None


def test_cli_no_op_decision_writes_log_and_exits_zero(
    chdir_to_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[dict[str, object]] = []
    monkeypatch.setattr(
        "sentiment.training.baseline.run_baseline",
        lambda **kw: (called.append(kw), _fake_run_baseline(**kw))[1],
    )
    report_path = chdir_to_tmp / "drift.json"
    report_path.write_text(json.dumps(_stable_report()))

    exit_code = retrain_on_drift.main(["--backend", "catboost", "--drift-report", str(report_path)])
    assert exit_code == 0
    assert called == []
    entries = _read_log_entries(chdir_to_tmp)
    assert len(entries) == 1
    assert entries[0]["decision"] is False
    assert entries[0]["registry"] is None


def test_cli_fire_decision_invokes_catboost_backend_and_logs_registry(
    chdir_to_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sentiment.training.baseline.run_baseline", _fake_run_baseline)
    report_path = chdir_to_tmp / "drift.json"
    report_path.write_text(json.dumps(_fire_report()))

    exit_code = retrain_on_drift.main(["--backend", "catboost", "--drift-report", str(report_path)])
    assert exit_code == 0
    entry = _read_log_entries(chdir_to_tmp)[-1]
    assert entry["decision"] is True
    registry = entry["registry"]
    assert isinstance(registry, dict)
    assert registry["name"] == "catboost-baseline"
    assert registry["version"] == "42"


def test_cli_fire_decision_invokes_lora_backend_and_logs_registry(
    chdir_to_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sentiment.training.lora.run_lora_training", _fake_run_lora)
    report_path = chdir_to_tmp / "drift.json"
    report_path.write_text(json.dumps(_fire_report()))

    exit_code = retrain_on_drift.main(["--backend", "lora", "--drift-report", str(report_path)])
    assert exit_code == 0
    entry = _read_log_entries(chdir_to_tmp)[-1]
    registry = entry["registry"]
    assert isinstance(registry, dict)
    assert registry["name"] == "arabert-lora"
    assert registry["version"] == "7"


def test_cli_reads_drift_report_from_stdin(
    chdir_to_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sentiment.training.baseline.run_baseline", _fake_run_baseline)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_fire_report())))

    exit_code = retrain_on_drift.main(["--backend", "catboost", "--drift-report", "-"])
    assert exit_code == 0
    entry = _read_log_entries(chdir_to_tmp)[-1]
    assert entry["drift_report_path"] == "-"
    assert entry["decision"] is True


def test_cli_malformed_json_exits_non_zero_no_log_line(chdir_to_tmp: Path) -> None:
    report_path = chdir_to_tmp / "drift.json"
    report_path.write_text("{not json")

    exit_code = retrain_on_drift.main(["--backend", "catboost", "--drift-report", str(report_path)])
    assert exit_code == 2
    log_path = chdir_to_tmp / "reports" / "retrain-decisions.jsonl"
    assert not log_path.exists()
