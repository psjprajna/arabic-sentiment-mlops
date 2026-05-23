"""CLI: read a /metrics/drift response → apply the gate → optionally retrain.

Slice 9a per spec 009. Pure-gate decision lives in
sentiment.domain.retrain_policy; this module is the operator-facing seam.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from sentiment.domain.retrain_policy import (
    RetrainDecision,
    RetrainGate,
    should_retrain,
)

_DECISION_LOG: Path = Path("reports") / "retrain-decisions.jsonl"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = _load_report(args.drift_report)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"error: cannot read drift report: {exc}", file=sys.stderr)
        return 2
    gate = RetrainGate(minimum_observed_count=args.gate_min_observed)
    decision = should_retrain(report, gate=gate)
    registry: dict[str, str] | None = None
    if decision.should_retrain and not args.dry_run:
        registry = _invoke_backend(args.backend)
    _append_log(args, decision, registry)
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="retrain_on_drift")
    parser.add_argument("--backend", choices=("catboost", "lora"), required=True)
    parser.add_argument(
        "--drift-report",
        required=True,
        help="path to a /metrics/drift JSON response, or '-' for stdin",
    )
    parser.add_argument("--gate-min-observed", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _load_report(path: str) -> dict[str, object]:
    if path == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_splits() -> object:
    """Loaded lazily so `--help` is fast and tests can monkeypatch around HF."""
    from sentiment.adapters.hard_dataset import HARDDataset

    return HARDDataset(source="default", seed=42).load()


def _invoke_backend(backend: str) -> dict[str, str]:
    splits = _load_splits()
    if backend == "catboost":
        from sentiment.training.baseline import run_baseline

        result = run_baseline(
            splits=splits,
            model_dir=Path("models/catboost-baseline-v1"),
            report_path=Path("reports/catboost-baseline-v1.json"),
        )
    else:
        from sentiment.training.lora import run_lora_training

        result = run_lora_training(
            splits=splits,
            model_dir=Path("models/arabert-lora-v1"),
            report_path=Path("reports/arabert-lora-v1.json"),
        )
    registry = result.get("registry")
    if not isinstance(registry, dict):
        raise RuntimeError("training did not return a registry block")
    return {str(k): str(v) for k, v in registry.items()}


def _append_log(
    args: argparse.Namespace,
    decision: RetrainDecision,
    registry: dict[str, str] | None,
) -> None:
    entry: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "backend": args.backend,
        "drift_report_path": args.drift_report,
        "decision": decision.should_retrain,
        "reason": decision.reason,
        "observed_count": decision.observed_count,
        "triggering_signals": [list(item) for item in decision.triggering_signals],
        "dry_run": args.dry_run,
        "registry": registry,
    }
    _DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _DECISION_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
