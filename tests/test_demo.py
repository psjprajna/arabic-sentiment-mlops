"""Tests for the CatBoost demo CLI (mocked adapter, stdout via capsys)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sentiment import demo
from sentiment.demo import CURATED_CASES
from sentiment.domain.models import Sentiment, SentimentResult


class _StubAdapter:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def predict(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        return SentimentResult(text=text, sentiment=Sentiment.POSITIVE, confidence=0.42)


def test_single_text_mode_prints_row(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("sentiment.demo.CatBoostAdapter", _StubAdapter):
        rc = demo.main(["--text", "مرحبا", "--model-dir", "ignored"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "مرحبا" in out
    assert "positive" in out
    assert "0.42" in out


def test_showcase_mode_prints_markdown_table(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("sentiment.demo.CatBoostAdapter", _StubAdapter):
        rc = demo.main(["--model-dir", "ignored"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "| # | input | predicted | confidence |" in out
    assert "|---" in out
    # One row per curated case
    data_lines = [ln for ln in out.splitlines() if ln.startswith("| ") and "---" not in ln]
    # header + N data rows
    assert len(data_lines) == 1 + len(CURATED_CASES)
