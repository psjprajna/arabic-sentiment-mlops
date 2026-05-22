"""Unit tests for compute_dialect_breakdown.

Pure-function tests over synthetic (texts, y_true, y_pred) inputs.
No model loading, no IO. Verifies the public schema (tagger marker,
gulf/msa buckets, gulf_share), the insufficient-examples branch at
n<20, ordering invariants on the confusion matrix, and graceful
behavior on degenerate inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sentiment.domain.dialect import Dialect
from sentiment.domain.models import Sentiment
from sentiment.training.dialect_breakdown import compute_dialect_breakdown

_POS = 0
_NEG = 1
_NEU = 2


def _gulf_text(i: int) -> str:
    # Contains the Gulf marker "شلون" — guaranteed GULF tag.
    return f"الفندق شلون الخدمة {i}"


def _msa_text(i: int) -> str:
    # Pure-MSA sentence, no Gulf markers — guaranteed MSA tag.
    return f"الفندق ممتاز جدا والإقامة مريحة {i}"


def _make_fixture(
    n_gulf: int, n_msa: int, *, perfect: bool = True
) -> tuple[list[str], np.ndarray, np.ndarray]:
    texts: list[str] = []
    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    labels_cycle = (_POS, _NEG, _NEU)
    for i in range(n_gulf):
        texts.append(_gulf_text(i))
        lbl = labels_cycle[i % 3]
        y_true_list.append(lbl)
        y_pred_list.append(lbl if perfect else labels_cycle[(i + 1) % 3])
    for i in range(n_msa):
        texts.append(_msa_text(i))
        lbl = labels_cycle[i % 3]
        y_true_list.append(lbl)
        y_pred_list.append(lbl if perfect else labels_cycle[(i + 1) % 3])
    return texts, np.array(y_true_list, dtype=np.int64), np.array(y_pred_list, dtype=np.int64)


def test_breakdown_returns_tagger_name() -> None:
    texts, y_true, y_pred = _make_fixture(30, 30)
    result = compute_dialect_breakdown(texts, y_true, y_pred)
    assert result["tagger"] == "lexicon-v1"


def test_breakdown_returns_both_buckets_when_sufficient() -> None:
    texts, y_true, y_pred = _make_fixture(30, 30)
    result = compute_dialect_breakdown(texts, y_true, y_pred)
    for bucket in ("gulf", "msa"):
        block = result[bucket]
        assert isinstance(block, dict)
        assert "f1_per_class" in block
        assert "f1_macro" in block
        assert "confusion_matrix" in block
        assert block["n"] == 30


def test_breakdown_gulf_share_computed_correctly() -> None:
    texts, y_true, y_pred = _make_fixture(30, 30)
    result = compute_dialect_breakdown(texts, y_true, y_pred)
    assert abs(float(result["gulf_share"]) - 0.5) < 1e-9


def test_breakdown_perfect_predictions_yield_f1_one() -> None:
    texts, y_true, y_pred = _make_fixture(30, 30, perfect=True)
    result = compute_dialect_breakdown(texts, y_true, y_pred)
    for bucket in ("gulf", "msa"):
        block = result[bucket]
        assert isinstance(block, dict)
        assert block["f1_macro"] == 1.0
        for cls in ("positive", "negative", "neutral"):
            assert block["f1_per_class"][cls] == 1.0


def test_breakdown_gulf_bucket_below_floor_returns_insufficient() -> None:
    texts, y_true, y_pred = _make_fixture(5, 30)
    result = compute_dialect_breakdown(texts, y_true, y_pred)
    assert result["gulf"] == {"insufficient_examples": True, "n": 5}
    msa_block = result["msa"]
    assert isinstance(msa_block, dict)
    assert msa_block["n"] == 30
    assert "f1_macro" in msa_block


def test_breakdown_msa_bucket_below_floor_returns_insufficient() -> None:
    texts, y_true, y_pred = _make_fixture(30, 5)
    result = compute_dialect_breakdown(texts, y_true, y_pred)
    assert result["msa"] == {"insufficient_examples": True, "n": 5}
    gulf_block = result["gulf"]
    assert isinstance(gulf_block, dict)
    assert gulf_block["n"] == 30
    assert "f1_macro" in gulf_block


def test_breakdown_both_below_floor_does_not_raise() -> None:
    texts, y_true, y_pred = _make_fixture(5, 5)
    result = compute_dialect_breakdown(texts, y_true, y_pred)
    assert result["gulf"] == {"insufficient_examples": True, "n": 5}
    assert result["msa"] == {"insufficient_examples": True, "n": 5}
    assert abs(float(result["gulf_share"]) - 0.5) < 1e-9


def test_breakdown_uses_injected_tagger() -> None:
    texts, y_true, y_pred = _make_fixture(15, 15)

    def _force_gulf(_text: str) -> Dialect:
        return Dialect.GULF

    result = compute_dialect_breakdown(texts, y_true, y_pred, tagger=_force_gulf)
    assert result["msa"] == {"insufficient_examples": True, "n": 0}
    gulf_block = result["gulf"]
    assert isinstance(gulf_block, dict)
    assert gulf_block["n"] == 30
    assert abs(float(result["gulf_share"]) - 1.0) < 1e-9


def test_breakdown_confusion_matrix_matches_label_order() -> None:
    # All Gulf rows: y_true POSITIVE, y_pred NEGATIVE → cm[0][1] == 30.
    texts: list[str] = [_gulf_text(i) for i in range(30)]
    y_true = np.array([_POS] * 30, dtype=np.int64)
    y_pred = np.array([_NEG] * 30, dtype=np.int64)
    result = compute_dialect_breakdown(texts, y_true, y_pred)
    gulf_block = result["gulf"]
    assert isinstance(gulf_block, dict)
    cm = gulf_block["confusion_matrix"]
    assert isinstance(cm, list)
    assert cm[0][1] == 30
    flat = [v for row in cm for v in row]
    assert sum(flat) == 30
    assert all(v == 0 for v in flat if v != 30)


def test_breakdown_empty_inputs_does_not_raise() -> None:
    texts: Sequence[str] = []
    y_true = np.array([], dtype=np.int64)
    y_pred = np.array([], dtype=np.int64)
    result = compute_dialect_breakdown(texts, y_true, y_pred)
    assert result["gulf"] == {"insufficient_examples": True, "n": 0}
    assert result["msa"] == {"insufficient_examples": True, "n": 0}
    assert float(result["gulf_share"]) == 0.0


def test_breakdown_label_order_matches_baseline() -> None:
    from sentiment.training.baseline import _LABEL_ORDER as BASELINE_LABEL_ORDER
    from sentiment.training.dialect_breakdown import _LABEL_ORDER as BREAKDOWN_LABEL_ORDER

    assert tuple(BREAKDOWN_LABEL_ORDER) == tuple(BASELINE_LABEL_ORDER)
    assert tuple(BREAKDOWN_LABEL_ORDER) == (
        Sentiment.POSITIVE,
        Sentiment.NEGATIVE,
        Sentiment.NEUTRAL,
    )
