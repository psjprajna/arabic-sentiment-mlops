"""Tests for HARDDataset loader (hermetic — no network)."""

from __future__ import annotations

import pytest

from sentiment.adapters.hard_dataset import (
    DatasetSplits,
    Example,
    HARDDataset,
    _rating_to_sentiment,
)
from sentiment.domain.models import Sentiment


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        (1, Sentiment.NEGATIVE),
        (2, Sentiment.NEGATIVE),
        (3, Sentiment.NEUTRAL),
        (4, Sentiment.POSITIVE),
        (5, Sentiment.POSITIVE),
    ],
)
def test_rating_to_sentiment_mapping(rating: int, expected: Sentiment) -> None:
    assert _rating_to_sentiment(rating) is expected


@pytest.mark.parametrize("rating", [0, 6, -1, 10])
def test_rating_to_sentiment_rejects_out_of_range(rating: int) -> None:
    with pytest.raises(ValueError):
        _rating_to_sentiment(rating)


def _balanced_rows(n_per_class: int = 10) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for i in range(n_per_class):
        rows.append((f"رائع جدا {i}", 5))
        rows.append((f"سيء للغاية {i}", 1))
        rows.append((f"عادي ومتوسط {i}", 3))
    return rows


def test_loader_schema_returns_typed_examples() -> None:
    rows = _balanced_rows(10)
    splits = HARDDataset.from_rows(rows, seed=42)
    assert isinstance(splits, DatasetSplits)
    for split in (splits.train, splits.dev, splits.test):
        assert split, "splits must be non-empty"
        for item in split:
            assert isinstance(item, Example)
            assert isinstance(item.text, str)
            assert isinstance(item.sentiment, Sentiment)


def test_stratified_split_deterministic() -> None:
    rows = _balanced_rows(10)
    a = HARDDataset.from_rows(rows, seed=42)
    b = HARDDataset.from_rows(rows, seed=42)
    assert [e.text for e in a.train] == [e.text for e in b.train]
    assert [e.text for e in a.dev] == [e.text for e in b.dev]
    assert [e.text for e in a.test] == [e.text for e in b.test]


def test_stratified_split_proportions_within_one_sample() -> None:
    rows = _balanced_rows(20)  # 60 total, perfectly balanced
    splits = HARDDataset.from_rows(rows, seed=42)
    total = len(splits.train) + len(splits.dev) + len(splits.test)
    assert total == len(rows)
    # No example duplicated across splits
    train_t = {e.text for e in splits.train}
    dev_t = {e.text for e in splits.dev}
    test_t = {e.text for e in splits.test}
    assert len(train_t | dev_t | test_t) == total
    # Each class appears in each split within +/-1 of the perfect proportion
    for split in (splits.train, splits.dev, splits.test):
        counts = {s: 0 for s in Sentiment}
        for ex in split:
            counts[ex.sentiment] += 1
        ideal = len(split) / 3
        for c in counts.values():
            assert abs(c - ideal) <= 1, f"class imbalance {counts} in split of size {len(split)}"
