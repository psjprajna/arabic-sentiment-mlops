"""Unit tests for GulfToMSANormalizer (transliterator-v1)."""

from __future__ import annotations

import pytest

from sentiment.adapters.gulf_to_msa_normalizer import GulfToMSANormalizer
from sentiment.domain.dialect import GULF_MARKERS


@pytest.fixture
def normalizer() -> GulfToMSANormalizer:
    return GulfToMSANormalizer()


def test_normalize_substitutes_single_gulf_marker(normalizer: GulfToMSANormalizer) -> None:
    out = normalizer.normalize("شلون الجو")
    assert "كيف" in out
    assert "شلون" not in out


def test_normalize_substitutes_multiple_markers(normalizer: GulfToMSANormalizer) -> None:
    out = normalizer.normalize("ابغى اشوف الفندق وايد")
    assert "أريد" in out
    assert "كثيرا" in out
    assert "ابغى" not in out
    assert "وايد" not in out


def test_normalize_handles_clitic_prefix(normalizer: GulfToMSANormalizer) -> None:
    out = normalizer.normalize("وشلون الفندق")
    assert "وكيف" in out
    assert "شلون" not in out


def test_normalize_handles_definite_article_inside_marker(
    normalizer: GulfToMSANormalizer,
) -> None:
    out = normalizer.normalize("البارح كان زين")
    assert "أمس" in out
    assert "البارح" not in out


def test_normalize_handles_clitic_plus_article(normalizer: GulfToMSANormalizer) -> None:
    out = normalizer.normalize("بالبارح")
    assert "أمس" in out


def test_normalize_idempotent_on_pure_msa(normalizer: GulfToMSANormalizer) -> None:
    msa = "الفندق نظيف والموظفون لطفاء"
    assert normalizer.normalize(msa) == msa


def test_normalize_idempotent_twice(normalizer: GulfToMSANormalizer) -> None:
    text = "شلون الجو اليوم"
    once = normalizer.normalize(text)
    twice = normalizer.normalize(once)
    assert once == twice


def test_normalize_preserves_non_arabic_runs(normalizer: GulfToMSANormalizer) -> None:
    out = normalizer.normalize("شلون the hotel? 5 stars!")
    assert "the hotel?" in out
    assert "5 stars!" in out
    assert "كيف" in out


def test_normalize_preserves_diacritics_on_unmatched_tokens(
    normalizer: GulfToMSANormalizer,
) -> None:
    # "الْفُنْدُقُ" — fully voweled MSA word, not a Gulf marker. Tashkeel must
    # survive unchanged. Project Arabic rule: only matched markers mutate.
    text = "الْفُنْدُقُ"
    assert normalizer.normalize(text) == text


def test_normalize_preserves_diacritics_alongside_substitution(
    normalizer: GulfToMSANormalizer,
) -> None:
    # Tashkeel on a non-marker word stays; marker is replaced.
    text = "شلون الْفُنْدُقُ"
    out = normalizer.normalize(text)
    assert "كيف" in out
    assert "الْفُنْدُقُ" in out


def test_normalize_handles_normalized_alef_in_marker(
    normalizer: GulfToMSANormalizer,
) -> None:
    # "أبغى" — hamzated alef; matches plain-alef key "ابغى" after lookup
    # normalization. The replacement is emitted in clean MSA form.
    out = normalizer.normalize("أبغى غرفة")
    assert "أريد" in out
    assert "أبغى" not in out


def test_normalize_empty_string(normalizer: GulfToMSANormalizer) -> None:
    assert normalizer.normalize("") == ""


def test_normalize_whitespace_only(normalizer: GulfToMSANormalizer) -> None:
    assert normalizer.normalize("   ") == "   "


def test_every_gulf_marker_has_msa_mapping() -> None:
    missing = set(GULF_MARKERS) - set(GulfToMSANormalizer._GULF_TO_MSA)
    assert not missing, f"GulfToMSANormalizer missing MSA mapping for: {sorted(missing)}"
