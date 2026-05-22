"""Unit tests for the curated-lexicon Gulf/MSA dialect classifier.

Covers marker hits, MSA defaults, normalization (tashkeel, tatweel, alef
variants), clitic-prefix tolerance, the substring false-positive guard,
and the empty / whitespace / numeric / Latin edge cases. All cases must
run in <1 second — pure Python, no external resources.
"""

from __future__ import annotations

import pytest

from sentiment.domain.dialect import GULF_MARKERS, Dialect, classify_dialect


def test_gulf_markers_is_nonempty_frozenset() -> None:
    assert isinstance(GULF_MARKERS, frozenset)
    assert len(GULF_MARKERS) >= 25
    for marker in GULF_MARKERS:
        assert isinstance(marker, str) and marker


@pytest.mark.parametrize("marker", sorted(GULF_MARKERS))
def test_classify_dialect_returns_gulf_on_marker_hit(marker: str) -> None:
    assert classify_dialect(f"الفندق {marker} رائع") is Dialect.GULF


@pytest.mark.parametrize(
    "text",
    [
        "الفندق ممتاز جدا والخدمة رائعة",
        "كانت الإقامة لطيفة والإفطار متنوع",
        "الموقع قريب من المركز التجاري ومن المطار",
        "النظافة جيدة لكن الواي فاي ضعيف في الغرفة",
        "أنصح بهذا الفندق لمن يبحث عن الهدوء",
        "السعر مناسب مقابل الخدمات المقدمة",
        "غرفة واسعة وإطلالة جميلة على البحر",
        "تجربة لا تنسى استمتعنا بكل لحظة",
        "موظفو الاستقبال محترفون ومرحبون",
        "اللوبي أنيق والإضاءة هادئة ومريحة",
    ],
)
def test_classify_dialect_returns_msa_on_pure_msa(text: str) -> None:
    assert classify_dialect(text) is Dialect.MSA


def test_classify_dialect_ignores_tashkeel_on_marker() -> None:
    assert classify_dialect("الفندق زَيْنٌ جداً") is Dialect.GULF


@pytest.mark.parametrize("variant", ["أبغى", "إبغى", "آبغى", "ابغى"])
def test_classify_dialect_unifies_alef_variants(variant: str) -> None:
    assert classify_dialect(f"والله {variant} روحه") is Dialect.GULF


def test_classify_dialect_strips_tatweel() -> None:
    assert classify_dialect("والله شلـــون الفندق") is Dialect.GULF


@pytest.mark.parametrize("clitic", ["و", "ف", "ل", "ب", "ك"])
@pytest.mark.parametrize("marker", ["شلون", "ابغى", "زين", "عاد", "يلا"])
def test_classify_dialect_handles_clitic_prefix(clitic: str, marker: str) -> None:
    assert classify_dialect(f"الفندق {clitic}{marker} هنا") is Dialect.GULF


@pytest.mark.parametrize("token", ["الزين", "وزين", "بزين", "والزين", "فالزين"])
def test_classify_dialect_handles_al_prefix(token: str) -> None:
    assert classify_dialect(f"الفندق {token} جدا") is Dialect.GULF


def test_classify_dialect_msa_substring_not_matched() -> None:
    # "زين" is a Gulf marker; "زينة" must NOT match (different word).
    assert classify_dialect("الفندق زينة المدينة الجديدة") is Dialect.MSA


def test_classify_dialect_returns_msa_on_empty_text() -> None:
    assert classify_dialect("") is Dialect.MSA


def test_classify_dialect_returns_msa_on_whitespace_only() -> None:
    assert classify_dialect("   \n\t  ") is Dialect.MSA


def test_classify_dialect_returns_msa_on_numeric_only() -> None:
    assert classify_dialect("123 456 789") is Dialect.MSA


def test_classify_dialect_returns_msa_on_latin_only() -> None:
    assert classify_dialect("very nice hotel and clean rooms") is Dialect.MSA


def test_classify_dialect_returns_gulf_on_code_switch() -> None:
    assert classify_dialect("الفندق ممتاز، شلون الخدمة عندكم") is Dialect.GULF


@pytest.mark.parametrize("punct", ["،", "!", ".", "؟", ","])
def test_classify_dialect_handles_punctuation_glued_marker(punct: str) -> None:
    assert classify_dialect(f"الفندق شلون{punct} الخدمة") is Dialect.GULF
