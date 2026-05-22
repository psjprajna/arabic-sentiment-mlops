"""Curated-lexicon Gulf-vs-MSA dialect tagger for hotel-review text.

This module ships a pure-Python binary tagger (Gulf vs MSA) used by the
training scripts to produce dialect-stratified F1 reports. No port, no
adapter — single implementation, no infra imports (ADR-0002 boundary).

The tagger is deliberately a heuristic. HARD has no dialect ground truth,
and publicly available dialect classifiers are Twitter-trained and do not
transfer to formal hotel reviews. Every consumer of the breakdown sees
`tagger: "lexicon-v1"` so the methodology stays explicit; a future swap
bumps the marker.

Algorithm: strip tashkeel + tatweel, unify alef variants, extract runs of
Arabic letters, then check each token (and its clitic-stripped form)
against `GULF_MARKERS`. A single marker hit flips the label to GULF.

Normalization is local to this module — the raw text passed in is never
mutated (project rule: domain preserves diacritics).
"""

from __future__ import annotations

import re
from enum import StrEnum


class Dialect(StrEnum):
    GULF = "gulf"
    MSA = "msa"


# Curated Gulf-only marker list. Categories:
#   - Question words:        شلون، وش، ايش، اشلون
#   - Discourse / agreement: زين، عاد، ترا، يلا، وايد، خوش
#   - Dialect verbs:         ابغى، فديت، شفت، شفيه، خلني
#   - Time references:       البارح، بكره
#   - Vocatives / kinship:   يبا، ياخوي، ياخي
#   - Misc dialect:          بعدين، هذيله، فدوه، خبل، ودي، عبالي، يم
# Dropped from earlier drafts as too MSA-common: مال, حقي, حلو, روحي,
# يطلع, هاك, عقب — each appears in formal MSA hotel-review contexts.
# Bumping the marker list is forward-compatible under `lexicon-v1`; an
# algorithmic change (multi-token threshold, weighted scoring, etc.)
# would bump to `lexicon-v2`.
GULF_MARKERS: frozenset[str] = frozenset(
    {
        "شلون",
        "وش",
        "ايش",
        "اشلون",
        "زين",
        "عاد",
        "ترا",
        "يلا",
        "وايد",
        "خوش",
        "ابغى",
        "فديت",
        "شفت",
        "شفيه",
        "خلني",
        "البارح",
        "بكره",
        "يبا",
        "ياخوي",
        "ياخي",
        "بعدين",
        "هذيله",
        "فدوه",
        "خبل",
        "ودي",
        "عبالي",
        "يم",
    }
)

# Combining diacritics (U+064B–U+065F: fathatan through hamza-below
# variants), superscript alef (U+0670), Quranic marks (U+06D6–U+06ED),
# and tatweel/kashida (U+0640).
_TASHKEEL_RE = re.compile(r"[ـً-ٰٟۖ-ۭ]")

# Alef variants normalize to plain alef for matching (does NOT mutate
# the caller's text — applies only inside classify_dialect).
_ALEF_VARIANTS_RE = re.compile(r"[آأإٱ]")

# Arabic letter runs only. Skips Latin, digits, punctuation, whitespace.
_ARABIC_LETTERS_RE = re.compile(r"[ء-ي]+")

_CLITIC_PREFIXES = ("و", "ف", "ل", "ب", "ك")
_DEFINITE_ARTICLE = "ال"


def _normalize(text: str) -> str:
    return _ALEF_VARIANTS_RE.sub("ا", _TASHKEEL_RE.sub("", text))


def _strip_clitic(token: str) -> str:
    if token and token[0] in _CLITIC_PREFIXES:
        token = token[1:]
    if token.startswith(_DEFINITE_ARTICLE):
        token = token[len(_DEFINITE_ARTICLE) :]
    return token


def classify_dialect(text: str) -> Dialect:
    """Tag a single piece of Arabic text as Gulf or MSA.

    Returns `Dialect.GULF` if any token (or its clitic-stripped form)
    matches `GULF_MARKERS`; otherwise `Dialect.MSA`. Empty, whitespace-
    only, numeric-only, or Latin-only inputs return `Dialect.MSA` by
    construction (no marker hits).
    """
    if not text or not text.strip():
        return Dialect.MSA
    normalized = _normalize(text)
    for token in _ARABIC_LETTERS_RE.findall(normalized):
        if token in GULF_MARKERS or _strip_clitic(token) in GULF_MARKERS:
            return Dialect.GULF
    return Dialect.MSA
