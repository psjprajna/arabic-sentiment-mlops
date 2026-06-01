"""Gulf-dialect to MSA transliterator adapter (transliterator-v1).

Implements `TextNormalizerPort` by walking Arabic-letter runs in the input
text and substituting each token (or its clitic-stripped form) against a
hand-curated Gulf->MSA mapping table keyed off `GULF_MARKERS` from the
domain dialect tagger. Non-Arabic spans (Latin, digits, punctuation,
whitespace) are emitted verbatim, and tashkeel on un-matched tokens is
preserved.

Coverage invariant: every entry in `dialect.GULF_MARKERS` must appear as a
key in `_GULF_TO_MSA`. A unit test enforces this so a future addition to
the marker lexicon forces an explicit mapping decision rather than
silently degrading recall.

Methodology marker: "transliterator-v1". A non-lexical change (e.g.
morphological analyser, learned mapping) would bump to v2.
"""

from __future__ import annotations

import re
from typing import ClassVar

from sentiment.domain.normalizer import TextNormalizerPort

# Reuse the tokenization shape from dialect.py without importing private
# regexes — the alphabet boundaries and clitic conventions are part of
# the lexicon's contract, not the tagger's implementation detail.
_TASHKEEL_RE = re.compile(r"[ـً-ٰٟۖ-ۭ]")
_ALEF_VARIANTS_RE = re.compile(r"[آأإٱ]")
_ARABIC_LETTERS_RE = re.compile(r"[ء-ي]+")
_CLITIC_PREFIXES = ("و", "ف", "ل", "ب", "ك")


def _normalize_for_lookup(token: str) -> str:
    return _ALEF_VARIANTS_RE.sub("ا", _TASHKEEL_RE.sub("", token))


def _strip_clitic(token: str) -> tuple[str, str]:
    """Return (clitic_prefix, stripped_token).

    Strips at most one clitic letter — the definite article is left
    attached because some markers (e.g. `البارح`) include their own
    article and must match against the un-stripped form.
    """
    if token and token[0] in _CLITIC_PREFIXES:
        return token[0], token[1:]
    return "", token


class GulfToMSANormalizer(TextNormalizerPort):
    """Substitute Gulf-marker tokens with MSA equivalents.

    Designed for train-side augmentation: callers concatenate the
    original Gulf-tagged training example with the normalized copy so
    the classifier learns both surface forms. Not intended for inference.
    """

    _GULF_TO_MSA: ClassVar[dict[str, str]] = {
        # Question words
        "شلون": "كيف",
        "وش": "ما",
        "ايش": "ما",
        "اشلون": "كيف",
        # Discourse / agreement
        "زين": "جيد",
        "عاد": "إذن",
        "ترا": "علما",
        "يلا": "هيا",
        "وايد": "كثيرا",
        "خوش": "جيد",
        # Dialect verbs
        "ابغى": "أريد",
        "فديت": "أحببت",
        "شفت": "رأيت",
        "شفيه": "ما به",
        "خلني": "دعني",
        # Time references
        "البارح": "أمس",
        "بكره": "غدا",
        # Vocatives / kinship
        "يبا": "أبي",
        "ياخوي": "أخي",
        "ياخي": "أخي",
        # Misc dialect
        "بعدين": "لاحقا",
        "هذيله": "هؤلاء",
        "فدوه": "فداء",
        "خبل": "مجنون",
        "ودي": "أود",
        "عبالي": "ببالي",
        "يم": "قرب",
    }

    def normalize(self, text: str) -> str:
        if not text:
            return text
        out: list[str] = []
        cursor = 0
        for match in _ARABIC_LETTERS_RE.finditer(text):
            out.append(text[cursor : match.start()])
            out.append(self._substitute(match.group()))
            cursor = match.end()
        out.append(text[cursor:])
        return "".join(out)

    def _substitute(self, token: str) -> str:
        normalized = _normalize_for_lookup(token)
        replacement = self._GULF_TO_MSA.get(normalized)
        if replacement is not None:
            return replacement
        prefix, stripped = _strip_clitic(normalized)
        if prefix:
            replacement = self._GULF_TO_MSA.get(stripped)
            if replacement is not None:
                return prefix + replacement
        return token
