"""Domain port: TextNormalizerPort.

Adapters live in sentiment/adapters/ and implement this port.
No infrastructure imports allowed here (ADR-0002).

The port itself imposes no transformation contract — adapters MAY strip
diacritics, transliterate dialect markers, or rewrite tokens for the
purposes of training-set augmentation. Domain code that consumes the port
must not assume the returned text is byte-equal to the input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextNormalizerPort(ABC):
    """Normalize Arabic text for downstream classification or augmentation.

    Implementations must never be imported inside this module.
    Wire adapters in the composition root or in training scripts.
    """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Return a transformed copy of `text`.

        Implementations should be idempotent on text that contains no
        markers they recognize (i.e. `normalize(normalize(x)) == normalize(x)`).
        Empty strings pass through verbatim.
        """
        ...
