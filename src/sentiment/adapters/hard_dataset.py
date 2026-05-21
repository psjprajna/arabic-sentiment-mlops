"""HARD Arabic Reviews dataset loader.

Maps 5-star ratings to the 3-class Sentiment enum and produces a deterministic
stratified 80/10/10 split.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from sentiment.domain.models import Sentiment


@dataclass(frozen=True)
class Example:
    text: str
    sentiment: Sentiment


@dataclass(frozen=True)
class DatasetSplits:
    train: list[Example]
    dev: list[Example]
    test: list[Example]


def _rating_to_sentiment(rating: int) -> Sentiment:
    if rating in (1, 2):
        return Sentiment.NEGATIVE
    if rating == 3:
        return Sentiment.NEUTRAL
    if rating in (4, 5):
        return Sentiment.POSITIVE
    raise ValueError(f"rating must be in 1..5, got {rating}")


_HF_SOURCES: tuple[str, ...] = ("arbml/HARD", "Asma-Mekki/HARD")


class HARDDataset:
    """Loader for the HARD Arabic-Reviews dataset.

    Default behavior is lazy: actual download happens in `load()`. Tests use
    `from_rows` to avoid network.
    """

    def __init__(self, source: str = "default", seed: int = 42) -> None:
        self.source = source
        self.seed = seed

    def load(self) -> DatasetSplits:
        rows = _fetch_hf_rows(self.source)
        return self.from_rows(rows, seed=self.seed)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[tuple[str, int]],
        seed: int = 42,
    ) -> DatasetSplits:
        examples = [
            Example(text=text, sentiment=_rating_to_sentiment(rating))
            for text, rating in rows
            if text and text.strip()
        ]
        if not examples:
            raise ValueError("no usable rows after filtering")

        labels = [ex.sentiment.value for ex in examples]
        train, rest, _, rest_labels = train_test_split(
            examples,
            labels,
            test_size=0.20,
            random_state=seed,
            stratify=labels,
        )
        dev, test = train_test_split(
            rest,
            test_size=0.50,
            random_state=seed,
            stratify=rest_labels,
        )
        return DatasetSplits(train=list(train), dev=list(dev), test=list(test))


def _fetch_hf_rows(source: str) -> list[tuple[str, int]]:
    """Pull the HARD dataset rows lazily.

    Tries the configured source order if `source == "default"`, else uses the
    explicit source name. Network access is isolated to this helper so unit
    tests never hit it.
    """
    from datasets import load_dataset  # local import — heavy + needs network

    sources = _HF_SOURCES if source == "default" else (source,)
    last_err: Exception | None = None
    for name in sources:
        try:
            ds = load_dataset(name, split="train")
        except Exception as err:  # noqa: BLE001 — surface fallback failures together
            last_err = err
            continue
        rows: list[tuple[str, int]] = []
        for record in ds:
            text = record.get("text") or record.get("review")
            rating_raw = record.get("rating") or record.get("label")
            if text is None or rating_raw is None:
                continue
            rows.append((str(text), int(rating_raw)))
        if rows:
            return rows
    raise RuntimeError(f"could not load HARD from any source; last error: {last_err}")
