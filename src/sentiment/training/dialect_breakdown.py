"""Per-dialect F1 + confusion-matrix utility for training-report enrichment.

Consumed by `sentiment.training.baseline` and `sentiment.training.lora`
to stratify the test-set evaluation by Gulf-vs-MSA bucket and embed the
breakdown under a new top-level `dialect_breakdown` key in the report
JSON. Pure function — no IO, no logging.

Schema:

    {
      "tagger": "lexicon-v1",
      "gulf":   <bucket-block>,
      "msa":    <bucket-block>,
      "gulf_share": <float>,
    }

where each <bucket-block> is either
`{"insufficient_examples": True, "n": <int>}` when fewer than 20 test
rows tagged into that bucket, or the full
`{"n", "f1_per_class", "f1_macro", "confusion_matrix"}` shape mirroring
`baseline._evaluate` exactly.

The `tagger: "lexicon-v1"` marker is the methodology declaration —
swapping the tagger (e.g. to a model-based classifier) bumps to
`lexicon-v2` / `madar-v1` and is a breaking change for downstream
consumers reasoning about the labels' provenance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from sentiment.domain.dialect import Dialect, classify_dialect
from sentiment.domain.models import Sentiment

_LABEL_ORDER: tuple[Sentiment, ...] = (
    Sentiment.POSITIVE,
    Sentiment.NEGATIVE,
    Sentiment.NEUTRAL,
)
_MINIMUM_BUCKET_N: int = 20
_TAGGER_NAME: str = "lexicon-v1"


def _bucket_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    n = int(len(y_true))
    if n < _MINIMUM_BUCKET_N:
        return {"insufficient_examples": True, "n": n}
    label_ids = list(range(len(_LABEL_ORDER)))
    per_class = f1_score(
        y_true,
        y_pred,
        labels=label_ids,
        average=None,
        zero_division=0.0,
    )
    macro = float(
        f1_score(
            y_true,
            y_pred,
            labels=label_ids,
            average="macro",
            zero_division=0.0,
        )
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    return {
        "n": n,
        "f1_per_class": {
            _LABEL_ORDER[i].value: float(per_class[i]) for i in range(len(_LABEL_ORDER))
        },
        "f1_macro": macro,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def compute_dialect_breakdown(
    texts: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tagger: Callable[[str], Dialect] = classify_dialect,
) -> dict[str, object]:
    """Stratify F1 metrics by Gulf vs MSA dialect bucket.

    Returns the schema documented in the module docstring. The optional
    `tagger` argument exists for test injection — production code uses
    the default `classify_dialect` lexicon tagger.
    """
    dialects = np.array([tagger(t) for t in texts], dtype=object)
    gulf_mask = dialects == Dialect.GULF
    msa_mask = dialects == Dialect.MSA
    n_gulf = int(gulf_mask.sum())
    n_msa = int(msa_mask.sum())
    total = n_gulf + n_msa
    gulf_share = float(n_gulf / total) if total > 0 else 0.0
    return {
        "tagger": _TAGGER_NAME,
        "gulf": _bucket_block(y_true[gulf_mask], y_pred[gulf_mask]),
        "msa": _bucket_block(y_true[msa_mask], y_pred[msa_mask]),
        "gulf_share": gulf_share,
    }
