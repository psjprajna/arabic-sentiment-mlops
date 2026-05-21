# Phase 1 — CatBoost + TF-IDF baseline on HARD

A CPU-only baseline trained on the **HARD Arabic Reviews** corpus (Booking.com hotel
reviews, 5★ scale collapsed to 3-class positive/neutral/negative). This is the bar that
the Phase 2 AraBERT-LoRA work has to beat to justify itself.

## Headline

| Metric | Value |
|---|---|
| F1 macro (test) | **0.765** |
| F1 positive | 0.901 |
| F1 negative | 0.717 |
| F1 neutral | 0.676 |

Acceptance gate (D3 in spec 001): **F1 macro ≥ 0.65**. Cleared by ~0.11.

## Dataset

Source: [`elnagara/HARD-Arabic-Dataset`](https://github.com/elnagara/HARD-Arabic-Dataset),
the *unbalanced* split (the balanced split has no rating=3 examples and so cannot exercise
the neutral class).

5★ → 3-class mapping:

- 1, 2 → negative
- 3 → neutral
- 4, 5 → positive

Stratified 80/10/10 split, fixed seed 42.

| Split | Size |
|---|---|
| train | 327,649 |
| dev | 40,956 |
| test | 40,957 |

## Confusion matrix (test split)

Rows: true label. Columns: predicted label. Order: `[positive, negative, neutral]`.

|             | pred pos | pred neg | pred neu |
|-------------|---------:|---------:|---------:|
| **true pos** |  24,142 |    1,459 |    2,038 |
| **true neg** |     382 |    4,476 |      427 |
| **true neu** |   1,407 |    1,266 |    5,360 |

## Pipeline

- **Normalization (adapter-internal):** strip URLs, `@mentions`, `#` from hashtags, collapse
  whitespace. **Diacritics (tashkeel) preserved** — the domain rule treats them as
  meaningful, so the vectorizer learns them as part of the token. The original text is
  returned in `SentimentResult.text` unchanged.
- **Features:** TF-IDF, word analyzer, `(1, 2)`-grams, `max_features=20_000`, `min_df=3`,
  `sublinear_tf=True`.
- **Classifier:** CatBoost, 200 iterations, depth 6, LR 0.1, `loss_function="MultiClass"`,
  `auto_class_weights="Balanced"` (HARD is positive-skewed; without this, the model
  converges to a constant-positive predictor in one iteration). Early stopping wait 20.

## Demo — curated Arabic cases

Run `uv run python -m sentiment.demo` after training. The default mode iterates 8 fixed
cases covering MSA positive/negative/neutral, Gulf dialect, negation, and OOV inputs:

| # | input | predicted | confidence |
|---|-------|-----------|------------|
| 1 | الفندق ممتاز والخدمة رائعة جدا | positive | 0.94 |
| 2 | الغرفة كانت قذرة والموظفون غير محترمين | negative | 0.48 |
| 3 | الفندق عادي، لا أكثر ولا أقل | negative | 0.51 |
| 4 | الفندق وايد زين وأهل البحرين يحبونه | positive | 0.37 |
| 5 | ما عجبني المكان أبداً | positive | 0.37 |
| 6 | الخدمة ليست سيئة | negative | 0.54 |
| 7 | 🌟🌟🌟🌟🌟 | positive | 0.37 |
| 8 | Great hotel, excellent service | positive | 0.37 |

Behavior notes — these are honest failures the baseline is not equipped to handle:

- **Case 3** ("ordinary, no more no less"): the dataset has very few subtle neutral
  reviews; the model leans negative when polarity words are absent.
- **Case 5** ("I didn't like the place at all"): Gulf dialect negation (`ما + verb`). TF-IDF
  bigrams cannot reason about negation scope; AraBERT-LoRA in Phase 2 should fix this.
- **Case 6** ("the service isn't bad"): another negation case — TF-IDF sees the token "bad"
  and votes negative. Same root cause as case 5.
- **Cases 7–8:** emoji-only and English. Both produce near-uniform low confidence
  (~0.37 = `1/n_classes`), which is the desired OOV behavior — a downstream caller can
  threshold on it.

## Reproduce

```bash
# from app/
uv pip install -e ".[ml,dev]"
# Fetches HARD unbalanced reviews from the upstream GitHub repo if absent.
uv run python -m sentiment.training.baseline
uv run python -m sentiment.demo
```

Outputs land in `models/catboost-baseline-v1/` and `reports/catboost-baseline-v1.json` —
both gitignored.

## What's deferred

This phase deliberately does not:

- Wire the adapter into `/predict` — the API still returns the stub. That swap is Phase 3.
- Track to MLflow — the JSON report is the artifact. MLflow lands in Phase 2 alongside the
  LoRA work where the comparison story matters.
- Tune hyperparameters or train on Gulf-only data — that's Phase 4.
