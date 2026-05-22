# Arabic Sentiment Analysis

A FastAPI service that classifies Arabic text — Modern Standard Arabic (MSA) and Gulf /
Emirati dialect — as **positive**, **negative**, or **neutral**, with two interchangeable
model backends sitting behind a single port: a fast **CatBoost + TF-IDF** baseline and an
**AraBERT** classifier adapted with **LoRA** for parameter-efficient fine-tuning on
free-tier hardware. The architecture, evaluation, and serving path are built around the
constraints that make Arabic sentiment genuinely hard — not the easy MSA-only case the
literature usually reports.

## What makes this non-trivial

- **Diglossia.** UAE-relevant text constantly code-switches between MSA (formal,
  news-style) and Gulf dialect (colloquial, often transliterated, missing standard
  spelling). Most pretrained sentiment models are MSA-only and drop several F1 points on
  Gulf input. This project measures that gap explicitly (per-class F1 with a dialect
  breakdown — see Phase 5), rather than reporting a single dataset-wide accuracy.
- **Diacritics (tashkeel) carry meaning.** Stripping them is the easy default — and it
  silently flips sentiment on edge cases (e.g. negation particles). The domain layer
  preserves the original text; normalization is an adapter concern, so the choice is
  auditable rather than buried inside a preprocessing script.
- **No GPU budget.** Full fine-tuning of AraBERT is off the table; LoRA cuts the trainable
  parameter count to ~1% of the base model so the project trains on free-tier GPUs.
  CatBoost is the no-GPU baseline that has to be beaten to justify the LoRA work.
- **Swappable backends, mechanically enforced.** `SentimentClassifierPort` is the only
  thing the API depends on. The stub (Phase 0), CatBoost adapter (Phase 1), and
  AraBERT-LoRA adapter (Phase 2) implement it without the domain or API ever importing
  CatBoost, Torch, Transformers, FastAPI, or MLflow. A fitness test
  (`tests/test_fitness.py`) parses the domain modules and fails the build if any of those
  imports sneak in — boundary as code, not as a wiki page.
- **Evaluation rigor.** Accuracy alone is reported nowhere. Every experiment logs per-class
  F1 + confusion matrix to MLflow, with the model registry tracking which adapter and
  dataset version produced each metric.

## Out of scope (v1)

- Full fine-tuning of AraBERT — LoRA only.
- A frontend — the contract is the JSON API.
- Authentication middleware — lands alongside the Azure deployment in Phase 5.
- A CI/CD pipeline — local `uv run pytest` is the merge gate until then.

## Stack

Python 3.12 · uv · FastAPI · Pydantic v2 · CatBoost · AraBERT / CAMeL-BERT (HuggingFace
Transformers) · MLflow · pytest · ruff · Docker · Azure ML + Container Apps (UAE North).

## Architecture

Hexagonal (Ports & Adapters). The domain layer (`src/sentiment/domain/`) is pure Python with
no infrastructure imports — no FastAPI, no MLflow, no Transformers, no Torch, no CatBoost.
External concerns live in `src/sentiment/adapters/`. The boundary is enforced mechanically by
`tests/test_fitness.py`, which fails the build if any forbidden import sneaks into the domain.

```
src/
├── sentiment/
│   ├── domain/            pure Python — Sentiment enum, SentimentResult, classifier port
│   └── adapters/          implementations: StubClassifier (P0) → CatBoost (P1) → AraBERT (P2)
└── api/                   FastAPI composition root — wires adapter into the port
tests/
├── test_fitness.py        ADR-0002 boundary check (runs in <1s, no network)
└── test_api.py            FastAPI smoke tests via TestClient
```

## Quick start

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run ruff check src/ tests/
uv run pytest -v
uv run uvicorn api.main:app --reload
```

## API

- `GET /health` → `{"status": "ok", "model": "<backend-id>"}`
- `POST /predict` with `{"text": "..."}` → `{"text": "...", "sentiment": "positive|negative|neutral", "confidence": float}`
- `GET /metrics/drift` → two-signal Population Stability Index over the
  most-recent served predictions (predicted-class distribution +
  confidence-bucket distribution), with `drift_level` ∈ `{stable, moderate,
  significant}` per industry-standard PSI bands. Returns HTTP 503 with
  `{"detail": "drift monitoring disabled for stub backend"}` when running
  the stub (no reference distribution to compare against).

Empty or whitespace-only text returns HTTP 422 (Pydantic validation). If the
adapter raises an unexpected exception during inference, the response is
HTTP 500 with body `{"detail": "internal inference error"}` — the input text
and the stack trace stay server-side (logged via `logger.exception`).

Drift recording is best-effort: a raise inside the monitor's `record(...)`
is logged at WARNING and swallowed, so `/predict` returns 200 with an intact
body even if the drift path is broken. The buffer holds only
`(Sentiment, confidence_bucket_label)` tuples — no input text by type
signature.

### Backend selection (env vars)

| Env var               | Values                        | Default                          | `/health` `model`         |
|-----------------------|-------------------------------|----------------------------------|---------------------------|
| `SENTIMENT_BACKEND`   | `stub` / `catboost` / `lora`  | `stub`                           | `stub` / `catboost-baseline-v1` / `arabert-lora-v1` |
| `LORA_MODEL_DIR`      | path                          | `models/arabert-lora-v1`         | (lora only)               |
| `CATBOOST_MODEL_DIR`  | path                          | `models/catboost-baseline-v1`    | (catboost only)           |
| `DRIFT_BUFFER_SIZE`   | int ≥ 1                       | `1000`                           | drift ring-buffer capacity |
| `DRIFT_REPORTS_DIR`   | path                          | `reports/`                       | source for `<backend>.json` reference distributions |

```bash
# Serve the LoRA fine-tune (requires models/arabert-lora-v1/ on disk)
SENTIMENT_BACKEND=lora uv run uvicorn api.main:app --port 8000

# Serve the CatBoost baseline
SENTIMENT_BACKEND=catboost uv run uvicorn api.main:app --port 8000
```

Unknown backend values or missing model directories fail fast at startup —
the container does not come up, so a misconfigured deploy is visible to
the orchestrator (no silent fallback to stub).

## Status

**Phase 5 — Gulf-vs-MSA dialect breakdown.** Every retrained backend
report now carries a `dialect_breakdown` block (Gulf and MSA buckets,
per-class F1 + confusion matrix + bucket size) under a `tagger:
"lexicon-v1"` methodology marker. Dialect labels are produced by a
curated 27-marker Gulf lexicon in `sentiment/domain/dialect.py` — pure
Python, no new dependencies, zero infra imports. The lexicon is noisy
by design (HARD has no ground-truth dialect labels and Twitter-trained
classifiers don't transfer to formal hotel-review text); the `tagger`
marker makes the methodology explicit so a future model-based swap is
a clean version bump.

Real numbers on the full HARD test split (~40,957 rows; **n_gulf=420,
gulf_share=1.0%**):

| Backend  | Gulf macro F1 | MSA macro F1 |    Δ    |
|----------|--------------:|-------------:|--------:|
| CatBoost |    **0.702**  |   **0.765**  | +0.063  |
| LoRA     |    **0.767**  |   **0.838**  | +0.072  |

The diglossia gap **widens** with the better backend — LoRA fine-tuning
improves MSA macro by +0.073 but Gulf macro by only +0.064. Neutral is
the hardest class in both buckets on both backends. This is the gap the
README has been promising since Phase 0; it's now in the report JSONs as
hard numbers rather than aspirational copy. **186 tests green** on
the branch; fitness still clean; `f1_macro` and `confidence_histogram`
preserved across the retrain.

**Phase 4 — PSI drift monitoring.** `GET /metrics/drift` returns
Population Stability Index for two independent signals computed off one
in-memory ring buffer of `(Sentiment, bucket_label)` tuples: the
predicted-class distribution vs. the column sums of the existing
`reports/<backend>.json` confusion matrix, and a 3-band confidence-bucket
distribution (`low [0,0.6) / medium [0.6,0.8) / high [0.8,1.0]`) vs. a
`confidence_histogram` field backfilled by
`python -m sentiment.training.build_confidence_reference --backend <…>`.
Per-signal graceful degradation: a missing reference field
(`confusion_matrix` or `confidence_histogram`) only blinds that signal —
the other one keeps reporting. The stub backend short-circuits the
endpoint to 503 since it has no reference to compare against. Recording
adds **<1 ms median** to `/predict` (measurement noise dominates).
**88 tests green** on the branch, fitness still clean.

**Phase 3 — `/predict` serves the real model.** `api/main.py` is a
composition root only: at FastAPI lifespan startup it reads
`SENTIMENT_BACKEND` and constructs the chosen adapter once
(`stub` / `CatBoostAdapter` / `AraBERTLoRAAdapter`), storing it on
`app.state.classifier`. The endpoint receives it through
`Depends(get_classifier)`; tests override the seam with
`app.dependency_overrides`. Default remains `stub` so offline `uv run pytest`
stays fast (no model weights touched).

**Phase 2 — AraBERT LoRA fine-tune.** F1 macro **0.8344** on the HARD test
split, +0.070 over the CatBoost baseline (+9.1% relative). Per-class:
positive 0.945 / negative 0.826 / neutral 0.733. Metrics + 3×3 confusion
matrix + a curated demo are in [`reports/`](reports/).

**Phase 1 — CatBoost + TF-IDF baseline.** Per-class F1: positive 0.901 /
negative 0.717 / neutral 0.676, **F1 macro 0.765**. See
[`docs/phase-01-baseline-results.md`](docs/phase-01-baseline-results.md)
for the results card.

Real-model `/predict` latency on dev hardware (Apple-silicon MPS, LoRA
backend, post-warmup): median **~16 ms**, p95 **~17 ms** over 20
single-text POSTs.

Roadmap:

- ~~**Phase 0**~~ — walking skeleton (stub adapter, fitness test, FastAPI shape). ✅
- ~~**Phase 1**~~ — CatBoost + TF-IDF baseline on HARD; per-class F1. ✅
- ~~**Phase 2**~~ — AraBERT LoRA fine-tuning; MLflow experiment tracking. ✅
- ~~**Phase 3**~~ — `/predict` dispatches on `SENTIMENT_BACKEND` to the real adapter. ✅
- ~~**Phase 4**~~ — PSI drift monitoring on `/metrics/drift` (predicted-class + confidence-bucket). ✅
- ~~**Phase 5**~~ — Gulf vs. MSA dialect breakdown (lexicon-v1 tagger, both backends, real numbers in `reports/`). ✅
- **Phase 6** — Model registry versioning; MLflow surfacing of drift + dialect metrics; Azure Container Apps deployment (UAE North).

## Author

Prajna Shetty.
