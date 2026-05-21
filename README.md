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
  breakdown in Phase 4), rather than reporting a single dataset-wide accuracy.
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

- `GET /health` → `{"status": "ok", "model": "stub"}`
- `POST /predict` with `{"text": "..."}` → `{"text": "...", "sentiment": "positive|negative|neutral", "confidence": float}`

Empty or whitespace-only text returns HTTP 422 (Pydantic validation).

## Status

**Phase 0 — walking skeleton.** The current adapter is a stub that always returns `neutral`
with confidence `0.0`. This phase exists to lock in the architecture (domain boundary,
adapter wiring, API contract, tests) before adding a real model.

Roadmap:

- **Phase 1** — CatBoost + TF-IDF baseline on the HARD Arabic Sentiment Analysis dataset;
  per-class F1 evaluation.
- **Phase 2** — AraBERT LoRA fine-tuning; MLflow experiment tracking and model registry.
- **Phase 3** — Swap the stub for the real model in the inference path.
- **Phase 4** — PSI drift monitoring.
- **Phase 5** — Azure Container Apps deployment (UAE North).

## Author

Prajna Shetty.
