# Arabic Sentiment MLOps

Full MLOps pipeline for Arabic sentiment analysis (positive / negative / neutral) on
UAE-relevant text. Portfolio proof-of-work for Applied AI / Gen AI engineer roles in the UAE.

## Why this project

End-to-end demonstration of the modern MLOps lifecycle on a real, linguistically interesting
problem:

- Data loading and Arabic-specific preprocessing (MSA and Gulf/Emirati dialect).
- Two interchangeable model backends — a fast **CatBoost + TF-IDF** baseline and an
  **AraBERT LoRA** fine-tuned classifier — wired behind the same port so they can be swapped
  without touching API or domain code.
- Experiment tracking with **MLflow** (metrics, artifacts, model registry).
- Per-class F1 evaluation with confusion matrices; accuracy alone is not reported.
- **FastAPI** serving with input validation and automatic OpenAPI docs.
- Roadmap to **PSI drift monitoring** and **Azure Container Apps** deployment in UAE North.

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
