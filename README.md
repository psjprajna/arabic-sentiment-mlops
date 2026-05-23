---
title: Arabic Sentiment LoRA Demo
emoji: ⓐ
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

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

| Env var                | Values                        | Default                          | Purpose                                                                              |
|------------------------|-------------------------------|----------------------------------|--------------------------------------------------------------------------------------|
| `SENTIMENT_BACKEND`    | `stub` / `catboost` / `lora`  | `stub`                           | Picks the active classifier                                                          |
| `MODEL_VERSION`        | string (e.g. `"10"`)          | latest non-archived              | Phase 7 — pins a specific MLflow Model Registry version; unset → resolve latest      |
| `MLFLOW_TRACKING_URI`  | URI                           | `sqlite:///mlflow.db`            | Phase 8 — SQLite-backed registry per ADR-0005 (supersedes ADR-0003's file:// default)|
| `LORA_MODEL_DIR`       | path                          | `models/arabert-lora-v1`         | Filesystem fallback when the registry path fails (empty `mlruns/`, MlflowException)  |
| `CATBOOST_MODEL_DIR`   | path                          | `models/catboost-baseline-v1`    | Filesystem fallback when the registry path fails                                     |
| `DRIFT_BUFFER_SIZE`    | int ≥ 1                       | `1000`                           | Drift ring-buffer capacity                                                           |
| `DRIFT_REPORTS_DIR`    | path                          | `reports/`                       | Source for `<backend>.json` PSI reference distributions                              |

```bash
# Serve the latest registered LoRA version (resolved from the registry at startup)
SENTIMENT_BACKEND=lora uv run uvicorn api.main:app --port 8000

# Pin to a specific registered version (recommended for production —
# the test-pollution registered v11 means DESC default doesn't serve v10)
SENTIMENT_BACKEND=catboost MODEL_VERSION=10 uv run uvicorn api.main:app --port 8000

# Force the filesystem fallback path (empty registry → falls back to *_MODEL_DIR)
MLFLOW_TRACKING_URI=sqlite:////tmp/empty.db \
  SENTIMENT_BACKEND=catboost \
  CATBOOST_MODEL_DIR=models/catboost-baseline-v1 \
  uv run uvicorn api.main:app --port 8000
```

Unknown backend values, missing fallback model directories, or a pinned
`MODEL_VERSION` that does not exist all fail fast at startup — the
container does not come up, so a misconfigured deploy is visible to the
orchestrator. The fallback path itself is opt-in by the absence of
registry data (empty / unreachable `mlruns/`); a populated registry is
always preferred and `/health.model_version` reports it.

### Retrain on drift

When `/metrics/drift` reports `drift_level: "significant"` on either
signal AND `observed_count >= 200`, trigger a retrain via the slice-9a CLI
(ADR-0006). The gate is pure-domain (`sentiment.domain.retrain_policy`);
the CLI is the only seam — no scheduler, no HTTP admin endpoint.

```bash
# Snapshot the live drift state
curl -s http://localhost:8000/metrics/drift > /tmp/drift.json

# Inspect the gate decision without retraining (recommended first)
uv run python -m sentiment.training.retrain_on_drift \
    --backend catboost --drift-report /tmp/drift.json --dry-run

# Fire the retrain (CatBoost: ~15 min wall; LoRA: ~60–90 min on MPS)
uv run python -m sentiment.training.retrain_on_drift \
    --backend catboost --drift-report /tmp/drift.json

# Or pipe the drift report via stdin
curl -s http://localhost:8000/metrics/drift | \
  uv run python -m sentiment.training.retrain_on_drift \
      --backend lora --drift-report -
```

A new model version is registered automatically when the gate fires.
**Restart `uvicorn` to start serving it** — Phase 7's registry-resolved
serving picks up the new version on cold-start (no in-process hot-reload).

Every invocation (fire or no-op) appends one line to
`reports/retrain-decisions.jsonl` (gitignored). Inspect with:

```bash
wc -l reports/retrain-decisions.jsonl
tail -1 reports/retrain-decisions.jsonl | jq .
```

Override the `observed_count` floor for emergency operator action:

```bash
uv run python -m sentiment.training.retrain_on_drift \
    --backend catboost --drift-report /tmp/drift.json --gate-min-observed 100
```

The CLI is synchronous and blocks the operator's shell. A scheduled
wrapper (slice 9b — cron / systemd / GitHub Action) is deferred until
operator workflow demands it; the same CLI is the wrapped entrypoint.

## Deploy to Hugging Face Spaces

Phase 10 ships the LoRA backend as a public demo on the Spaces Docker
SDK ($0 always-on, 16 GB RAM / 50 GB disk). The `Dockerfile` is the
contract; the YAML frontmatter at the top of this README is the Space
metadata (port `7860`, sdk `docker`). The CatBoost backend, retrain
CLI, and MLflow registry stay on the operator side — only the LoRA
serving path ships in the container.

Prerequisites — a Hugging Face account, the `huggingface_hub` CLI
(`hf`, formerly `huggingface-cli`, renamed in `huggingface_hub` 1.x),
and `git-lfs` for the Space repo push.

### One-time setup

```bash
brew install git-lfs && git lfs install
uv tool install huggingface_hub
hf auth login                 # paste a "Write" access token

# Create the public deploy bundle repo (holds LoRA weights, mlflow.db,
# mlruns/ subtree). Phase 10c (ADR-0008) — the runtime image pulls this
# at build time instead of baking from the local filesystem, so HF Spaces
# builds don't need the 525 MB bundle in the github main repo.
hf repos create arabic-sentiment-lora-deploy --repo-type model

# Create the Space (Docker SDK)
hf repos create arabic-sentiment-lora --repo-type space --space-sdk docker
```

### Per-release: upload the bundle to HF Hub

Run after every retrain or whenever the deployed artifacts should change:

```bash
hf upload PrajnaShetty/arabic-sentiment-lora-deploy mlflow.db mlflow.db
hf upload PrajnaShetty/arabic-sentiment-lora-deploy models/arabert-lora-v1 models/arabert-lora-v1
hf upload PrajnaShetty/arabic-sentiment-lora-deploy \
  mlruns/1/models/m-<logged-model-id> mlruns/1/models/m-<logged-model-id>
```

The `<logged-model-id>` is the `model_id` column from `mlflow.db`'s
`logged_models` table for the version you want to ship. The bundle on
HF Hub is the source-of-truth for what the next Space build will run.

### Push the code to the Space

```bash
git -C app remote add space https://huggingface.co/spaces/<your-handle>/arabic-sentiment-lora
git -C app push space main:main
```

The Space build runs `docker build` against the pushed git repo. The
cache-warm stage fetches the AraBERT base (~400 MB) and the deploy
bundle (~525 MB) from HF Hub. Expect 5–8 minutes for the first build;
subsequent pushes are incremental.

### Local smoke before pushing

The same contract the Space will hit (the local build also pulls from
HF Hub, so it tests the end-to-end resolution path):

```bash
docker build -t arabic-sentiment:phase10c app/
docker run --rm -p 7860:7860 arabic-sentiment:phase10c
curl -s localhost:7860/health | jq
# {
#   "status": "ok",
#   "model": "arabert-lora-v1",
#   "model_version": {
#     "name": "arabert-lora",
#     "version": "1",
#     "run_id": "94cccbd929a845dd9e47d46a5c3db759",
#     "source": "registry"
#   }
# }

curl -s -X POST localhost:7860/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"الفيلم كان رائعا"}' | jq
# {"text":"الفيلم كان رائعا","sentiment":"positive","confidence":0.99}
```

**Phase 10c — pull-on-build (ADR-0008).** The runtime image fetches the
deploy bundle from `PrajnaShetty/arabic-sentiment-lora-deploy` on HF Hub
via `huggingface_hub.snapshot_download` during the cache-warm stage. The
bundle layout (`models/`, `mlflow.db`, `mlruns/1/models/m-<id>/`) is
identical to what Phase 10b baked from the local filesystem. The
Phase-10b path-rewrite step still runs in the runtime stage (the bundle
on HF Hub carries the original dev-host paths in `mlflow.db`), as does
the container-only `sitecustomize.py` that maps MPS-tagged tensor
storage to CPU during `torch.load` (the LoRA cloudpickle was logged on
Apple Silicon).

**Phase 9b retrains break the public demo until rebuild + redeploy.**
The retrain CLI registers a new MLflow version on the operator's
machine; the Space still serves the baked LoRA adapter until the
operator rebuilds the image and pushes again. The scheduler slice
(9b) does not change this contract.

**Free CPU Spaces auto-sleep** after ~48 h of no traffic; the next
request takes ~30–60 s to wake the container. Acceptable for a
portfolio URL; pay for an "always-on" upgrade if continuous
availability matters.

## Status

**Phase 7 — Registry-resolved serving.** The MLflow Model Registry is
now the serving source-of-truth (ADR-0004). At startup,
`_build_classifier` calls `load_from_registry_or_fallback`, which
resolves a version (`MODEL_VERSION` env var when pinned, else the
latest non-archived via `MlflowClient.search_model_versions` ordered
`version_number DESC`), constructs `MLflowRegistryAdapter` around a
`mlflow.pyfunc.load_model("models:/<name>/<version>")` call, and
returns a `(SentimentClassifierPort, RegistryVersionInfo | None)`
tuple. On an explicit exception set
(`MlflowException`, `RestException`, `OSError`, `FileNotFoundError`,
empty-registry sentinel) the loader falls back to the existing
filesystem adapter from `*_MODEL_DIR` and the second element is
`None`. `GET /health.model_version` now reflects what was *actually
loaded* — `{name, version, run_id, source: "registry"}` on the
registry path, `null` on the filesystem-fallback or stub path. The
report-file extraction that previously fed `/health` is gone; the
loader is the single source-of-truth.

```bash
# Latest registered LoRA version (resolved at startup; LoRA registry is clean):
SENTIMENT_BACKEND=lora uv run uvicorn api.main:app --port 8000 &
curl -s http://localhost:8000/health | jq .model_version
# {"name":"arabert-lora","version":"1","run_id":"94cccbd9…","source":"registry"}

# CatBoost — pin v10 (the production retrain) explicitly; DESC default
# resolves to v11 due to a documented test-pollution path (see lessons.md
# 2026-05-22 | testing and ADR-0005 consequences):
SENTIMENT_BACKEND=catboost MODEL_VERSION=10 uv run uvicorn api.main:app --port 8000 &
curl -s http://localhost:8000/health | jq .model_version
# {"name":"catboost-baseline","version":"10","run_id":"609e3412…","source":"registry"}

# Forced fallback (registry unreachable → *_MODEL_DIR filesystem path):
MLFLOW_TRACKING_URI=sqlite:////tmp/empty.db SENTIMENT_BACKEND=catboost \
  uv run uvicorn api.main:app --port 8000 &
curl -s http://localhost:8000/health | jq .model_version
# null   — WARNING logged: "registry load failed … using filesystem fallback"
```

The hexagonal boundary is preserved: `api/main.py` no longer imports
concrete adapters; the only serving-side `mlflow` import lives in
`sentiment/adapters/mlflow_registry_classifier.py`. `test_fitness.py`
still passes. **220 tests green** (post-Phase-8). SQLite-backed MLflow
metadata has retired the Phase-7 `file://` deprecation warning — see
Phase 8 below.

**Phase 8 — SQLite registry migration.** Default `MLFLOW_TRACKING_URI`
flips from `file:./mlruns` to `sqlite:///mlflow.db` (ADR-0005, supersedes
ADR-0003's deferred position). Both backends retrained against the new
backend; the Phase-4 `confidence_histogram` and Phase-5 `dialect_breakdown`
fields are carried forward through the retrain. CatBoost macro-F1 stayed
at **0.7648** (deterministic backend, byte-identical); LoRA macro-F1
landed at **0.8377** (within the ±0.005 MPS noise band, above the
Phase-2 0.83 acceptance bar). The two `FutureWarning`s for the
filesystem tracking/registry backend are gone (0 of 0 across the test
suite). `app/mlruns/` (575 MB) and `app/reports/` were archived to
`*-legacy.bak/` before the cutover; both stay on the local filesystem
as the rollback path until container deploy ships.

Known test-pollution path: `tests/test_mlflow_logging.py::_build_catboost_fixture`
calls `run_baseline(...)` without passing `mlflow_tracking_uri=`, so
test-time fixture builds register `catboost-baseline` versions against
the production `app/mlflow.db`. Under the file backend this wrote into
the gitignored `app/mlruns/` (invisible); under SQLite it accumulates
in a single file. `catboost-baseline` v1–v9 are pre-retrain pollution;
v10 is the Step-2 production retrain; v11 is a diagnostic registration
created during investigation. DESC resolution defaults to v11 — set
`MODEL_VERSION=10` for production-grade serving. Full root cause in
workspace `tasks/lessons.md` 2026-05-22 | testing.

**Phase 6 — MLflow Model Registry versioning.** Every retrain of either
backend registers a new version into the local Model Registry
(`file:./mlruns`, `models/<name>/version-N/`) under one polymorphic
`SentimentPyfunc` wrapper that dispatches on `backend_type` at load
time. Each backend's report JSON carries a top-level `registry` block —
`{name, version, run_id, model_uri, registered_at}` — and the registry
state is the source Phase 7 reads from. Current versions on the
development tree:

| Backend            | Production Version | Run ID prefix | Notes                              |
|--------------------|-------------------:|---------------|------------------------------------|
| `catboost-baseline`| `"10"` (pin via `MODEL_VERSION=10`) | `609e3412…` | v1–v9 fixture pollution + v11 diagnostic; v10 is the Step-2 retrain — see Phase 8 |
| `arabert-lora`     | `"1"`              | `94cccbd9…`   | Clean registry; DESC resolves correctly |

```bash
# Inspect via UI:
uv run mlflow ui --backend-store-uri file:./mlruns --port 5001
```

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
- ~~**Phase 6**~~ — MLflow Model Registry versioning; `/health` surfaces `model_version`; carry-forward of Phase-4/5 fields across retrains. ✅
- ~~**Phase 7**~~ — Registry-resolved serving (load via `models:/<name>/<version>`); `MODEL_VERSION` env var; filesystem fallback for offline dev; `/health.model_version` reflects what is loaded. ✅
- ~~**Phase 8**~~ — SQLite migration off the deprecated `file://` registry backend (`sqlite:///mlflow.db` default; ADR-0005). ✅
- ~~**Phase 9**~~ — Auto-retraining trigger (slice 9a: manual CLI gate-and-invoke on `/metrics/drift` signals; ADR-0006). ✅
- **Phase 9b** — Background scheduler wrapping the same CLI (deferred until operator workflow demands it).
- ~~**Phase 10**~~ — Container deploy on Hugging Face Spaces (Docker SDK, LoRA-only, ADR-0007). Local image smoke-tested at 3 GB; `/health.model_version=null` accepted regression on the filesystem-fallback path. ✅
- **Phase 11** — Streamlit demo + ops UI (Try-it + Behind-the-scenes tabs).
