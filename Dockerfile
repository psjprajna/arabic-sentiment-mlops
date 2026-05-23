# syntax=docker/dockerfile:1.7
# Multi-stage build for the LoRA-only FastAPI serving image.
# Target: Hugging Face Spaces (Docker SDK), public port 7860, app user UID 1000.

ARG PYTHON_VERSION=3.12-slim
ARG BASE_MODEL=aubmindlab/bert-base-arabertv2


# ---------- Stage 1: builder ----------
# Resolves the ml-runtime extra into a self-contained venv we copy forward.
# uv version is pinned in the COPY --from= ref (BuildKit does not substitute
# ARGs into --from references reliably).
FROM python:${PYTHON_VERSION} AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build
COPY pyproject.toml uv.lock ./

RUN uv sync --extra ml-runtime --frozen --no-dev --no-install-project


# ---------- Stage 2: cache-warm ----------
# Pre-downloads the AraBERT base model so the runtime image never fetches
# ~400 MB on first /predict (Phase 7 fallback path loads the base via HF).
FROM python:${PYTHON_VERSION} AS cache-warm

ARG BASE_MODEL
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_OFFLINE=0

RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
m='${BASE_MODEL}'; \
AutoTokenizer.from_pretrained(m); \
AutoModelForSequenceClassification.from_pretrained(m, num_labels=3)"


# ---------- Stage 3: runtime ----------
FROM python:${PYTHON_VERSION} AS runtime

# HF Spaces runs containers as UID 1000 by convention.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=cache-warm --chown=app:app /opt/hf-cache /home/app/.cache/huggingface

COPY --chown=app:app src/ /app/src/
COPY --chown=app:app models/arabert-lora-v1/ /app/models/arabert-lora-v1/

# Phase 10b: bake the MLflow registry so /health surfaces model_version.
# mlflow.db holds the model_versions metadata; the logged-model subtree
# under mlruns/1/models/m-<id>/artifacts/ holds the pyfunc bundle (MLmodel
# manifest + nested LoRA adapter). Resolution path:
#   models:/arabert-lora/1 -> model_versions.source (models:/m-<id>)
#   -> logged_models.artifact_location (filesystem path)
# logged_models.artifact_location is absolute on the build host, so we
# rewrite it to /app/<...> in the image. SQL params are bound (?), not
# string-concatenated, so dev-machine path characters (incl. spaces) are
# safe. The models/arabert-lora-v1/ COPY above stays as the filesystem
# fallback if registry resolution ever fails.
COPY --chown=app:app mlflow.db /app/mlflow.db
COPY --chown=app:app mlruns/1/models/m-f9cac40424504a5a91d9449f38f5bd7c/ \
                   /app/mlruns/1/models/m-f9cac40424504a5a91d9449f38f5bd7c/

RUN /opt/venv/bin/python <<'PYEOF'
import sqlite3

# MLflow 3.x scatters absolute build-host paths across several columns; ALL
# of them must be rewritten or the registry resolver gets handed a path that
# doesn't exist inside the container and falls back to an empty artifact URI
# ("MlflowException: No such artifact: ''"). Critical for resolution:
#   - logged_models.artifact_location   (logged-model store)
#   - model_versions.storage_location   (registry-side mirror; what
#                                        pyfunc.load_model("models:/<name>/<v>")
#                                        actually reads in MLflow 3.x)
#   - runs.artifact_uri                 (run-side mirror; belt-and-braces)
#   - experiments.artifact_location     (experiment-default; belt-and-braces)
# Tag columns also carry the venv pytest path; harmless but swept for hygiene.
conn = sqlite3.connect("/app/mlflow.db")
cur = conn.cursor()
row = cur.execute("SELECT artifact_location FROM logged_models LIMIT 1").fetchone()
if not row:
    raise SystemExit("logged_models is empty; cannot bake registry")
old_loc = row[0]
marker = "/mlruns/"
if marker not in old_loc:
    raise SystemExit(f"unexpected artifact_location shape: {old_loc!r}")
old_prefix = old_loc.split(marker, 1)[0]
if old_prefix == "/app":
    print(f"mlflow.db already rewritten ({old_loc}); skipping")
else:
    rewrites = (
        ("logged_models", "artifact_location"),
        ("model_versions", "storage_location"),
        ("runs", "artifact_uri"),
        ("experiments", "artifact_location"),
        ("tags", "value"),
        ("logged_model_tags", "tag_value"),
    )
    for table, col in rewrites:
        cur.execute(
            f"UPDATE {table} SET {col} = REPLACE({col}, ?, ?) WHERE {col} LIKE ?",
            (old_prefix, "/app", f"%{old_prefix}%"),
        )
    conn.commit()
# Confirm no residue of the old prefix remains in any critical column.
for table, col in (
    ("logged_models", "artifact_location"),
    ("model_versions", "storage_location"),
    ("runs", "artifact_uri"),
    ("experiments", "artifact_location"),
):
    leftover = cur.execute(
        f"SELECT {col} FROM {table} WHERE {col} LIKE ? LIMIT 1",
        (f"{old_prefix}%",),
    ).fetchone()
    if leftover is not None:
        raise SystemExit(f"path rewrite missed {table}.{col}: {leftover[0]!r}")
post = cur.execute(
    "SELECT artifact_location FROM logged_models WHERE model_id='m-f9cac40424504a5a91d9449f38f5bd7c'"
).fetchone()
sloc = cur.execute(
    "SELECT storage_location FROM model_versions WHERE name='arabert-lora' AND version=1"
).fetchone()
print(f"arabert-lora v1 logged_models.artifact_location -> {post[0] if post else 'MISSING'}")
print(f"arabert-lora v1 model_versions.storage_location -> {sloc[0] if sloc else 'MISSING'}")
conn.close()
PYEOF

# Phase 10b: torch compatibility shim. The LoRA artifact baked into this
# image was trained on Apple Silicon MPS; cloudpickle stored tensors carry
# `location="mps"`. The CPU-only Linux torch wheel has no MPS backend, so
# mlflow.pyfunc.load_model("models:/arabert-lora/1") crashes on storage
# deserialization with `Storage device not recognized: mps` and the
# container fails to start. sitecustomize.py runs at every Python startup
# (loaded by site.py before any user code), registers a torch package
# handler at priority 5 — checked before the default MPS handler at
# priority 30 — and returns the already-allocated CPU storage when the
# location starts with "mps". The file lives only in this image's
# /opt/venv; the dev-machine venv is unaffected.
RUN cat > /opt/venv/lib/python3.12/site-packages/sitecustomize.py <<'PYEOF'
"""Container-only torch shim: map MPS-tagged storage to CPU at load."""

import torch.serialization

torch.serialization.register_package(
    5,
    lambda obj: None,
    lambda storage, location: storage if location.startswith("mps") else None,
)
PYEOF

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    SENTIMENT_BACKEND=lora \
    LORA_MODEL_DIR=/app/models/arabert-lora-v1 \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db \
    PORT=7860

USER app
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
