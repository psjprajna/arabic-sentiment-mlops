# syntax=docker/dockerfile:1.7
# Multi-stage build for the LoRA-only FastAPI serving image.
# Target: Hugging Face Spaces (Docker SDK), public port 7860, app user UID 1000.

ARG PYTHON_VERSION=3.12-slim
ARG BASE_MODEL=aubmindlab/bert-base-arabertv2
ARG DEPLOY_REPO=PrajnaShetty/arabic-sentiment-lora-deploy


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
# Pre-downloads the AraBERT base model AND the LoRA+registry deploy bundle.
# Both downloads live in this stage so the runtime image never fetches
# anything at startup (TRANSFORMERS_OFFLINE=1, no HF Hub call on first
# /predict). Phase 10c switched from "bake at build" (artifacts in build
# context) to "pull on build" (artifacts in a public HF Hub model repo)
# so the HF Spaces git clone can complete without LFS-pushing 525 MB
# through the github main repo.
FROM python:${PYTHON_VERSION} AS cache-warm

ARG BASE_MODEL
ARG DEPLOY_REPO
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_OFFLINE=0

# Pre-download AraBERT base (~400 MB) so the runtime never fetches it.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
m='${BASE_MODEL}'; \
AutoTokenizer.from_pretrained(m); \
AutoModelForSequenceClassification.from_pretrained(m, num_labels=3)"

# Pull the full deploy bundle (models/, mlflow.db, mlruns/) from HF Hub
# into /opt/deploy. The bundle layout mirrors what was previously a local
# COPY in Phase 10b — same paths, same files. No HF token needed (public
# repo). snapshot_download writes the actual files (not symlinks) when
# local_dir is set, so /opt/deploy is self-contained.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${DEPLOY_REPO}', repo_type='model', local_dir='/opt/deploy')"


# ---------- Stage 3: runtime ----------
FROM python:${PYTHON_VERSION} AS runtime

# HF Spaces runs containers as UID 1000 by convention.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=cache-warm --chown=app:app /opt/hf-cache /home/app/.cache/huggingface

# Bundle from HF Hub: drops models/, mlflow.db, mlruns/ into /app/<...>.
COPY --from=cache-warm --chown=app:app /opt/deploy /app

COPY --chown=app:app src/ /app/src/

# Phase 10b: path rewrite (still required). `logged_models.artifact_location`
# and `model_versions.storage_location` in the baked mlflow.db carry the
# absolute build-host path from where the bundle was originally logged on
# the dev machine. MLflow 3.x reads `model_versions.storage_location` for
# `pyfunc.load_model("models:/<name>/<v>")` resolution, so a missed column
# crashes with `MlflowException: No such artifact: ''`. Sweep all six
# path-bearing columns with bound SQL params, then assert no residue
# remains in any of the four critical columns.
RUN /opt/venv/bin/python <<'PYEOF'
import sqlite3

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

# Phase 10b: torch compatibility shim. The LoRA cloudpickle in
# python_model.pkl was logged on Apple Silicon MPS; the CPU-only Linux
# torch wheel cannot create MPS storages and crashes deserialization
# with `Storage device not recognized: mps`. sitecustomize.py runs at
# every Python startup, registers a torch package handler at priority 5
# (before the default MPS handler at 21), and maps MPS-tagged storage to
# CPU. Container-only; dev-machine venv is untouched.
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
