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

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    SENTIMENT_BACKEND=lora \
    LORA_MODEL_DIR=/app/models/arabert-lora-v1 \
    MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow-empty.db \
    PORT=7860

USER app
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
