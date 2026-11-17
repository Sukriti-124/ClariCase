# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

RUN pip install --no-cache-dir uv==0.4.*
WORKDIR /app

# CPU-only torch, own layer (see embed-api).
RUN uv pip install --system --no-cache torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY platform/shared /app/platform/shared
COPY platform/services/agentic-api/pyproject.toml /app/platform/services/agentic-api/pyproject.toml
COPY platform/services/agentic-api/src /app/platform/services/agentic-api/src

RUN uv pip install --system --no-cache -e /app
RUN uv pip install --system --no-cache -e /app/platform/shared
RUN uv pip install --system --no-cache -e "/app/platform/services/agentic-api[encoder,test]"

# Same encoder as embed-api; the example pool was embedded with it.
ARG ENCODER_NAME=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${ENCODER_NAME}')"

# improved_model_2's bundle only names the encoder; the pool and the LLM
# cache are mounted (see docker-compose.yml).
COPY docs/metrics/improved_model_2/model.joblib /app/docs/metrics/improved_model_2/model.joblib
COPY data/processed/team_mapping.csv  /app/data/processed/team_mapping.csv
COPY data/processed/issue_mapping.csv /app/data/processed/issue_mapping.csv
COPY platform/services/agentic-api/tests /app/platform/services/agentic-api/tests

ENV AGENTIC_API_PORT=8000 \
    HF_HUB_OFFLINE=1 \
    PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "agentic_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
