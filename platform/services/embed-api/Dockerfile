# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

RUN pip install --no-cache-dir uv==0.4.*

WORKDIR /app

# Build context is the repo root (see docker-compose.yml).

# 1. CPU-only torch first: the default wheel pulls ~2 GB of CUDA libraries the
#    container never uses. Own layer so it is cached across code changes.
RUN uv pip install --system --no-cache torch \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Root pyproject + src tree (makes `claricase-models` importable), the
#    shared platform library, and this service.
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY platform/shared /app/platform/shared
COPY platform/services/embed-api/pyproject.toml /app/platform/services/embed-api/pyproject.toml
COPY platform/services/embed-api/src /app/platform/services/embed-api/src

RUN uv pip install --system --no-cache -e /app
RUN uv pip install --system --no-cache -e /app/platform/shared
RUN uv pip install --system --no-cache -e "/app/platform/services/embed-api[encoder,test]"

# 3. Bake the sentence encoder into the image so the container never downloads
#    at startup. Must be the encoder the model was trained with (it is recorded
#    in model.joblib as `encoder_name`); a mismatch fails fast at startup
#    because the hub is offline at runtime.
ARG ENCODER_NAME=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${ENCODER_NAME}')"

# 4. Model artifact. Produced by `python -m src.improved_model_2.train`.
COPY docs/metrics/improved_model_2/model.joblib \
     /app/docs/metrics/improved_model_2/model.joblib

# 5. Tests, for in-container pytest.
COPY platform/services/embed-api/tests /app/platform/services/embed-api/tests

ENV EMBED_MODEL_PATH=/app/docs/metrics/improved_model_2/model.joblib \
    EMBED_API_PORT=8000 \
    HF_HUB_OFFLINE=1 \
    PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "embed_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
