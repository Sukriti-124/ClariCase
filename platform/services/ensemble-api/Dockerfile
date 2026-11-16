# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir uv==0.4.*

WORKDIR /app

# Build context is the repo root (see docker-compose.yml).
# `claricase-models` is needed for src/ensemble/gate.py, which builds the gate
# features exactly as they were built in training.
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY platform/shared /app/platform/shared
COPY platform/services/ensemble-api/pyproject.toml /app/platform/services/ensemble-api/pyproject.toml
COPY platform/services/ensemble-api/src /app/platform/services/ensemble-api/src

RUN uv pip install --system --no-cache -e /app
RUN uv pip install --system --no-cache -e /app/platform/shared
RUN uv pip install --system --no-cache -e "/app/platform/services/ensemble-api[test]"

# Stack artifact (stacker + gate + thresholds + source prior).
# Produced by `python -m src.ensemble.train_stack`.
COPY docs/metrics/ensemble_stack/stack.joblib \
     /app/docs/metrics/ensemble_stack/stack.joblib

COPY platform/services/ensemble-api/tests /app/platform/services/ensemble-api/tests

ENV STACK_PATH=/app/docs/metrics/ensemble_stack/stack.joblib \
    ENSEMBLE_API_PORT=8000 \
    PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "ensemble_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
