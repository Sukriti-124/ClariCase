# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install uv (fast installer)
RUN pip install --no-cache-dir uv==0.4.*

WORKDIR /app

# Build context is the repo root (see docker-compose.yml).
# Copy only the pieces we need, in dependency order for better layer caching.

# 1. Root pyproject + src tree — makes `claricase-models` importable.
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

# 2. Shared platform library.
COPY platform/shared /app/platform/shared

# 3. svm-api package.
COPY platform/services/svm-api/pyproject.toml /app/platform/services/svm-api/pyproject.toml
COPY platform/services/svm-api/src /app/platform/services/svm-api/src

# 4. Install the three packages into the system interpreter. Separate steps
#    so a failure in one gives a clear error and layer caching is predictable.
#    DEVIATION FROM PLAN: svm-api is installed with [test] extra so pytest+httpx
#    are available for running Task 7/8 tests inside the container. For Phase 1
#    this is acceptable — splitting prod vs test images is a Phase 2 concern.
RUN uv pip install --system --no-cache -e /app
RUN uv pip install --system --no-cache -e /app/platform/shared
RUN uv pip install --system --no-cache -e "/app/platform/services/svm-api[test]"

# 5. Model artifact + reference data.
COPY docs/metrics/improved_model_1/model.joblib \
     /app/docs/metrics/improved_model_1/model.joblib
# baseline_model_2 (TF-IDF + Naive Bayes) rides along as a second opinion for
# the ensemble. 16 MB; same scikit-learn pin.
COPY docs/metrics/baseline_model_2/model.joblib \
     /app/docs/metrics/baseline_model_2/model.joblib
COPY data/processed/team_mapping.csv  /app/data/processed/team_mapping.csv
COPY data/processed/issue_mapping.csv /app/data/processed/issue_mapping.csv

# Preprocessing module referenced by the pickled vectorizer at joblib.load time.
# `data/preparation/` is a namespace package (no __init__.py needed on 3.11+).
COPY data/preparation /app/data/preparation

# 6. Tests — only needed for in-container pytest (safe to include; small).
COPY platform/services/svm-api/tests /app/platform/services/svm-api/tests

ENV MODEL_PATH=/app/docs/metrics/improved_model_1/model.joblib \
    BASELINE2_MODEL_PATH=/app/docs/metrics/baseline_model_2/model.joblib \
    SVM_API_PORT=8000 \
    PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "svm_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
