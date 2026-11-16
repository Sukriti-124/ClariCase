# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir uv==0.4.*

WORKDIR /app

# Build context is the repo root (see docker-compose.yml).
COPY platform/shared /app/platform/shared
COPY platform/services/router/pyproject.toml /app/platform/services/router/pyproject.toml
COPY platform/services/router/src /app/platform/services/router/src

# DEVIATION FROM PLAN: router is installed with [test] extra so pytest+respx
# are available for in-container test runs if ever needed. Host-side tests
# cover the router already (6 passed in Task 11); this is a small
# convenience that costs ~10 MB.
RUN uv pip install --system --no-cache -e /app/platform/shared
RUN uv pip install --system --no-cache -e "/app/platform/services/router[test]"

# Tests — same rationale as svm-api.
COPY platform/services/router/tests /app/platform/services/router/tests

ENV ROUTER_PORT=8000
EXPOSE 8000
CMD ["uvicorn", "router.main:app", "--host", "0.0.0.0", "--port", "8000"]
