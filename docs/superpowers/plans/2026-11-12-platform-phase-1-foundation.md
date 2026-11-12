# Platform Phase 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the first vertical slice of the platform (`router` + `svm-api` + `postgres`) on docker-compose, with the shared HTTP contract frozen and observability hooks (structured logs, Prometheus `/metrics`, no-op OTel) wired into every service for later phases to adopt.

**Architecture:** Three containers on a compose network. `router` (FastAPI) proxies `POST /predict` to `svm-api` via `PRIMARY_URL`. `svm-api` (FastAPI) loads `improved_model_1`'s `model.joblib` at startup and serves predictions. `postgres` is up with schema loaded but unused in this phase. Shared code lives in `platform/shared/` and is imported by both services. Model code is reused from the existing `src/` tree via a root `pyproject.toml` that declares it as package `claricase-models`.

**Tech Stack:** Python 3.11, `uv` for dependency management, FastAPI, httpx, structlog, prometheus-fastapi-instrumentator, OpenTelemetry (no-op in P1), pytest, respx (for mocking httpx in tests), Docker + Docker Compose v2, PostgreSQL 16.

**Spec reference:** `docs/superpowers/specs/2026-11-11-platform-phase-1-foundation-design.md`

**Branch:** `week-11-designing-a-system` (already checked out).

---

## File Structure

### Modified (one file, minimal touch)

- `pyproject.toml` (NEW at repo root) — declares `src/` as package `claricase-models` so svm-api can import `src.improved_model_1.predict`. Does not change behavior of existing `src/` imports.

### New (all under `platform/`)

```
platform/
  README.md
  Makefile
  docker-compose.yml
  .env.example
  .gitignore
  shared/
    pyproject.toml
    src/claricase_platform_shared/
      __init__.py
      contracts.py       # PredictRequest, IssueProb, PredictResponse
      logging.py         # structlog JSON config + request_id contextvar
      telemetry.py       # Prom middleware + OTel no-op setup
    tests/
      test_contracts.py
      test_logging.py
      test_telemetry.py
  services/
    router/
      Dockerfile
      pyproject.toml
      src/router/
        __init__.py
        config.py        # pydantic Settings
        main.py          # FastAPI factory
        routes.py        # /predict, /healthz
      tests/
        conftest.py
        test_routes.py
    svm-api/
      Dockerfile
      pyproject.toml
      src/svm_api/
        __init__.py
        config.py
        main.py
        model.py         # load + predict wrapper
        routes.py        # /predict, /healthz
      tests/
        conftest.py
        test_model.py
        test_routes.py
  db/
    schema/
      001_init.sql
  tests/
    smoke.sh             # end-to-end compose smoke test
```

**Responsibilities summary:**
- `shared/contracts.py`: single source of truth for HTTP wire schema used by router and all model APIs.
- `shared/logging.py`: one `configure_logging(service_name)` entry point; threads `request_id` via contextvar.
- `shared/telemetry.py`: `setup_observability(app, service_name)` installs Prometheus middleware and OTel instrumentation; OTel exporter is no-op unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
- `services/svm-api/src/svm_api/model.py`: loads `model.joblib` once at startup, computes its sha256 as `model_version`, delegates prediction to `src.improved_model_1.predict.predict`.
- `services/svm-api/src/svm_api/routes.py`: validates input, calls `model.predict`, shapes response per shared contract.
- `services/router/src/router/routes.py`: generates `request_id`, proxies to `PRIMARY_URL`, maps errors to `503`.
- `db/schema/001_init.sql`: `complaints` and `predictions` tables with stable primary keys.
- `docker-compose.yml`: three services (`postgres`, `svm-api`, `router`) on one network; only `router` host-exposed.
- `Makefile`: `build | up | down | logs | test | smoke | clean` wrappers.
- `tests/smoke.sh`: brings compose up, curls `/predict`, asserts response shape, tears down.

### Build context note

All service Dockerfiles use the **repo root** as build context (`context: ..` in compose). This is required because svm-api needs to copy `src/`, `data/processed/team_mapping.csv`, `data/processed/issue_mapping.csv`, and `docs/metrics/improved_model_1/model.joblib` into the image, and because both services need `platform/shared/`.

---

## Task 1: Root `pyproject.toml` + regression check

**Why first:** Everything else depends on being able to `pip install -e .` the existing `src/` tree as `claricase-models`. Verify the change does not break existing imports or tests before building on top.

**Files:**
- Create: `pyproject.toml` (repo root)

- [ ] **Step 1: Capture baseline — existing tests pass today**

Run: `python -m pytest -q -m "not slow"`
Expected: all pass (record counts, we'll re-run after the change).

- [ ] **Step 2: Create root `pyproject.toml`**

Create `pyproject.toml` at repo root:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "claricase-models"
version = "0.0.1"
description = "ClariCase model code (reused by platform/ services)."
requires-python = ">=3.11"
# Runtime deps intentionally not re-declared here — services declare what
# they need. This package just exposes the existing src/ tree as importable.

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

- [ ] **Step 3: Install editable and verify existing tests still pass**

Run (in the project venv — Windows PowerShell: `.venv\Scripts\python.exe -m pip install -e .`; bash/macOS: `python -m pip install -e .`):

```
python -m pip install -e .
python -c "from src.improved_model_1 import predict as p; print('ok', p.__file__)"
python -m pytest -q -m "not slow"
```

Expected: `ok <path>` printed; same tests pass as in Step 1.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "Add root pyproject.toml exposing src/ as claricase-models package"
```

---

## Task 2: Platform scaffolding + `.gitignore` + `.env.example`

**Files:**
- Create: `platform/.gitignore`
- Create: `platform/.env.example`
- Create: `platform/README.md` (stub)
- Create: `platform/docker-compose.yml` (stub, filled later)

- [ ] **Step 1: Create `platform/.gitignore`**

```gitignore
# Secrets
.env
.env.*
!.env.example

# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/

# uv
.uv-cache/

# Test / build artifacts
.pytest_cache/
.coverage
htmlcov/
dist/
build/

# Local data
*.sqlite
*.sqlite-journal
```

- [ ] **Step 2: Create `platform/.env.example`**

```dotenv
# ---- Postgres ----
POSTGRES_USER=claricase
POSTGRES_PASSWORD=change-me-locally
POSTGRES_DB=claricase

# ---- Router ----
ROUTER_PORT=8080
PRIMARY_URL=http://svm-api:8000
LOG_LEVEL=INFO

# ---- svm-api ----
SVM_API_PORT=8000
MODEL_PATH=/app/docs/metrics/improved_model_1/model.joblib

# ---- OpenTelemetry (leave blank in P1 — Phase 7 fills this in) ----
OTEL_EXPORTER_OTLP_ENDPOINT=
OTEL_SERVICE_NAME=
```

- [ ] **Step 3: Create `platform/README.md` stub**

```markdown
# Platform (Phase 1)

First vertical slice of the ClariCase platform rebuild. See
`docs/superpowers/specs/2026-11-11-platform-phase-1-foundation-design.md`
for the design.

Run:

```bash
cp .env.example .env
make up
```

(This README is filled in by Task 20.)
```

- [ ] **Step 4: Commit**

```bash
git add platform/.gitignore platform/.env.example platform/README.md
git commit -m "Scaffold platform/ with .gitignore, .env.example, README stub"
```

---

## Task 3: Shared library — `contracts.py` (TDD)

**Files:**
- Create: `platform/shared/src/claricase_platform_shared/__init__.py`
- Create: `platform/shared/src/claricase_platform_shared/contracts.py`
- Create: `platform/shared/tests/test_contracts.py`
- Create: `platform/shared/pyproject.toml`

- [ ] **Step 1: Create `platform/shared/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "claricase-platform-shared"
version = "0.0.1"
description = "Shared contracts, logging, and telemetry for ClariCase platform services."
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6,<3",
    "structlog>=24.1,<25",
    "prometheus-fastapi-instrumentator>=7.0,<8",
    "opentelemetry-api>=1.24,<2",
    "opentelemetry-sdk>=1.24,<2",
    "opentelemetry-instrumentation-fastapi>=0.46b0,<1",
]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create `platform/shared/src/claricase_platform_shared/__init__.py`**

```python
"""Shared library for ClariCase platform services."""
```

- [ ] **Step 3: Write failing contracts test**

Create `platform/shared/tests/test_contracts.py`:

```python
import pytest
from pydantic import ValidationError

from claricase_platform_shared.contracts import (
    IssueProb,
    PredictRequest,
    PredictResponse,
)


def test_predict_request_accepts_valid_text():
    req = PredictRequest(text="I have a problem with my credit card bill today.")
    assert req.text.startswith("I have")
    assert req.request_id is None


def test_predict_request_rejects_short_text():
    with pytest.raises(ValidationError):
        PredictRequest(text="short")


def test_predict_request_round_trips_request_id():
    req = PredictRequest(text="A" * 20, request_id="abc-123")
    assert req.request_id == "abc-123"


def test_predict_response_requires_all_fields():
    resp = PredictResponse(
        model_name="improved_model_1",
        model_version="sha256:deadbeef",
        team="T01",
        team_confidence=0.82,
        top_issues=[IssueProb(issue="I001", prob=0.55)],
        auto_route=True,
        request_id="abc-123",
        latency_ms=42,
    )
    assert resp.auto_route is True
    assert resp.top_issues[0].prob == 0.55


def test_predict_response_rejects_bad_confidence():
    with pytest.raises(ValidationError):
        PredictResponse(
            model_name="m",
            model_version="v",
            team="T01",
            team_confidence=1.5,  # out of range
            top_issues=[],
            auto_route=False,
            request_id="r",
            latency_ms=1,
        )
```

- [ ] **Step 4: Run test to verify it fails**

From repo root:
```
cd platform/shared
python -m pip install -e .
python -m pytest tests/test_contracts.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'claricase_platform_shared.contracts'`.

- [ ] **Step 5: Implement `contracts.py`**

Create `platform/shared/src/claricase_platform_shared/contracts.py`:

```python
"""HTTP wire contract for ClariCase model-serving services.

This module is imported by the router and every model API service
(svm-api, seq2seq-api, agentic-api). Changes here affect the public
API of the platform — treat as a stable contract.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class IssueProb(BaseModel):
    issue: str
    prob: float = Field(ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    text: str = Field(min_length=10)
    request_id: str | None = None


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    team: str
    team_confidence: float = Field(ge=0.0, le=1.0)
    top_issues: list[IssueProb]
    auto_route: bool
    request_id: str
    latency_ms: int = Field(ge=0)
```

- [ ] **Step 6: Run test to verify it passes**

```
python -m pytest tests/test_contracts.py -v
```

Expected: all 5 pass.

- [ ] **Step 7: Commit**

```bash
git add platform/shared/
git commit -m "Add shared pydantic contracts for platform HTTP wire schema"
```

---

## Task 4: Shared library — `logging.py` (TDD)

**Files:**
- Create: `platform/shared/src/claricase_platform_shared/logging.py`
- Create: `platform/shared/tests/test_logging.py`

- [ ] **Step 1: Write failing test**

Create `platform/shared/tests/test_logging.py`:

```python
import io
import json
import logging

import pytest

from claricase_platform_shared.logging import (
    bind_request_id,
    configure_logging,
    get_request_id,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    logging.getLogger().handlers.clear()
    yield
    logging.getLogger().handlers.clear()


def test_configure_logging_writes_json_lines(capsys):
    configure_logging(service_name="svm-api")
    logger = logging.getLogger("svm-api")
    logger.info("hello", extra={"k": "v"})
    captured = capsys.readouterr().out.strip().splitlines()
    assert len(captured) == 1
    row = json.loads(captured[0])
    assert row["event"] == "hello"
    assert row["service"] == "svm-api"
    assert row["k"] == "v"


def test_bind_request_id_propagates_through_contextvar(capsys):
    configure_logging(service_name="router")
    bind_request_id("req-42")
    assert get_request_id() == "req-42"
    logging.getLogger("router").info("routed")
    row = json.loads(capsys.readouterr().out.strip().splitlines()[0])
    assert row["request_id"] == "req-42"


def test_request_id_absent_when_unbound(capsys):
    configure_logging(service_name="router")
    logging.getLogger("router").info("no-ctx")
    row = json.loads(capsys.readouterr().out.strip().splitlines()[0])
    assert "request_id" not in row or row["request_id"] is None
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_logging.py -v
```

Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `logging.py`**

Create `platform/shared/src/claricase_platform_shared/logging.py`:

```python
"""JSON logging with per-request context via contextvar.

Services call `configure_logging(service_name)` once at startup. Any
HTTP middleware (or manual code) that generates a request_id calls
`bind_request_id(rid)` so log lines downstream in the same async task
include it.
"""
from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar

import structlog

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def bind_request_id(request_id: str | None) -> None:
    _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


def _add_request_id(_logger, _method_name, event_dict):
    rid = _request_id.get()
    if rid is not None:
        event_dict["request_id"] = rid
    return event_dict


def _add_service(service_name: str):
    def processor(_logger, _method_name, event_dict):
        event_dict["service"] = service_name
        return event_dict
    return processor


def configure_logging(service_name: str) -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_service(service_name),
            _add_request_id,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Mirror stdlib logging through structlog so `logging.getLogger(...).info()`
    # also lands in JSON.
    class _StructlogHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self._log = structlog.get_logger()

        def emit(self, record: logging.LogRecord) -> None:
            extras = {
                k: v for k, v in record.__dict__.items()
                if k not in logging.LogRecord(
                    "", 0, "", 0, "", None, None
                ).__dict__ and k != "message"
            }
            self._log.log(record.levelno, record.getMessage(), **extras)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(_StructlogHandler())
    root.setLevel(level)
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_logging.py -v
```

Expected: 3 pass.

- [ ] **Step 5: Commit**

```bash
git add platform/shared/
git commit -m "Add structured JSON logging with request_id contextvar"
```

---

## Task 5: Shared library — `telemetry.py` (TDD)

**Files:**
- Create: `platform/shared/src/claricase_platform_shared/telemetry.py`
- Create: `platform/shared/tests/test_telemetry.py`

- [ ] **Step 1: Write failing test**

Create `platform/shared/tests/test_telemetry.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

from claricase_platform_shared.telemetry import setup_observability


def test_metrics_endpoint_exposes_prometheus_format():
    app = FastAPI()
    setup_observability(app, service_name="svm-api")

    @app.get("/ping")
    def ping():
        return {"ok": True}

    client = TestClient(app)
    client.get("/ping")  # drive at least one request

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "# HELP" in body
    assert "http_request_duration" in body or "http_requests_total" in body


def test_otel_no_op_when_endpoint_unset(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    app = FastAPI()
    # Must not raise even though no collector exists:
    setup_observability(app, service_name="router")
```

Add the test-time dep if missing — `pyproject.toml` already has FastAPI's transitive deps via instrumentator, but we also need `fastapi` and `httpx` for the TestClient. Update `platform/shared/pyproject.toml` to add:

```toml
[project.optional-dependencies]
test = [
    "pytest>=8",
    "fastapi>=0.110,<1",
    "httpx>=0.27,<1",
]
```

Then reinstall: `python -m pip install -e ".[test]"` from `platform/shared/`.

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_telemetry.py -v
```

Expected: FAIL (`ImportError` for `setup_observability`).

- [ ] **Step 3: Implement `telemetry.py`**

Create `platform/shared/src/claricase_platform_shared/telemetry.py`:

```python
"""Observability setup: Prometheus metrics + OpenTelemetry tracing.

Call `setup_observability(app, service_name)` once in each FastAPI
service's app factory. Metrics are always on. OTel tracing exports
only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set; otherwise a no-op
tracer provider is installed so `trace.get_tracer(...)` still works.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_fastapi_instrumentator import Instrumentator


def _configure_tracing(service_name: str) -> None:
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if endpoint:
        # Imported lazily so services that don't configure OTel avoid
        # paying for gRPC at startup.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )

    trace.set_tracer_provider(provider)


def setup_observability(app: FastAPI, service_name: str) -> None:
    _configure_tracing(service_name)
    FastAPIInstrumentor.instrument_app(app)

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
```

If the OTLP HTTP exporter is not pulled by `opentelemetry-exporter-otlp` metapackage, add `opentelemetry-exporter-otlp-proto-http>=1.24,<2` to `platform/shared/pyproject.toml` dependencies and reinstall.

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_telemetry.py -v
```

Expected: both pass.

- [ ] **Step 5: Run all shared tests together**

```
python -m pytest -v
```

Expected: contracts (5) + logging (3) + telemetry (2) = 10 pass.

- [ ] **Step 6: Commit**

```bash
git add platform/shared/
git commit -m "Add Prometheus + OTel observability setup helper"
```

---

## Task 6: svm-api — `pyproject.toml` and uv setup

**Files:**
- Create: `platform/services/svm-api/pyproject.toml`
- Create: `platform/services/svm-api/src/svm_api/__init__.py`

- [ ] **Step 1: Create `platform/services/svm-api/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "svm-api"
version = "0.0.1"
description = "ClariCase svm-api: serves improved_model_1 (hierarchical TF-IDF + Linear SVM)."
requires-python = ">=3.11,<3.12"
dependencies = [
    "fastapi>=0.110,<1",
    "uvicorn[standard]>=0.29,<1",
    "pydantic-settings>=2.2,<3",
    "scikit-learn==1.9.1",   # MUST match the version the model was saved with
    "joblib>=1.3,<2",
    "numpy>=1.26,<2",
    "pandas>=2.2,<3",
    "pyarrow>=15,<20",
    "claricase-platform-shared",
    "claricase-models",
]

[project.optional-dependencies]
test = [
    "pytest>=8",
    "httpx>=0.27,<1",
]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create `platform/services/svm-api/src/svm_api/__init__.py`**

```python
"""svm-api: serves improved_model_1 behind the shared platform HTTP contract."""
```

- [ ] **Step 3: Install svm-api (editable) and verify imports resolve**

From repo root:
```
python -m pip install -e platform/shared
python -m pip install -e . platform/services/svm-api[test]
python -c "import svm_api; import src.improved_model_1.predict as p; print('ok')"
```

Expected: `ok` printed.

- [ ] **Step 4: Commit**

```bash
git add platform/services/svm-api/
git commit -m "Scaffold svm-api service package"
```

---

## Task 7: svm-api — `model.py` (TDD)

**Files:**
- Create: `platform/services/svm-api/src/svm_api/model.py`
- Create: `platform/services/svm-api/tests/__init__.py`
- Create: `platform/services/svm-api/tests/conftest.py`
- Create: `platform/services/svm-api/tests/test_model.py`

- [ ] **Step 1: Create `tests/__init__.py`** (empty file).

- [ ] **Step 2: Create `tests/conftest.py`** with a session-scoped fixture that loads the real artifact.

```python
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]  # platform/services/svm-api/tests → repo root
MODEL_PATH = REPO_ROOT / "docs" / "metrics" / "improved_model_1" / "model.joblib"


@pytest.fixture(scope="session")
def model_path() -> Path:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model artifact missing: {MODEL_PATH}")
    os.environ.setdefault("MODEL_PATH", str(MODEL_PATH))
    return MODEL_PATH
```

- [ ] **Step 3: Write failing test**

Create `platform/services/svm-api/tests/test_model.py`:

```python
import pytest

from svm_api.model import ModelBundle, load_bundle


SAMPLE_TEXT = (
    "I applied for a credit card and the bank reported late payments to the "
    "credit bureaus even though I paid on time every month for the last year."
)


def test_load_bundle_reads_artifact(model_path):
    bundle = load_bundle(model_path)
    assert isinstance(bundle, ModelBundle)
    assert bundle.model_name == "improved_model_1"
    assert bundle.version.startswith("sha256:")
    assert len(bundle.version) == len("sha256:") + 64


def test_predict_returns_contract_shaped_dict(model_path):
    bundle = load_bundle(model_path)
    out = bundle.predict(SAMPLE_TEXT, top_k=3)
    assert isinstance(out["team"], str)
    assert 0.0 <= out["team_confidence"] <= 1.0
    assert isinstance(out["top_issues"], list)
    assert len(out["top_issues"]) <= 3
    for row in out["top_issues"]:
        assert set(row.keys()) == {"issue", "prob"}
        assert 0.0 <= row["prob"] <= 1.0
    assert isinstance(out["auto_route"], bool)


def test_predict_rejects_empty_text(model_path):
    bundle = load_bundle(model_path)
    with pytest.raises(ValueError):
        bundle.predict("", top_k=3)
```

- [ ] **Step 4: Run test to verify it fails**

From repo root:
```
python -m pytest platform/services/svm-api/tests/test_model.py -v
```

Expected: FAIL (`ModuleNotFoundError: svm_api.model`).

- [ ] **Step 5: Implement `model.py`**

Create `platform/services/svm-api/src/svm_api/model.py`:

```python
"""Model artifact loader and prediction wrapper for svm-api.

Thin wrapper around `src.improved_model_1.predict.predict` that:
- loads `model.joblib` once at startup
- computes a sha256 fingerprint for `model_version`
- reshapes the existing predict output into the shared HTTP contract shape
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from src.improved_model_1 import predict as upstream


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


@dataclass(frozen=True)
class ModelBundle:
    raw: Any
    version: str
    model_name: str
    artifact_path: Path

    def predict(self, text: str, top_k: int = 3) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("text is empty")
        up = upstream.predict(text, top_k=top_k, model=self.raw)

        return {
            "model_name": self.model_name,
            "model_version": self.version,
            "team": up["predicted_team_id"],
            "team_confidence": float(up["confidence"]),
            "top_issues": [
                {"issue": row["issue_id"], "prob": float(row["confidence"])}
                for row in up["issue_top_k"]
            ],
            "auto_route": up["route"] == "auto",
        }


def load_bundle(path: Path) -> ModelBundle:
    raw = joblib.load(path)
    model_name = raw.get("model_name", "improved_model_1")
    return ModelBundle(
        raw=raw,
        version=_fingerprint(path),
        model_name=model_name,
        artifact_path=path,
    )
```

- [ ] **Step 6: Run test to verify it passes**

```
python -m pytest platform/services/svm-api/tests/test_model.py -v
```

Expected: 3 pass.

- [ ] **Step 7: Commit**

```bash
git add platform/services/svm-api/
git commit -m "Add svm-api model loader + prediction wrapper over improved_model_1"
```

---

## Task 8: svm-api — `config.py` + `main.py` + `routes.py` (TDD)

**Files:**
- Create: `platform/services/svm-api/src/svm_api/config.py`
- Create: `platform/services/svm-api/src/svm_api/main.py`
- Create: `platform/services/svm-api/src/svm_api/routes.py`
- Create: `platform/services/svm-api/tests/test_routes.py`

- [ ] **Step 1: Create `config.py`**

```python
"""svm-api configuration (env-driven)."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    service_name: str = Field(default="svm-api")
    port: int = Field(default=8000, alias="SVM_API_PORT")
    model_path: Path = Field(
        default=Path("/app/docs/metrics/improved_model_1/model.joblib"),
        alias="MODEL_PATH",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 2: Write failing routes test**

Create `platform/services/svm-api/tests/test_routes.py`:

```python
from fastapi.testclient import TestClient

from svm_api.main import create_app


def test_healthz_is_200_after_model_loaded(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_predict_returns_contract_shape(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            json={"text": "I disputed a charge on my credit card and never got a response from the bank."},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model_name"] == "improved_model_1"
        assert body["model_version"].startswith("sha256:")
        assert isinstance(body["team"], str)
        assert 0.0 <= body["team_confidence"] <= 1.0
        assert isinstance(body["top_issues"], list)
        assert isinstance(body["auto_route"], bool)
        assert body["request_id"]
        assert body["latency_ms"] >= 0


def test_predict_rejects_short_text(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    app = create_app()
    with TestClient(app) as client:
        resp = client.post("/predict", json={"text": "short"})
        assert resp.status_code == 422


def test_predict_preserves_request_id(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            json={
                "text": "The credit bureau refuses to remove an incorrect collections account from my report.",
                "request_id": "rid-xyz-1",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["request_id"] == "rid-xyz-1"


def test_metrics_endpoint_present(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    app = create_app()
    with TestClient(app) as client:
        client.get("/healthz")
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "http_requests_total" in resp.text or "http_request_duration" in resp.text
```

- [ ] **Step 3: Run test to verify it fails**

```
python -m pytest platform/services/svm-api/tests/test_routes.py -v
```

Expected: FAIL (`ModuleNotFoundError: svm_api.main`).

- [ ] **Step 4: Implement `main.py`**

```python
"""svm-api FastAPI application factory."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from claricase_platform_shared.logging import configure_logging
from claricase_platform_shared.telemetry import setup_observability

from svm_api.config import get_settings
from svm_api.model import load_bundle
from svm_api.routes import router as http_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.service_name)
    app.state.settings = settings
    app.state.bundle = load_bundle(settings.model_path)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="svm-api", lifespan=lifespan)
    setup_observability(app, service_name="svm-api")
    app.include_router(http_router)
    return app


app = create_app()
```

- [ ] **Step 5: Implement `routes.py`**

```python
"""svm-api HTTP routes."""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Request

from claricase_platform_shared.contracts import (
    IssueProb,
    PredictRequest,
    PredictResponse,
)
from claricase_platform_shared.logging import bind_request_id

router = APIRouter()


@router.get("/healthz")
def healthz(request: Request) -> dict[str, str]:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return {"status": "loading"}
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    rid = body.request_id or str(uuid.uuid4())
    bind_request_id(rid)

    bundle = request.app.state.bundle
    t0 = time.perf_counter()
    raw: dict[str, Any] = bundle.predict(body.text, top_k=3)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return PredictResponse(
        model_name=raw["model_name"],
        model_version=raw["model_version"],
        team=raw["team"],
        team_confidence=raw["team_confidence"],
        top_issues=[IssueProb(**r) for r in raw["top_issues"]],
        auto_route=raw["auto_route"],
        request_id=rid,
        latency_ms=latency_ms,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

```
python -m pytest platform/services/svm-api/tests/ -v
```

Expected: 3 (model) + 5 (routes) = 8 pass.

- [ ] **Step 7: Commit**

```bash
git add platform/services/svm-api/
git commit -m "Add svm-api FastAPI app with /predict, /healthz, /metrics"
```

---

## Task 9: svm-api — Dockerfile

**Files:**
- Create: `platform/services/svm-api/Dockerfile`

- [ ] **Step 1: Create the Dockerfile**

Create `platform/services/svm-api/Dockerfile`:

```dockerfile
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
RUN uv pip install --system --no-cache -e /app
RUN uv pip install --system --no-cache -e /app/platform/shared
RUN uv pip install --system --no-cache -e /app/platform/services/svm-api

# 5. Model artifact + reference data.
COPY docs/metrics/improved_model_1/model.joblib \
     /app/docs/metrics/improved_model_1/model.joblib
COPY data/processed/team_mapping.csv  /app/data/processed/team_mapping.csv
COPY data/processed/issue_mapping.csv /app/data/processed/issue_mapping.csv

ENV MODEL_PATH=/app/docs/metrics/improved_model_1/model.joblib \
    SVM_API_PORT=8000

EXPOSE 8000
CMD ["uvicorn", "svm_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Build the image locally to verify it works**

From repo root:
```
docker build -f platform/services/svm-api/Dockerfile -t claricase/svm-api:dev .
```

Expected: build succeeds.

- [ ] **Step 3: Run the image standalone and hit `/healthz`**

```
docker run --rm -d --name svm-api-smoke -p 18000:8000 claricase/svm-api:dev
# Wait up to 30s for startup (model load takes a few seconds)
for i in 1 2 3 4 5 6 7 8 9 10; do
  curl -sf http://localhost:18000/healthz && break
  sleep 3
done
docker stop svm-api-smoke
```

Expected: `{"status":"ok"}` on one of the attempts.

- [ ] **Step 4: Commit**

```bash
git add platform/services/svm-api/Dockerfile
git commit -m "Add svm-api Dockerfile (python:3.11-slim + uv, model baked in)"
```

---

## Task 10: router — `pyproject.toml` and package scaffold

**Files:**
- Create: `platform/services/router/pyproject.toml`
- Create: `platform/services/router/src/router/__init__.py`

- [ ] **Step 1: Create `platform/services/router/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "router"
version = "0.0.1"
description = "ClariCase router: proxies /predict to a configured primary model service."
requires-python = ">=3.11,<3.12"
dependencies = [
    "fastapi>=0.110,<1",
    "uvicorn[standard]>=0.29,<1",
    "pydantic-settings>=2.2,<3",
    "httpx>=0.27,<1",
    "claricase-platform-shared",
]

[project.optional-dependencies]
test = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create `platform/services/router/src/router/__init__.py`**

```python
"""router: proxies /predict to a configured primary model service."""
```

- [ ] **Step 3: Install and verify**

```
python -m pip install -e platform/services/router[test]
python -c "import router; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add platform/services/router/pyproject.toml platform/services/router/src/router/__init__.py
git commit -m "Scaffold router service package"
```

---

## Task 11: router — `config.py`, `main.py`, `routes.py` (TDD)

**Files:**
- Create: `platform/services/router/src/router/config.py`
- Create: `platform/services/router/src/router/routes.py`
- Create: `platform/services/router/src/router/main.py`
- Create: `platform/services/router/tests/__init__.py`
- Create: `platform/services/router/tests/test_routes.py`

- [ ] **Step 1: Create `config.py`**

```python
"""router configuration (env-driven)."""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    service_name: str = Field(default="router")
    port: int = Field(default=8080, alias="ROUTER_PORT")
    primary_url: str = Field(
        default="http://svm-api:8000",
        alias="PRIMARY_URL",
    )
    primary_timeout_s: float = Field(default=5.0, alias="PRIMARY_TIMEOUT_S")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 2: Create `tests/__init__.py`** (empty).

- [ ] **Step 3: Write failing routes test**

Create `platform/services/router/tests/test_routes.py`:

```python
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from router.main import create_app


PRIMARY_RESPONSE = {
    "model_name": "improved_model_1",
    "model_version": "sha256:abc",
    "team": "T01",
    "team_confidence": 0.82,
    "top_issues": [{"issue": "I001", "prob": 0.55}],
    "auto_route": True,
    "request_id": "server-generated",
    "latency_ms": 11,
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PRIMARY_URL", "http://svm-api.test")
    app = create_app()
    with TestClient(app) as c:
        yield c


@respx.mock
def test_predict_proxies_to_primary_and_returns_response(client):
    route = respx.post("http://svm-api.test/predict").mock(
        return_value=httpx.Response(200, json=PRIMARY_RESPONSE)
    )
    resp = client.post(
        "/predict",
        json={"text": "I have a long enough complaint text to pass validation."},
    )
    assert route.called
    assert resp.status_code == 200
    body = resp.json()
    assert body["team"] == "T01"


@respx.mock
def test_predict_generates_request_id_when_absent(client):
    captured: dict = {}

    def _record(request):
        captured["body"] = request.read().decode()
        return httpx.Response(200, json=PRIMARY_RESPONSE)

    respx.post("http://svm-api.test/predict").mock(side_effect=_record)
    resp = client.post(
        "/predict",
        json={"text": "I have a long enough complaint text to pass validation."},
    )
    assert resp.status_code == 200
    assert "request_id" in captured["body"]


@respx.mock
def test_predict_preserves_client_request_id(client):
    captured: dict = {}

    def _record(request):
        captured["body"] = request.read().decode()
        return httpx.Response(200, json=PRIMARY_RESPONSE)

    respx.post("http://svm-api.test/predict").mock(side_effect=_record)
    resp = client.post(
        "/predict",
        json={
            "text": "I have a long enough complaint text to pass validation.",
            "request_id": "rid-42",
        },
    )
    assert resp.status_code == 200
    assert "rid-42" in captured["body"]


@respx.mock
def test_primary_5xx_maps_to_503(client):
    respx.post("http://svm-api.test/predict").mock(
        return_value=httpx.Response(500, text="boom")
    )
    resp = client.post(
        "/predict",
        json={"text": "I have a long enough complaint text to pass validation."},
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "primary_unhealthy"


@respx.mock
def test_primary_timeout_maps_to_503(client):
    respx.post("http://svm-api.test/predict").mock(
        side_effect=httpx.TimeoutException("slow")
    )
    resp = client.post(
        "/predict",
        json={"text": "I have a long enough complaint text to pass validation."},
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "primary_unhealthy"


def test_metrics_endpoint_present(client):
    client.get("/healthz")  # ignore outcome
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "# HELP" in resp.text
```

- [ ] **Step 4: Run test to verify it fails**

```
python -m pytest platform/services/router/tests -v
```

Expected: FAIL (`ModuleNotFoundError: router.main`).

- [ ] **Step 5: Implement `routes.py`**

```python
"""router HTTP routes."""
from __future__ import annotations

import uuid

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from claricase_platform_shared.contracts import PredictRequest, PredictResponse
from claricase_platform_shared.logging import bind_request_id

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    client: httpx.AsyncClient = request.app.state.http
    settings = request.app.state.settings
    try:
        r = await client.get(f"{settings.primary_url}/healthz", timeout=2.0)
        if r.status_code == 200:
            return {"status": "ok"}
    except httpx.HTTPError:
        pass
    raise HTTPException(status_code=503, detail="primary_unreachable")


@router.post("/predict")
async def predict(body: PredictRequest, request: Request) -> JSONResponse:
    rid = body.request_id or str(uuid.uuid4())
    bind_request_id(rid)

    client: httpx.AsyncClient = request.app.state.http
    settings = request.app.state.settings

    payload = body.model_dump()
    payload["request_id"] = rid

    try:
        resp = await client.post(
            f"{settings.primary_url}/predict",
            json=payload,
            timeout=settings.primary_timeout_s,
        )
    except httpx.HTTPError:
        return JSONResponse(
            status_code=503,
            content={"error_code": "primary_unhealthy", "request_id": rid},
        )

    if resp.status_code >= 500:
        return JSONResponse(
            status_code=503,
            content={"error_code": "primary_unhealthy", "request_id": rid},
        )

    # Pass non-5xx responses through unchanged. We validate the schema only
    # on 2xx so a surprising 4xx body from the primary still reaches the
    # client untouched.
    if 200 <= resp.status_code < 300:
        validated = PredictResponse(**resp.json())
        return JSONResponse(status_code=resp.status_code, content=validated.model_dump())
    return JSONResponse(status_code=resp.status_code, content=resp.json())
```

- [ ] **Step 6: Implement `main.py`**

```python
"""router FastAPI application factory."""
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from claricase_platform_shared.logging import configure_logging
from claricase_platform_shared.telemetry import setup_observability

from router.config import get_settings
from router.routes import router as http_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.service_name)
    app.state.settings = settings
    app.state.http = httpx.AsyncClient()
    try:
        yield
    finally:
        await app.state.http.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="router", lifespan=lifespan)
    setup_observability(app, service_name="router")
    app.include_router(http_router)
    return app


app = create_app()
```

- [ ] **Step 7: Run tests to verify they pass**

```
python -m pytest platform/services/router/tests -v
```

Expected: 6 pass.

- [ ] **Step 8: Commit**

```bash
git add platform/services/router/
git commit -m "Add router FastAPI app proxying /predict with 503 on primary failure"
```

---

## Task 12: router — Dockerfile

**Files:**
- Create: `platform/services/router/Dockerfile`

- [ ] **Step 1: Create the Dockerfile**

```dockerfile
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

RUN uv pip install --system --no-cache -e /app/platform/shared
RUN uv pip install --system --no-cache -e /app/platform/services/router

ENV ROUTER_PORT=8000
EXPOSE 8000
CMD ["uvicorn", "router.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Note: router listens on port 8000 inside the container; docker-compose maps host 8080 → container 8000.

- [ ] **Step 2: Build locally**

From repo root:
```
docker build -f platform/services/router/Dockerfile -t claricase/router:dev .
```

Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add platform/services/router/Dockerfile
git commit -m "Add router Dockerfile"
```

---

## Task 13: Postgres schema

**Files:**
- Create: `platform/db/schema/001_init.sql`

- [ ] **Step 1: Create the schema file**

```sql
-- Phase 1: schema is defined but neither table is read or written by any service
-- in this phase. `predictions` becomes active in Phase 4 (shadow mode).

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS complaints (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL    DEFAULT now(),
    text            TEXT        NOT NULL,
    predicted_team  TEXT        NULL,
    tracking_id     TEXT        UNIQUE      NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL    DEFAULT now(),
    request_id      TEXT        NOT NULL,
    complaint_id    UUID        NULL REFERENCES complaints(id) ON DELETE SET NULL,
    role            TEXT        NOT NULL    CHECK (role IN ('primary', 'shadow')),
    model_name      TEXT        NOT NULL,
    model_version   TEXT        NOT NULL,
    team            TEXT        NOT NULL,
    team_confidence DOUBLE PRECISION NOT NULL,
    top_issues      JSONB       NOT NULL,
    auto_route      BOOLEAN     NOT NULL,
    latency_ms      INTEGER     NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_predictions_request_id ON predictions(request_id);
CREATE INDEX IF NOT EXISTS ix_predictions_created_at ON predictions(created_at);
```

- [ ] **Step 2: Commit**

```bash
git add platform/db/schema/001_init.sql
git commit -m "Add initial Postgres schema (complaints, predictions)"
```

---

## Task 14: `docker-compose.yml`

**Files:**
- Create: `platform/docker-compose.yml`

- [ ] **Step 1: Create `platform/docker-compose.yml`**

```yaml
name: claricase-platform

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/schema:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 3s
      retries: 10

  svm-api:
    build:
      context: ..
      dockerfile: platform/services/svm-api/Dockerfile
    image: claricase/svm-api:dev
    environment:
      LOG_LEVEL: ${LOG_LEVEL}
      MODEL_PATH: ${MODEL_PATH}
      OTEL_EXPORTER_OTLP_ENDPOINT: ${OTEL_EXPORTER_OTLP_ENDPOINT}
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/healthz',timeout=2).getcode()==200 else sys.exit(1)"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 30s
    depends_on:
      postgres:
        condition: service_healthy

  router:
    build:
      context: ..
      dockerfile: platform/services/router/Dockerfile
    image: claricase/router:dev
    environment:
      PRIMARY_URL: ${PRIMARY_URL}
      LOG_LEVEL: ${LOG_LEVEL}
      OTEL_EXPORTER_OTLP_ENDPOINT: ${OTEL_EXPORTER_OTLP_ENDPOINT}
    ports:
      - "${ROUTER_PORT}:8000"
    depends_on:
      svm-api:
        condition: service_healthy

volumes:
  pgdata:
```

- [ ] **Step 2: Bring the stack up manually**

From `platform/`:
```
cp .env.example .env
docker compose build
docker compose up -d
docker compose ps
```

Expected: all three services `running`; `svm-api` reports `healthy` within ~60 s.

- [ ] **Step 3: Exercise the slice**

```bash
curl -s http://localhost:8080/healthz
curl -s -X POST http://localhost:8080/predict \
     -H 'content-type: application/json' \
     -d '{"text":"The bank reported wrong information to the credit bureau and has not corrected it after multiple disputes."}'
```

Expected: `{"status":"ok"}` from healthz; a valid PredictResponse JSON with `model_name: "improved_model_1"` from predict.

- [ ] **Step 4: Tear down and commit**

```bash
docker compose down
```

```bash
git add platform/docker-compose.yml
git commit -m "Add docker-compose for postgres + svm-api + router"
```

---

## Task 15: `Makefile`

**Files:**
- Create: `platform/Makefile`

- [ ] **Step 1: Create `platform/Makefile`**

```makefile
# All targets run from the platform/ directory. Compose uses this dir as its
# project root; build contexts are the repo root (one level up).

SHELL := /bin/bash
DC    := docker compose
PY    := python
REPO_ROOT := $(shell cd .. && pwd)

.PHONY: help build up down logs ps test smoke clean

help:
	@echo "Targets:"
	@echo "  build   Build all service images"
	@echo "  up      Start the stack (detached)"
	@echo "  down    Stop and remove containers (keeps pgdata volume)"
	@echo "  logs    Tail logs from all services"
	@echo "  ps      List running services"
	@echo "  test    Run per-service unit tests on the host"
	@echo "  smoke   End-to-end smoke test (brings compose up + curls /predict + tears down)"
	@echo "  clean   Down + remove pgdata volume"

build:
	$(DC) build

up:
	@[[ -f .env ]] || cp .env.example .env
	$(DC) up -d

down:
	$(DC) down

logs:
	$(DC) logs -f

ps:
	$(DC) ps

test:
	cd $(REPO_ROOT) && $(PY) -m pytest platform/shared/tests -v
	cd $(REPO_ROOT) && $(PY) -m pytest platform/services/svm-api/tests -v
	cd $(REPO_ROOT) && $(PY) -m pytest platform/services/router/tests -v

smoke:
	bash tests/smoke.sh

clean:
	$(DC) down -v
```

- [ ] **Step 2: Verify `make help` works**

```
cd platform && make help
```

Expected: target list printed.

- [ ] **Step 3: Commit**

```bash
git add platform/Makefile
git commit -m "Add platform Makefile wrapping compose + tests + smoke"
```

---

## Task 16: End-to-end smoke test

**Files:**
- Create: `platform/tests/smoke.sh`

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
# End-to-end smoke test for the Phase 1 slice.
# Brings the compose stack up, hits /predict, validates the response shape,
# then tears down. Fails fast on any step.

set -euo pipefail

cd "$(dirname "$0")/.."

[[ -f .env ]] || cp .env.example .env

cleanup() {
    docker compose down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo ">> building"
docker compose build

echo ">> starting"
docker compose up -d

echo ">> waiting for router /healthz"
for i in $(seq 1 30); do
    if curl -sf http://localhost:8080/healthz >/dev/null; then
        break
    fi
    sleep 2
    if [[ $i -eq 30 ]]; then
        echo "!! router did not become healthy in time"
        docker compose logs
        exit 1
    fi
done

echo ">> POST /predict"
BODY=$(curl -sf -X POST http://localhost:8080/predict \
  -H 'content-type: application/json' \
  -d '{"text":"The bank reported wrong information to the credit bureau and has not corrected it after multiple disputes."}')

echo "   response: $BODY"

echo "$BODY" | python -c '
import json, sys
d = json.loads(sys.stdin.read())
assert d["model_name"] == "improved_model_1", d
assert d["model_version"].startswith("sha256:"), d
assert isinstance(d["team"], str) and d["team"]
assert 0.0 <= d["team_confidence"] <= 1.0
assert isinstance(d["top_issues"], list)
assert isinstance(d["auto_route"], bool)
assert d["request_id"]
assert isinstance(d["latency_ms"], int) and d["latency_ms"] >= 0
print("   schema: OK")
'

echo ">> smoke test passed"
```

- [ ] **Step 2: Make it executable (committed as executable)**

```
chmod +x platform/tests/smoke.sh
```

(On Windows: the file content matters, not the local file mode; `make smoke` runs it via `bash`.)

- [ ] **Step 3: Run the smoke test end-to-end**

```
cd platform && make smoke
```

Expected: ends with `>> smoke test passed`.

- [ ] **Step 4: Commit**

```bash
git add platform/tests/smoke.sh
git commit -m "Add end-to-end smoke test for Phase 1 slice"
```

---

## Task 17: Fill in `platform/README.md`

**Files:**
- Modify: `platform/README.md`

- [ ] **Step 1: Replace stub with the real README**

```markdown
# ClariCase Platform — Phase 1

First vertical slice of the platform rebuild:

```
curl → router → svm-api → prediction response
                 ↓
            postgres (schema present; unused in Phase 1)
```

**Spec:** `../docs/superpowers/specs/2026-11-11-platform-phase-1-foundation-design.md`
**Plan:** `../docs/superpowers/plans/2026-11-12-platform-phase-1-foundation.md`

## Services

| Service      | Role                                              | Host port |
|--------------|---------------------------------------------------|-----------|
| `router`     | Proxies `/predict` to `PRIMARY_URL` (svm-api)     | 8080      |
| `svm-api`    | Serves `improved_model_1` (TF-IDF + hierarchical SVM) | internal |
| `postgres`   | Schema loaded; no reads/writes in Phase 1         | internal  |

## Requirements

- Docker + Docker Compose v2
- (for host-side unit tests) Python 3.11 and the project venv at the repo root

## Run

```bash
cd platform
cp .env.example .env
make up
curl -s http://localhost:8080/healthz
curl -s -X POST http://localhost:8080/predict \
     -H 'content-type: application/json' \
     -d '{"text":"The bank reported wrong information to the credit bureau after multiple disputes."}'
```

## Everyday commands

```bash
make build    # build images
make up       # start stack (detached)
make down     # stop stack (keeps data volume)
make logs     # tail logs
make test     # run host-side unit tests (shared + svm-api + router)
make smoke    # end-to-end smoke test (brings stack up + asserts shape + tears down)
make clean    # down + remove data volume
```

## Observability

Phase 1 wires observability hooks into every service but does **not** deploy a
backend for them. You can inspect them manually:

- `/metrics` on each service — Prometheus exposition format. From the host:
  `docker compose exec svm-api curl -s localhost:8000/metrics | head`
- Structured JSON logs — `docker compose logs svm-api` emits one JSON line per
  event with `service`, `request_id`, `level`, `timestamp`, `event`.
- OpenTelemetry — SDK is imported and FastAPI is auto-instrumented. The
  exporter is a no-op unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

Phase 5 (Prometheus + Grafana), Phase 6 (Loki), and Phase 7 (Tempo) each
connect a backend; no code change in this service is required.

## Not in Phase 1

- Kubernetes (Phase 2)
- Other model services — seq2seq-api, agentic-api (Phase 3)
- Shadow mode + Redis queue (Phase 4)
- Prometheus / Grafana / Loki / Tempo deployed (Phases 5–7)
- Next.js frontend (Phase 8)
- Nightly eval CronJob (Phase 9)
- Cloud deploy on DOKS (Phase 10)
- Any change to `main`, the existing Streamlit app, or `src/app/`
```

- [ ] **Step 2: Commit**

```bash
git add platform/README.md
git commit -m "Fill in platform README with run instructions and phase context"
```

---

## Task 18: Final acceptance verification against spec Section 11

**Files:** (none — verification only)

- [ ] **Step 1: Verify criterion 1 — directory layout exists**

```
ls platform/shared/src/claricase_platform_shared/ \
   platform/services/svm-api/src/svm_api/ \
   platform/services/router/src/router/ \
   platform/db/schema/
```

Expected: all four directories present with the expected files.

- [ ] **Step 2: Verify criteria 2, 3, 4 — `make build`, `make up`, `make smoke`, `make test`**

```
cd platform
make build
make up
docker compose ps
make test
make smoke
make down
```

Expected: all commands succeed; `make smoke` ends with `>> smoke test passed`.

- [ ] **Step 3: Verify criterion 5 — svm-api `/metrics`**

```
cd platform
make up
docker compose exec svm-api curl -s http://localhost:8000/metrics | \
    grep -E 'http_requests_total|http_request_duration' | head
make down
```

Expected: at least one matching line.

- [ ] **Step 4: Verify criterion 6 — structured JSON logs with request_id**

```
cd platform
make up
RID="rid-verify-$(date +%s)"
curl -s -X POST http://localhost:8080/predict \
     -H 'content-type: application/json' \
     -d "{\"text\":\"The bank reported wrong information to the credit bureau after multiple disputes.\",\"request_id\":\"$RID\"}" >/dev/null
# Give logs a moment to flush
sleep 1
docker compose logs router  | grep "$RID" | head
docker compose logs svm-api | grep "$RID" | head
make down
```

Expected: at least one JSON line from each service containing the request_id.

- [ ] **Step 5: Verify criterion 7 — existing tests still pass**

From repo root:
```
python -m pytest -q -m "not slow"
```

Expected: same as the baseline captured in Task 1 Step 1.

- [ ] **Step 6: Verify criterion 8 — `main` branch and existing code untouched**

```bash
git log --oneline origin/main..HEAD -- streamlit_app.py src/app/ src/improved_model_1/train.py 2>/dev/null
```

Expected: empty output (no commits on this branch touch those paths).

Also:

```bash
git diff origin/main..HEAD -- streamlit_app.py src/app/ | head
```

Expected: empty.

- [ ] **Step 7: Verify criterion 9 — `platform/README.md` documents the system**

```
test -s platform/README.md && echo OK
```

Expected: `OK`.

- [ ] **Step 8: Final commit (empty, marks completion)**

```bash
git commit --allow-empty -m "Phase 1 acceptance checks pass — foundation complete"
```

---

## Self-Review (checked by the author of this plan)

**Spec coverage (against `2026-11-11-platform-phase-1-foundation-design.md`):**

- §2 Phase 1 goal — satisfied by Tasks 6–8 (svm-api), 10–11 (router), 13 (Postgres), 14 (compose).
- §2 Explicit non-goals — enforced by scope: no tasks touch K8s, Redis, other model services, Next.js, eval CronJob, or any `main`/`streamlit_app.py` path.
- §3 Scope of changes / root `pyproject.toml` — Task 1, with the regression check in Task 1 Step 3 and Task 18 Step 5.
- §4 Directory layout — Tasks 2–16 collectively create every file listed in §4.
- §5 Shared HTTP contract — Task 3 (pydantic models), reused by svm-api routes (Task 8) and router routes (Task 11).
- §6.1 Router behavior — Task 11 (routes.py) implements `PRIMARY_URL` env, request_id generation, 503 on 5xx/timeout, no fallback.
- §6.2 svm-api behavior — Tasks 7–8 (model load, sha256 fingerprint, reuse of `src.improved_model_1.predict.predict`).
- §6.3 Postgres — Task 13 (schema), Task 14 (compose brings postgres up with init SQL mounted); neither service reads/writes it in P1, as required.
- §6.4 Observability hooks — Tasks 4 (logging), 5 (telemetry); each service wires them via `configure_logging` + `setup_observability` in its `main.py` (Tasks 8 and 11).
- §7 Compose topology — Task 14, including the "only router host-exposed" invariant.
- §9 Design decisions — all honored in task choices: `uv` in each service pyproject (Tasks 6, 10), Python 3.11 (every pyproject), `sha256(model.joblib)` as `model_version` (Task 7 `_fingerprint`), model baked into image (Task 9 Dockerfile).
- §10 Risks — mitigated: regression check for root pyproject (Tasks 1, 18); `.env` and artifact gitignored (Task 2); `scikit-learn==1.9.1` pinned (Task 6).
- §11 Acceptance criteria — each of the 9 criteria has a dedicated verification step in Task 18.

**Placeholder scan:** no `TBD` / `TODO` / "appropriate error handling" / "similar to Task N" language; every code step shows the code.

**Type consistency:**
- `ModelBundle.predict` return keys (Task 7) — `team`, `team_confidence`, `top_issues`, `auto_route`, `model_name`, `model_version` — match the fields used by the svm-api route in Task 8 and the `PredictResponse` shape in Task 3.
- `IssueProb(**r)` in Task 8 relies on `{"issue": str, "prob": float}` keys, which is what Task 7's `top_issues` emits.
- Router test expects `error_code = "primary_unhealthy"` (Task 11 test) — that's the literal used in `routes.py` (Task 11 implementation).
- Router's `PRIMARY_URL` env is read in Task 11 `config.py` and set in `docker-compose.yml` (Task 14) + `.env.example` (Task 2).

No inconsistencies found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-11-12-platform-phase-1-foundation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach?
