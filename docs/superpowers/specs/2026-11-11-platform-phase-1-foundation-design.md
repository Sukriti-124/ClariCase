# Platform Phase 1 — Foundation + First Vertical Slice

**Date:** 2026-11-11
**Branch:** `week-11-designing-a-system`
**Status:** Draft design, pending user review
**Scope:** Phase 1 of the multi-phase platform rebuild. Phase 1 only. Later phases are listed for context but out of scope here.

---

## 1. Purpose and context

ClariCase currently runs as a single Streamlit app on Streamlit Community Cloud, calling `improved_model_1` (hierarchical TF-IDF + Linear SVM) in-process. The goal of the platform rebuild is to replace that deployment with a containerized multi-service system on Kubernetes, with full-stack observability (metrics, logs, traces) and an active+shadow routing pattern across three model families.

This is a hybrid portfolio + hands-on-learning build. Success looks like:

- A working multi-service system you can demo locally on `kind` and later deploy to DigitalOcean Kubernetes (DOKS).
- Real exposure to Docker, Kubernetes primitives, Prometheus/Grafana/Loki/Tempo, and inter-service patterns.
- No disruption to the current `main` branch, existing Streamlit app, or existing training pipelines.

The full target architecture and phase decomposition are summarized in Section 8. **This spec covers Phase 1 only** — the foundation and first vertical slice on `docker-compose`.

## 2. Phase 1 goal

Stand up one end-to-end vertical slice locally on `docker-compose`:

```
curl → router → svm-api → prediction response
                  ↓
              postgres (schema present, unused)
```

With observability *hooks* wired in every service — Prometheus `/metrics`, structured JSON logs, OTel SDK imported — but not yet connected to a backend (that's Phases 5–7).

### Explicit non-goals for Phase 1

- Kubernetes — Phase 2.
- Shadow mode and Redis — Phase 4.
- Other model services (`seq2seq-api`, `agentic-api`) — Phase 3.
- Next.js frontend — Phase 8.
- Nightly evaluation CronJob — Phase 9.
- Deployed Prometheus / Grafana / Loki / Tempo — Phases 5–7.
- Cloud deployment — Phase 10.
- Touching `main`, `streamlit_app.py`, `src/improved_model_1/train.py`, or the existing `src/app/` storage layer.

## 3. Scope of changes

**New files** live under a new top-level `platform/` directory on branch `week-11-designing-a-system`.

**Existing-code touch points (minimal):**
- A new root `pyproject.toml` declaring `src/` as an installable package named `claricase-models` so `svm-api` can import model code without duplication. Existing `python -m pytest` and existing imports must continue to work unchanged.
- No other existing files are modified.

## 4. Directory layout

```
platform/
  README.md                      # how to run the slice locally
  Makefile                       # build / up / down / logs / test / smoke / clean
  docker-compose.yml
  .env.example                   # all env vars documented
  shared/                        # importable Python library, used by every service
    __init__.py
    contracts.py                 # pydantic request/response models
    logging.py                   # structlog JSON config, request_id contextvar
    telemetry.py                 # Prometheus middleware + OTel SDK scaffold
  services/
    router/
      Dockerfile
      pyproject.toml             # uv-managed
      src/
        main.py                  # FastAPI app factory
        config.py                # env-driven settings (PRIMARY_URL, PORT, etc.)
        routes.py                # POST /predict, GET /healthz, GET /metrics
      tests/
        test_routes.py
    svm-api/
      Dockerfile
      pyproject.toml             # uv-managed; depends on claricase-models from repo root
      src/
        main.py
        model.py                 # loads model.joblib, prediction wrapper
        routes.py                # POST /predict, GET /healthz, GET /metrics
      tests/
        test_predict.py
      # model.joblib is COPIED into the image at build time from
      # docs/metrics/improved_model_1/model.joblib (see section 9)
  db/
    schema/
      001_init.sql               # complaints, predictions tables
```

Root-level additions:
- `pyproject.toml` (new, minimal — declares `src/` as package `claricase-models`)

Nothing else in the repository is changed.

## 5. Shared HTTP contract

Every model-serving service (including future `seq2seq-api` and `agentic-api`) implements this contract. Freezing it in Phase 1 means later phases drop in without API renegotiation.

```
POST /predict
  Request:
    { "text": "<complaint text, min 10 chars>",
      "request_id": "<optional uuid; service generates if absent>" }

  Response (200):
    { "model_name": "improved_model_1",
      "model_version": "<git-sha or artifact-hash>",
      "team": "<team code>",
      "team_confidence": 0.0 .. 1.0,
      "top_issues": [ { "issue": "<label>", "prob": 0.0 .. 1.0 }, ... ],
      "auto_route": true | false,
      "request_id": "<uuid>",
      "latency_ms": <int> }

  Error responses:
    400 — validation (missing/short text)
    500 — internal error during prediction
    503 — service not ready (model not loaded)

GET /healthz
  200 — model loaded, ready to serve
  503 — not ready

GET /metrics
  Prometheus exposition format
```

Pydantic models for this contract live in `platform/shared/contracts.py` and are imported by every service.

## 6. Service behaviors (Phase 1)

### 6.1 Router (`services/router/`)

- FastAPI app, exposes the same `/predict`, `/healthz`, `/metrics` contract as above (acting as a proxy).
- Reads `PRIMARY_URL` from env (default `http://svm-api:8000`).
- On `POST /predict`:
  1. If `request_id` missing, generate a uuid4.
  2. Bind `request_id` into the logging contextvar.
  3. Call `POST {PRIMARY_URL}/predict` with the request (using `httpx.AsyncClient`, timeout 5 s).
  4. On 2xx: return primary's response body unchanged, with `router_latency_ms` added as an HTTP response header.
  5. On 5xx, timeout, or connection error: return `503 { "error_code": "primary_unhealthy", "request_id": "..." }`. **No fallback in Phase 1** — that's shadow territory in Phase 4.
- `/healthz` returns 200 if the router can reach `PRIMARY_URL/healthz`, else 503.

### 6.2 svm-api (`services/svm-api/`)

- FastAPI app.
- On startup: loads `model.joblib` (copied into the image, see Section 9). If load fails, the process exits — Docker's restart policy handles it.
- `POST /predict`:
  1. Validate request via pydantic (min length 10).
  2. Delegate to the existing prediction logic imported from `claricase-models` (originally `src/improved_model_1/predict.py`). We call into the existing function; we do not re-implement.
  3. Build the response per the shared contract. `model_version` is the sha256 of `model.joblib`, computed at startup.
- `/healthz`: 200 once model is loaded.
- `/metrics`: standard Prometheus exposition (see 6.4).

### 6.3 Postgres (`db/`)

- Image: `postgres:16-alpine`.
- Named volume `platform_pgdata` for persistence.
- Init script `db/schema/001_init.sql` creates two tables:
  - `complaints` — mirrors enough of `src/app/schema.sql` to support eventual migration, but **not used by any service in Phase 1**.
  - `predictions` — for shadow logs in Phase 4. Defined now so the schema contract is locked early.
- Credentials come from `.env` (never committed). Defaults in `.env.example`.

The schema will be defined in detail in the file itself; the design-time commitment is simply: both tables exist with stable primary keys (`id uuid`), creation timestamps, and `request_id` on predictions.

### 6.4 Observability hooks (installed, not yet connected)

Every service wires three hooks on day one:

1. **Logging** — `structlog` configured to emit JSON lines to stdout. Every log line carries `service`, `request_id` (from contextvar), `env`. In Phase 6 a Loki/Promtail DaemonSet will scrape stdout — no code change required.
2. **Metrics** — `prometheus_client` FastAPI middleware. Standard counters/histograms: `http_requests_total{method,route,status}`, `http_request_duration_seconds{method,route}`. svm-api adds `model_predictions_total{team,auto_route}` and `model_confidence` histogram. `/metrics` is live from day one. Phase 5 adds a Prometheus to scrape it.
3. **Tracing** — OTel SDK imported, FastAPI auto-instrumented. Exporter defaults to a no-op; configured to OTLP only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Phase 7 provides the endpoint. Zero cost now; retrofit cost would be high.

All three are implemented once in `platform/shared/telemetry.py` and `platform/shared/logging.py`, and each service calls a single `setup_observability(app, service_name)` in its FastAPI factory.

## 7. docker-compose topology

```yaml
services:
  postgres:
    image: postgres:16-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./db/schema:/docker-entrypoint-initdb.d:ro
    healthcheck: pg_isready
  svm-api:
    build: ./services/svm-api
    env_file: .env
    depends_on:
      postgres: { condition: service_healthy }
    healthcheck: curl -f http://localhost:8000/healthz
  router:
    build: ./services/router
    env_file: .env
    environment:
      - PRIMARY_URL=http://svm-api:8000
    depends_on:
      svm-api: { condition: service_healthy }
    ports:
      - "8080:8000"
volumes:
  pgdata:
```

Only the router exposes a host port. svm-api is reachable only on the compose network (as it should be).

## 8. Target architecture and phase context (for reference)

Phase 1 is step one of ten:

| # | Phase | Scope |
|---|---|---|
| **1** | **Foundation + first vertical slice (this spec)** | Monorepo structure, router + svm-api + Postgres on docker-compose. |
| 2 | K8s on kind | Kustomize base + local overlay, StatefulSet for Postgres, Ingress, probes. |
| 3 | Add seq2seq-api + agentic-api | All three model services in cluster, router config-selectable. |
| 4 | Shadow mode + Redis queue | Fan-out, shadow-worker, predictions table used. |
| 5 | Observability — metrics (Prometheus + Grafana) | Scrape /metrics, provisioned dashboards. |
| 6 | Observability — logs (Loki + Promtail) | Aggregate stdout, link from metrics panels. |
| 7 | Observability — traces (Tempo + OTel Collector) | Distributed traces across router→worker→DB. |
| 8 | Next.js frontend | SPA, submit/tracking/shadow-debug views. |
| 9 | Nightly eval CronJob | Reads predictions, writes eval_runs, trend panel. |
| 10 | DOKS cloud deploy | Cloud overlay, managed Postgres, Sealed Secrets. |

**Deferred explicitly (not in v1 of this platform):** model-registry-triggered holdout eval, drift detection (PSI/KL), A/B percentage routing, ensemble routing, mTLS between services, OPA policies, NetworkPolicies, CI/CD. Each is a candidate for a future phase.

## 9. Key design decisions and rationale

| Decision | Choice | Rationale |
|---|---|---|
| Branch | `week-11-designing-a-system` (existing on GitHub) | User directive — reuse the already-set-up branch. |
| Service topology | 3 model microservices + router (Phase 3 brings 2 and 3) | Depends diverge (scikit vs torch vs LLM client); separate Dockerfiles make K8s meaningful. |
| Router behavior | Primary-only in P1; active+shadow added in P4 | Ship a working slice first; shadow needs Redis + worker which is a phase of its own. |
| Data layer | Postgres StatefulSet (later) + Redis queue (later) | Self-hosted to maximize K8s learning; Redis queue decouples primary from slow shadows. |
| Observability | Full three pillars, phased metrics → logs → traces | User chose maximum learning depth; phasing avoids big-bang deployment. |
| Frontend | Next.js SPA, Phase 8 | User choice; deferred until contract + obs are stable so frontend work isn't redone. |
| Local K8s | kind | Fastest to spin up, closest to real K8s APIs. |
| Cloud | DOKS (Phase 10) | Cheap managed K8s; design manifests cluster-agnostic via Kustomize overlays. |
| Code reuse | Root `pyproject.toml` installs `src/` as `claricase-models` | Avoids copy-paste; keeps existing `python -m pytest` working. |
| Python dep manager | `uv` inside each service | Fast, modern, lockfiles. |
| Python version | 3.11 across all services | Stable, broad library support. |
| Model artifact | Baked into svm-api image at build time (P1 only) | Simplest for docker-compose; Phase 2 graduates to PVC or init-container download. |
| `model_version` value | `sha256` of `model.joblib`, computed at service startup | Deterministic, derivable from the artifact alone, and stable across restarts of the same image. |
| Observability code location | `platform/shared/` imported by every service | One implementation of logging/metrics/tracing setup; each service calls one function. |

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Root `pyproject.toml` breaks existing `python -m pytest` or imports. | Add it minimally (only declaring `src/` as a package); run the existing test suite as the first check in Phase 1's implementation plan. |
| Model `.joblib` size bloats the svm-api image. | Expected size is manageable for Phase 1 (joblib is tens of MB). Phase 2 graduates to a mounted artifact strategy. |
| HTTP contract chosen now turns out wrong for `agentic-api` or `seq2seq-api`. | The contract is intentionally generic — `top_issues` is a list of dicts, `model_version` is a string, both accommodate different model families. Reviewed against existing predict scripts for all three model families before freezing. |
| `scikit-learn==1.9.1` pin in the existing repo may conflict with newer deps. | svm-api's `pyproject.toml` pins the same version. Isolation via container means no conflict with other services. |
| Developers accidentally commit model artifacts or `.env`. | `.gitignore` extended under `platform/` to exclude both; `.env.example` committed with placeholders. |

## 11. Acceptance criteria for Phase 1

Phase 1 is complete when all of the following are true:

1. `platform/` directory exists with the layout from Section 4 on branch `week-11-designing-a-system`.
2. `make build && make up` brings up three healthy containers (`postgres`, `svm-api`, `router`) with no errors in logs.
3. `make smoke` passes: it POSTs a sample complaint to `http://localhost:8080/predict`, receives a 200 response conforming to the schema in Section 5, and verifies `model_name == "improved_model_1"`.
4. `make test` runs per-service unit tests; all green.
5. `docker compose exec svm-api curl -s http://localhost:8000/metrics` returns a Prometheus exposition including `model_predictions_total` and `model_confidence`. (svm-api is not exposed to the host; verification happens inside the compose network.)
6. svm-api and router log structured JSON to stdout with `request_id` threaded end-to-end on a single request.
7. Existing `python -m pytest -q -m "not slow"` from the repo root still passes (no regression from the root `pyproject.toml` addition).
8. `main` branch and `streamlit_app.py` are unchanged.
9. `platform/README.md` documents how to run the slice and what each service does.

## 12. What this spec does not decide

These are deferred to the implementation plan (`writing-plans` phase) or to later specs:

- Exact file-by-file task breakdown and sequence.
- Specific library versions (beyond `scikit-learn==1.9.1` and Python `3.11`).
- Exact `docker-compose.yml` healthcheck intervals and retry policies.
- The exact SQL of `001_init.sql` beyond "`complaints` and `predictions` tables with `id uuid`, `created_at`, and (for predictions) `request_id`."
- How Phase 2 ports the baked-in model artifact to a K8s-friendly strategy (PVC vs init-container).

---

**Review checklist for the user:** Does this spec's Phase 1 scope, directory layout, HTTP contract, and acceptance criteria match what you want built first? Anything to add, cut, or reshape before we move to the implementation plan?
