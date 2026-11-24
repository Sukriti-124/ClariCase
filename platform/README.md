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

The optional [product-first issue backend](services/issue-api/README.md) adds
`POST /classify-issue` (complaint + predicted team) and a combined `POST /predict`
workflow. Start it with the `issues` Compose profile. It uses the original
73-issue catalog and does not change the existing model services.

| Service      | Role                                                  | Host port |
|--------------|-------------------------------------------------------|-----------|
| `router`     | Proxies `/predict` to `PRIMARY_URL` (svm-api)         | 8080      |
| `svm-api`    | Serves `improved_model_1` (TF-IDF + hierarchical SVM) | internal  |
| `postgres`   | Schema loaded; no reads/writes in Phase 1             | internal  |

## Requirements

- Docker + Docker Compose v2
- (for host-side unit tests) Python 3.11+ and the project venv at the repo root
- GNU Make is optional — every `make` target is a one-liner you can run by hand

## Run

```bash
cd platform
cp .env.example .env    # one-time
make up                 # or: docker compose up -d
curl -s http://localhost:8080/healthz
curl -s -X POST http://localhost:8080/predict \
     -H 'content-type: application/json' \
     -d '{"text":"The bank reported wrong information to the credit bureau after multiple disputes."}'
```

## Everyday commands

```bash
make build    # build images (or: docker compose build)
make up       # start stack (or: docker compose up -d)
make down     # stop stack, keep data volume (or: docker compose down)
make logs     # tail logs (or: docker compose logs -f)
make test     # host unit tests for shared + router (svm-api tests run in Docker; see below)
make smoke    # end-to-end smoke (compose up + /predict + assert + compose down -v)
make clean    # down + remove data volume (or: docker compose down -v)
```

## Testing

- **shared** (`platform/shared/tests`): 10 host tests, run with `pytest`.
- **router** (`platform/services/router/tests`): 6 host tests, run with `pytest`.
- **svm-api** (`platform/services/svm-api/tests`): 8 tests. **Not runnable on host** when the host venv is Python 3.14 — `scikit-learn==1.9.1` + `numpy<2` have no cp314 wheels. Run in-container:
  ```bash
  docker compose run --rm svm-api \
    bash -c "cd /app && pytest platform/services/svm-api/tests/ -v"
  ```
  (Or point `PYTHONPATH=/app` and invoke from any cwd — the Dockerfile sets this.)

## Observability

Phase 1 wires observability hooks into every service but does **not** deploy a
backend for them. You can inspect them manually:

- `/metrics` on each service — Prometheus exposition format. From the host:
  ```bash
  docker compose exec svm-api curl -s localhost:8000/metrics | head
  ```
- Structured JSON logs — `docker compose logs svm-api` emits one JSON line per
  event with `service`, `request_id`, `level`, `timestamp`, `event`.
- OpenTelemetry — SDK is imported and FastAPI is auto-instrumented. The
  exporter is a no-op unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

Phase 5 (Prometheus + Grafana), Phase 6 (Loki), and Phase 7 (Tempo) each connect a backend; no code change in this service is required.

## Ensemble profile (opt-in)

Two more model services sit behind the compose profile `ensemble`. Plain `docker compose up` does not start them, so the Phase 1 slice is unchanged.

| Service        | Serves                                                          | Host port |
|----------------|-----------------------------------------------------------------|-----------|
| `svm-api`      | `improved_model_1` (team + issues) and, co-hosted, `baseline_model_2` | internal |
| `embed-api`    | `improved_model_2` and, co-hosted, the `knn_embed` neighbour vote | internal |
| `ensemble-api` | Stacker + gate + thresholds + label-shift correction            | internal  |

How a request flows with `PRIMARY_URL=http://ensemble-api:8000`:

```
router -> ensemble-api -+-> svm-api   /predict  (improved_model_1 + baseline_model_2 probabilities, issues)
                        +-> embed-api /predict  (improved_model_2 + knn_embed probabilities)
                        '-> svm-api   /issues   only when the stacker's team differs from svm-api's
```

What `ensemble-api` returns, on top of the Phase 1 fields: `scenario` (`auto`, `assisted`, `contested` or `degraded`), `stack_team_probs` (the stacker's probability for every team, corrected for label shift when that is active) and `team_probs` (svm-api's own). `team_confidence` is the gate score, the estimated probability that the chosen team is right, and `auto_route` is that score against the chosen team's threshold.

What happens when something is down:

- `embed-api` unreachable: the answer still comes back, routed by the primary-alone policy that was validated offline, with `scenario: degraded` and an `x-missing-models` header. Nothing is silently auto-routed by an ungated model.
- `svm-api` unreachable: 503 `primary_unhealthy`, as the router does. Without it there are no issues and no primary probabilities.
- `ensemble-api` itself down (timeout, connection error or 5xx): if `FALLBACK_URL` is set (for example `http://svm-api:8000`), the router retries the request once there, with the same `request_id`, and marks the response `X-ClariCase-Served-By: fallback`. If the fallback is also down the router returns 503 `all_models_unhealthy`. A 4xx from the primary is returned as is: the request was rejected, so another model would reject it too. With `FALLBACK_URL` empty, behaviour is unchanged (503 `primary_unhealthy`).
- The stacker picks a team other than svm-api's and the `/issues` call fails: the team is returned with an empty issue list, never the other team's issues.

Label shift: the team mix of incoming complaints can differ from the validation mix the models were calibrated to (on the Jan to Feb 2026 holdout, Credit Reporting fell from 61% to 2%). `ensemble-api` keeps the last `PRIOR_WINDOW` stacker outputs, re-estimates the mix from them with no labels (EM, Saerens et al. 2002) every `PRIOR_REFRESH` requests once `PRIOR_MIN_ROWS` have been seen, and reweights the stacker's probabilities. The correction is bounded to a factor of 50 per team. `GET /prior` shows the live estimate, `POST /prior/reset` clears it, and the `label_shift_prior` gauge exposes it to Prometheus. Set `LABEL_SHIFT_ENABLED=false` to serve the stacker exactly as evaluated offline.

Artifacts the images and volumes need, all produced from the repo root:

```bash
python -m src.improved_model_2.train
python -m src.ensemble.export_predictions --splits val test holdout_2026   # also writes data/cache/knn_embed/index.npz
python -m src.ensemble.train_stack --eval-splits test holdout_2026         # writes docs/metrics/ensemble_stack/stack.joblib
```

Then:

```bash
make up-ensemble        # router pointed at ensemble-api
make smoke-ensemble     # build, start, one /predict through the router, /issues, /prior, tear down
```

Host tests for the three model services use fake encoders, a synthetic stack bundle and mocked HTTP, so they need neither torch nor the real artifacts (svm-api's tests do load the real `improved_model_1` artifact and skip without it):

```bash
pip install -e platform/services/svm-api -e platform/services/embed-api -e platform/services/ensemble-api
python -m pytest platform/services/svm-api/tests platform/services/embed-api/tests platform/services/ensemble-api/tests -v
```

The `embed-api` image bakes in the sentence encoder named by `ENCODER_NAME`. It must be the encoder the model and the neighbour index were built with; the service refuses to start on a mismatch.

Metrics specific to the ensemble: `routing_decisions_total{scenario}`, `routing_team_total{team,auto_route}`, `gate_score`, `stack_overrides_total`, `issue_refetch_total{result}`, `upstream_failures_total{service}`, `label_shift_prior{team}`, `label_shift_active`.

**Plan and results:** `../docs/superpowers/plans/2026-11-15-routing-gate-and-model-services.md`

## Not in Phase 1

- Kubernetes (Phase 2)
- Other model services — seq2seq-api, agentic-api (Phase 3)
- Shadow mode + Redis queue (Phase 4)
- Prometheus / Grafana / Loki / Tempo deployed (Phases 5–7)
- Next.js frontend (Phase 8)
- Nightly eval CronJob (Phase 9)
- Cloud deploy on DOKS (Phase 10)
- Any change to `main`, the existing Streamlit app, or `src/app/`
