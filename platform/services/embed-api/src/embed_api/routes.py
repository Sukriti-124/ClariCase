"""embed-api HTTP routes."""
from __future__ import annotations

import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from prometheus_client import Counter, Histogram

from claricase_platform_shared.contracts import PredictRequest, PredictResponse
from claricase_platform_shared.logging import bind_request_id

router = APIRouter()
log = structlog.get_logger()

# Same metric names as svm-api on purpose: dashboards compare model services
# by the Prometheus job/service label rather than by metric name.
PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total predictions served by embed-api, labeled by predicted team and routing outcome.",
    ["team", "auto_route"],
)
MODEL_CONFIDENCE = Histogram(
    "model_confidence",
    "Distribution of team-level confidence scores from embed-api predictions.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)


@router.get("/healthz")
def healthz(request: Request) -> dict:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model_not_loaded")
    return {"status": "ok", "extra_models": list(bundle.extra_models)}


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    rid = body.request_id or str(uuid.uuid4())
    bind_request_id(rid)

    bundle = request.app.state.bundle
    t0 = time.perf_counter()
    raw: dict[str, Any] = bundle.predict(body.text, top_k=3)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    PREDICTIONS_TOTAL.labels(
        team=raw["team"], auto_route=str(raw["auto_route"]).lower(),
    ).inc()
    MODEL_CONFIDENCE.observe(raw["team_confidence"])
    log.info(
        "predict_served",
        model=raw["model_name"],
        team=raw["team"],
        team_confidence=raw["team_confidence"],
        auto_route=raw["auto_route"],
        latency_ms=latency_ms,
    )
    return PredictResponse(**raw, request_id=rid, latency_ms=latency_ms)
