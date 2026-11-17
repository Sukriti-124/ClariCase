"""svm-api HTTP routes."""
from __future__ import annotations

import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from prometheus_client import Counter, Histogram

from claricase_platform_shared.contracts import (
    ExplainSectionsRequest,
    ExplainSectionsResponse,
    IssueProb,
    IssuesRequest,
    IssuesResponse,
    PredictRequest,
    PredictResponse,
)
from claricase_platform_shared.logging import bind_request_id

router = APIRouter()
log = structlog.get_logger()

# Custom model-level metrics (complement the auto-wired http_* metrics).
# Scraped by Prometheus (Phase 5) and rendered in Grafana dashboards.
PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total predictions served by svm-api, labeled by predicted team and routing outcome.",
    ["team", "auto_route"],
)
MODEL_CONFIDENCE = Histogram(
    "model_confidence",
    "Distribution of team-level confidence scores from svm-api predictions.",
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
        team=raw["team"],
        auto_route=str(raw["auto_route"]).lower(),
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

    return PredictResponse(
        model_name=raw["model_name"],
        model_version=raw["model_version"],
        team=raw["team"],
        team_confidence=raw["team_confidence"],
        top_issues=[IssueProb(**r) for r in raw["top_issues"]],
        auto_route=raw["auto_route"],
        request_id=rid,
        latency_ms=latency_ms,
        team_probs=raw["team_probs"],
        extra_team_probs=raw.get("extra_team_probs"),
    )


@router.post("/issues", response_model=IssuesResponse)
def issues(body: IssuesRequest, request: Request) -> IssuesResponse:
    """Rank issues inside a given team. Used by ensemble-api when its chosen
    team differs from this service's own top team."""
    rid = body.request_id or str(uuid.uuid4())
    bind_request_id(rid)
    bundle = request.app.state.bundle
    t0 = time.perf_counter()
    try:
        rows = bundle.issues(body.text, body.team, top_k=body.top_k)
    except KeyError:
        raise HTTPException(status_code=422, detail=f"unknown team {body.team}")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    log.info("issues_served", team=body.team, n=len(rows), latency_ms=latency_ms)
    return IssuesResponse(
        model_name=bundle.model_name, model_version=bundle.version,
        team=body.team, top_issues=[IssueProb(**r) for r in rows],
        request_id=rid, latency_ms=latency_ms,
    )


BM2_NOT_LOADED = "baseline_model_2 not loaded"


@router.post("/explain", response_model=ExplainSectionsResponse)
def explain(body: ExplainSectionsRequest, request: Request) -> ExplainSectionsResponse:
    """Internal: the improved_model_1 and baseline_model_2 explanation sections
    for a complaint ensemble-api has already routed (design: platform
    integration). Explainer failures become ``error`` sections; the route
    itself only fails if the model bundle is missing."""
    # Imported here so /predict and /issues never depend on src.explain.
    from src.explain.explainers import run_explainer, skipped_section
    from src.explain.explainers.bm2 import explain_bm2
    from src.explain.explainers.im1 import explain_im1

    bind_request_id(body.request_id)
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model_not_loaded")

    t0 = time.perf_counter()
    raw = bundle.raw
    im1 = run_explainer(
        "improved_model_1", explain_im1,
        raw["vectorizer"], raw["model"], raw.get("team_classes"),
        body.text, body.masked_text, body.chosen_team,
        served_issues=body.served_issues,
        feature_names=bundle.im1_feature_names,
    )
    versions = {"improved_model_1": bundle.version}
    if bundle.baseline2 is None:
        bm2 = skipped_section("baseline_model_2", BM2_NOT_LOADED)
    else:
        bm2 = run_explainer(
            "baseline_model_2", explain_bm2,
            bundle.baseline2, body.text, body.masked_text, body.chosen_team,
            feature_names=bundle.bm2_feature_names,
        )
        versions["baseline_model_2"] = bundle.baseline2_version

    latency_ms = int((time.perf_counter() - t0) * 1000)
    log.info("explain_served", im1_status=im1.status, bm2_status=bm2.status,
             latency_ms=latency_ms)
    return ExplainSectionsResponse(
        sections={
            "improved_model_1": im1.model_dump(mode="json"),
            "baseline_model_2": bm2.model_dump(mode="json"),
        },
        model_versions=versions,
    )
