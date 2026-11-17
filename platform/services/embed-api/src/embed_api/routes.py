"""embed-api HTTP routes."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException, Request
from prometheus_client import Counter, Histogram

from claricase_platform_shared.contracts import (
    ExplainSectionsRequest,
    ExplainSectionsResponse,
    PredictRequest,
    PredictResponse,
)
from claricase_platform_shared.logging import bind_request_id
from src.ensemble.probs import TEAMS, reorder
from src.explain.explainers import error_section, run_explainer
from src.explain.explainers.im2 import NAME as IM2_NAME, explain_im2
from src.explain.explainers.knn import NAME as KNN_NAME, explain_knn, skipped_no_index

from embed_api.config import get_occlusion_settings

router = APIRouter()
log = structlog.get_logger()

# At most one explanation uses the encoder at a time, so occlusion batches
# never pile up against /predict (design: platform integration, embed-api).
_EXPLAIN_SEMAPHORE = threading.Semaphore(1)

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


def _knn_version(knn: Any) -> str:
    """Cheap identifier for the loaded index (hashing ~200 MB per start is
    not worth it); changes whenever the encoder, k or row count changes."""
    return (f"knn:{knn.encoder_name}:len{knn.max_seq_length}"
            f":k{knn.k}:n{knn.n}")


def _team_probs(bundle: Any, q: np.ndarray) -> dict[str, float]:
    P = reorder(bundle.raw["classifier"].predict_proba(q[None, :]),
                bundle.raw["team_classes"], TEAMS)[0]
    return {t: float(p) for t, p in zip(TEAMS, P)}


@router.post("/explain", response_model=ExplainSectionsResponse)
def explain(body: ExplainSectionsRequest, request: Request) -> ExplainSectionsResponse:
    """Internal: improved_model_2 and knn_embed sections for a complaint
    ensemble-api has already routed. Never changes a prediction; any
    explainer failure becomes an ``error`` section, not an HTTP error."""
    bind_request_id(body.request_id)
    bundle = request.app.state.bundle
    occlusion = getattr(request.app.state, "occlusion", None) \
        or get_occlusion_settings()
    t0 = time.perf_counter()

    sections: dict[str, Any] = {}
    q: np.ndarray | None = None
    with _EXPLAIN_SEMAPHORE:
        try:
            # Re-embed the text /predict scored, so the served probabilities
            # and the kNN query are the ones the ensemble used.
            q = np.asarray(bundle.encoder.encode([body.text]))[0]
            served = _team_probs(bundle, q)
        except Exception as exc:
            sections[IM2_NAME] = error_section(IM2_NAME, exc)
            if bundle.knn is not None:
                sections[KNN_NAME] = error_section(KNN_NAME, exc)
        else:
            same_text = body.masked_text == body.text
            sections[IM2_NAME] = run_explainer(
                IM2_NAME, explain_im2, bundle.raw["classifier"],
                bundle.raw["team_classes"], bundle.encoder, body.masked_text,
                body.chosen_team, served, occlusion["enabled"], occlusion["k"],
                same_text, q)

    if bundle.knn is None:
        sections[KNN_NAME] = skipped_no_index()
    elif KNN_NAME not in sections:
        knn = bundle.knn

        def _knn_section():
            sims, idx = knn.neighbours(q)
            served_knn = {t: float(p) for t, p in zip(TEAMS, knn.team_vote(q))}
            return explain_knn(sims, idx, knn.team_idx, knn.issue_idx,
                               knn.issues, knn.complaint_ids, knn.k, knn.n,
                               served_knn)

        sections[KNN_NAME] = run_explainer(KNN_NAME, _knn_section)

    versions = {IM2_NAME: bundle.version}
    if bundle.knn is not None:
        versions[KNN_NAME] = _knn_version(bundle.knn)

    log.info("explain_served",
             statuses={k: s.status for k, s in sections.items()},
             latency_ms=int((time.perf_counter() - t0) * 1000))
    return ExplainSectionsResponse(
        sections={k: s.model_dump(mode="json") for k, s in sections.items()},
        model_versions=versions)
