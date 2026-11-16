"""ensemble-api HTTP routes.

POST /predict
  1. Ask svm-api and embed-api in parallel. Each returns its own team
     probabilities plus those of the model it co-hosts (baseline_model_2,
     knn_embed), so one call per container gives all four model outputs.
  2. The stack bundle decides team, gate score, auto_route and scenario.
  3. Issues come from svm-api's response when its own top team is the
     chosen team (about 9 in 10 requests); otherwise one more call to
     svm-api's /issues for the chosen team.
  4. If embed-api is unavailable the primary-alone policy applies
     (scenario "degraded"). If svm-api is unavailable the answer is 503,
     because without it there are neither team probabilities for the
     primary nor issues.

GET /prior shows the live label-shift estimate; POST /prior/reset clears it.
"""
from __future__ import annotations

import asyncio
import time
import uuid

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram

from claricase_platform_shared.contracts import (IssuesResponse,
                                                 PredictRequest,
                                                 PredictResponse)
from claricase_platform_shared.logging import bind_request_id

router = APIRouter()
log = structlog.get_logger()

DECISIONS = Counter("routing_decisions_total",
                    "Routing decisions by scenario.", ["scenario"])
CHOSEN_TEAM = Counter("routing_team_total",
                      "Chosen team by whether it was auto-routed.",
                      ["team", "auto_route"])
GATE_SCORE = Histogram(
    "gate_score", "Gate score: estimated P(chosen team is correct).",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0))
TEAM_OVERRIDES = Counter(
    "stack_overrides_total",
    "Requests where the stacker's team differed from the primary model's.")
ISSUE_REFETCH = Counter("issue_refetch_total",
                        "Extra /issues calls to svm-api.", ["result"])
UPSTREAM_FAIL = Counter("upstream_failures_total",
                        "Model service calls that failed.", ["service"])
PRIOR_ESTIMATE = Gauge("label_shift_prior",
                       "Current estimated team prior.", ["team"])
PRIOR_ACTIVE = Gauge("label_shift_active",
                     "1 when the rolling prior estimate is in use.")


async def _post(client: httpx.AsyncClient, url: str, payload: dict,
                timeout: float, model):
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        return model(**resp.json())
    except (httpx.HTTPError, ValueError):
        return None


def _collect(resp: PredictResponse | None, own_name: str) -> dict:
    """Model name -> team probabilities from one service's response."""
    out = {}
    if resp is None:
        return out
    if resp.team_probs:
        out[own_name] = resp.team_probs
    for name, probs in (resp.extra_team_probs or {}).items():
        out[name] = probs
    return out


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    client: httpx.AsyncClient = request.app.state.http
    s = request.app.state.settings

    async def _up(url: str):
        try:
            r = await client.get(f"{url}/healthz", timeout=2.0)
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    svm, emb = await asyncio.gather(_up(s.svm_api_url), _up(s.embed_api_url))
    if svm is None:
        raise HTTPException(status_code=503, detail="svm_api_unreachable")
    stack = request.app.state.stack
    return {"status": "ok", "scenario": stack.scenario_kind,
            "models": stack.models,
            "svm_api": svm, "embed_api": emb if emb else "unreachable",
            "label_shift_active": stack.prior.active}


@router.get("/prior")
def prior(request: Request) -> dict:
    stack = request.app.state.stack
    p = stack.prior
    return {"active": p.active, "n_seen": p.n_seen, "n_estimates": p.n_estimates,
            "source_prior": dict(zip(stack.teams, p.source_prior.round(4).tolist())),
            "estimated_prior": dict(zip(stack.teams, p.prior.round(4).tolist()))}


@router.post("/prior/reset")
def prior_reset(request: Request) -> dict:
    request.app.state.stack.prior.reset()
    PRIOR_ACTIVE.set(0)
    log.info("prior_reset")
    return {"status": "reset"}


@router.post("/predict")
async def predict(body: PredictRequest, request: Request) -> JSONResponse:
    rid = body.request_id or str(uuid.uuid4())
    bind_request_id(rid)
    client: httpx.AsyncClient = request.app.state.http
    s = request.app.state.settings
    stack = request.app.state.stack
    payload = {"text": body.text, "request_id": rid}
    t0 = time.perf_counter()

    svm, emb = await asyncio.gather(
        _post(client, f"{s.svm_api_url}/predict", payload,
              s.upstream_timeout_s, PredictResponse),
        _post(client, f"{s.embed_api_url}/predict", payload,
              s.upstream_timeout_s, PredictResponse))
    if svm is None:
        UPSTREAM_FAIL.labels(service="svm-api").inc()
        return JSONResponse(status_code=503, content={
            "error_code": "primary_unhealthy", "request_id": rid})
    if emb is None:
        UPSTREAM_FAIL.labels(service="embed-api").inc()

    probs = {**_collect(svm, "improved_model_1"),
             **_collect(emb, "improved_model_2")}
    missing = [m for m in stack.models if m not in probs]
    n_words = len(body.text.split())
    if missing:
        primary = probs.get(stack.primary) or svm.team_probs
        if not primary:
            return JSONResponse(status_code=503, content={
                "error_code": "primary_has_no_probabilities",
                "request_id": rid})
        d = stack.fallback(primary)
    else:
        d = stack.decide(probs, n_words)
        GATE_SCORE.observe(d.score)
        if d.team != svm.team:
            TEAM_OVERRIDES.inc()
    DECISIONS.labels(scenario=d.scenario).inc()
    CHOSEN_TEAM.labels(team=d.team, auto_route=str(d.auto_route).lower()).inc()
    PRIOR_ACTIVE.set(1 if d.prior_active else 0)
    if d.prior_active:
        for team, p in zip(stack.teams, stack.prior.prior):
            PRIOR_ESTIMATE.labels(team=team).set(float(p))

    # Issues under the chosen team.
    issues = svm.top_issues
    if d.team != svm.team:
        iss = await _post(client, f"{s.svm_api_url}/issues",
                          {"text": body.text, "team": d.team, "top_k": 3,
                           "request_id": rid},
                          s.upstream_timeout_s, IssuesResponse)
        if iss is None:
            ISSUE_REFETCH.labels(result="failed").inc()
            issues = []
        else:
            ISSUE_REFETCH.labels(result="ok").inc()
            issues = iss.top_issues

    latency_ms = int((time.perf_counter() - t0) * 1000)
    log.info("predict_served", model=stack.model_name, team=d.team,
             gate_score=round(d.score, 4), threshold=d.threshold,
             auto_route=d.auto_route, scenario=d.scenario,
             model_teams=d.model_teams, primary_team=svm.team,
             missing_models=missing, prior_active=d.prior_active,
             n_issues=len(issues), latency_ms=latency_ms)

    out = PredictResponse(
        model_name=stack.model_name, model_version=stack.version,
        team=d.team, team_confidence=min(max(d.score, 0.0), 1.0),
        top_issues=issues, auto_route=d.auto_route, request_id=rid,
        latency_ms=latency_ms, scenario=d.scenario,
        team_probs=svm.team_probs,
        stack_team_probs=(None if d.stack_probs is None else
                          dict(zip(stack.teams,
                                   [float(x) for x in d.stack_probs]))),
    )
    headers = {"x-routing-scenario": d.scenario}
    if missing:
        headers["x-missing-models"] = ",".join(missing)
    return JSONResponse(status_code=200, content=out.model_dump(),
                        headers=headers)
