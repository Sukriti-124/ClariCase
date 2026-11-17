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
import hmac
import time
import uuid

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from prometheus_client import Counter, Gauge, Histogram

from claricase_platform_shared.contracts import (ExplainSectionsRequest,
                                                 ExplainSectionsResponse,
                                                 IssuesResponse,
                                                 PredictRequest,
                                                 PredictResponse)
from claricase_platform_shared.logging import bind_request_id

from ensemble_api import agentic as A

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
AGENTIC_CALLS = Counter("agentic_calls_total",
                        "Calls to agentic-api by task and outcome.",
                        ["task", "outcome"])  # changed | kept | failed
RESCUED = Counter("agentic_rescued_total",
                  "Review complaints auto-routed after the LLM agreed with the stacker.")


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


_SERVICE_SECTIONS = {
    "svm-api": ("improved_model_1", "baseline_model_2"),
    "embed-api": ("improved_model_2", "knn_embed"),
}
EXPLAIN_NARRATE_TIMEOUT_S = 120


def _outcome_dict(d, issues, agentic_info) -> dict:
    """The served decision in the Decision_Record's Outcome shape."""
    from src.explain import labels as L
    from src.explain.labels import display_label
    tables = L.load_tables()
    by_label = {e.label: (i, e) for i, e in tables.issues.items()}
    refs = []
    for n, it in enumerate(issues[:3], 1):
        if it.issue in tables.issues:
            iid, e = it.issue, tables.issues[it.issue]
        elif it.issue in by_label:
            iid, e = by_label[it.issue]
        else:
            iid, e = it.issue, None
        label = e.label if e else it.issue
        refs.append({"rank": n, "issue_id": iid, "label": label,
                     "display_label": e.display_label if e
                     else display_label(label), "prob": it.prob})
    probs = [r["prob"] for r in refs]
    ordered = all(a >= b for a, b in zip(probs, probs[1:]))
    return {
        "chosen_team": {"id": d.team,
                        "name": tables.teams.get(d.team, d.team)},
        "top_issues": refs,
        "issue_order": ("probability" if ordered or not (
            agentic_info and "issue_choice" in agentic_info) else "agentic"),
        "auto_route": bool(d.auto_route),
        "route": "auto" if d.auto_route else "review",
        "scenario": d.scenario,
    }


async def _fetch_explain(client, url: str, payload: dict, timeout: float,
                         service: str):
    from src.explain.pipeline import UpstreamFailure
    try:
        resp = await client.post(f"{url}/explain", json=payload,
                                 timeout=timeout)
        if resp.status_code != 200:
            raise ValueError(resp.status_code)
        return ExplainSectionsResponse(**resp.json())
    except Exception:
        return UpstreamFailure(service)


async def explain_platform_task(app, ctx, text: str, rid: str, outcome: dict,
                                model_versions: dict, timings: dict) -> None:
    """Background: build the sections, store the record, start narration.
    Never raises and never touches the served response."""
    try:
        from src.app.pii import mask
        from src.explain.pipeline import (UpstreamFailure, explain_platform,
                                          narrate_and_store)
        st = app.state
        s = st.settings
        masked = mask(text)
        req = ExplainSectionsRequest(
            text=text, masked_text=masked, chosen_team=outcome["chosen_team"]["id"],
            served_issues=[i["issue_id"] for i in outcome["top_issues"]],
            request_id=rid).model_dump()
        missing = set(ctx.missing_models or ())
        svm_r, emb_r = await asyncio.gather(
            _fetch_explain(st.http, s.svm_api_url, req, s.upstream_timeout_s,
                           "svm-api"),
            _fetch_explain(st.http, s.embed_api_url, req, s.upstream_timeout_s,
                           "embed-api"))
        upstream: dict = {}
        versions = dict(model_versions)
        for service, r in (("svm-api", svm_r), ("embed-api", emb_r)):
            if isinstance(r, UpstreamFailure):
                for name in _SERVICE_SECTIONS[service]:
                    upstream[name] = r
            else:
                upstream.update(r.sections)
                versions.update(r.model_versions)
        sections = await asyncio.to_thread(explain_platform, ctx, upstream)
        narrator = st.explain_narrator
        deadline = narrator.clock() + st.explain_cfg.deadline_s("platform")
        coro = asyncio.wait_for(asyncio.to_thread(
            narrate_and_store, store=st.explain_store, narrator=narrator,
            request_id=rid, serving_path="platform", model_versions=versions,
            outcome=outcome, sections=sections, raw_text=text,
            masked_text=masked, deadline=deadline, timings=timings),
            EXPLAIN_NARRATE_TIMEOUT_S)
        task = asyncio.create_task(coro)
        st.explain_tasks.add(task)

        def _done(t):
            st.explain_tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                log.warning("explain_narration_failed", request_id=rid,
                            error=type(t.exception()).__name__)
        task.add_done_callback(_done)
    except Exception as exc:
        log.warning("explain_task_failed", request_id=rid,
                    error=type(exc).__name__)


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
    out = {"status": "ok", "scenario": stack.scenario_kind,
           "models": stack.models,
           "svm_api": svm, "embed_api": emb if emb else "unreachable",
           "label_shift_active": stack.prior.active}
    ncfg = getattr(request.app.state, "explain_cfg", None)
    out["narrator"] = {"enabled": bool(getattr(ncfg, "enabled", False)),
                       "backend": getattr(ncfg, "backend", None),
                       "model": getattr(ncfg, "model", None)}
    if s.agentic_enabled:
        ag = await _up(s.agentic_url)
        out["agentic_api"] = ag if ag else "unreachable"
        out["rescue_threshold"] = request.app.state.rescue_threshold
    return out


AGENT_TOKEN_HEADER = "X-ClariCase-Agent-Token"
EXPLAIN_WATCHDOG_S = 120
WATCHDOG_REASON = "explanation timed out after 120 s"


def _is_agent(request: Request) -> bool:
    cfg = getattr(request.app.state, "explain_cfg", None)
    token = getattr(cfg, "agent_token", "") or ""
    if not token:
        return False
    sent = request.headers.get(AGENT_TOKEN_HEADER) or ""
    return hmac.compare_digest(sent.encode("utf-8"), token.encode("utf-8"))


@router.get("/explain/{request_id}")
def explain_get(request_id: str, request: Request) -> JSONResponse:
    store = getattr(request.app.state, "explain_store", None)
    entry = store.get(request_id) if store is not None else None
    if entry is None:
        return JSONResponse(status_code=404, content={
            "error_code": "explanation_not_found", "request_id": request_id})
    if (entry.status == "pending"
            and store._clock() - entry.inserted_at > EXPLAIN_WATCHDOG_S):
        store.mark_failed(request_id, WATCHDOG_REASON)
        entry = store.get(request_id) or entry
        entry.status = "failed"
        entry.failure = {"reason": WATCHDOG_REASON}
    status = entry.status
    body: dict = {"request_id": request_id, "explanation_status": status}
    ex = entry.explanation
    reason = (entry.failure or {}).get("reason")
    if status == "ready" and ex is not None:
        body["consumer_summary"] = ex.consumer_summary
    elif status == "failed" and not reason:
        reason = "explanation failed"
    if _is_agent(request):
        if ex is not None and status == "ready":
            body.update(
                agent_summary=ex.agent_summary,
                cited_facts=[f.model_dump(mode="json") for f in ex.cited_facts],
                source=ex.source, validation=ex.validation.model_dump(mode="json"))
            reason = reason or (ex.fallback_reason if ex.source == "template"
                                else None)
        body["reason"] = reason
        body["decision_record"] = (entry.record.model_dump(mode="json")
                                   if entry.record is not None else None)
        body["audit"] = [a.model_dump(mode="json") for a in entry.audit]
    elif status == "failed":
        body["reason"] = reason
    return JSONResponse(status_code=202 if status == "pending" else 200,
                        content=body)


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
    PRIOR_ACTIVE.set(1 if d.prior_active else 0)
    if d.prior_active:
        for team, p in zip(stack.teams, stack.prior.prior):
            PRIOR_ESTIMATE.labels(team=team).set(float(p))

    pre_llm_team, pre_llm_auto = d.team, bool(d.auto_route)

    # ---- LLM stage (optional) ----
    agentic_info: dict | None = None
    if s.agentic_enabled and not missing:
        agentic_info = {}
        if not d.auto_route and d.candidates and len(d.candidates) > 1:
            cands = d.candidates[:s.agentic_n_candidates]
            ch = await A.ask(client, s.agentic_url, body.text, "team", cands,
                             None, rid, s.agentic_timeout_s)
            agentic_info["team_candidates"] = cands
            if ch is None:
                AGENTIC_CALLS.labels(task="team", outcome="failed").inc()
            else:
                agentic_info["team_choice"] = ch.choice
                thr = request.app.state.rescue_threshold
                rescued = (ch.choice == d.team and thr is not None
                           and d.score >= thr)
                AGENTIC_CALLS.labels(
                    task="team",
                    outcome="kept" if ch.choice == d.team else "changed").inc()
                if rescued:
                    RESCUED.inc()
                    d.auto_route, d.scenario = True, "auto"
                agentic_info["rescued"] = rescued
                d.team = ch.choice

    DECISIONS.labels(scenario=d.scenario).inc()
    CHOSEN_TEAM.labels(team=d.team, auto_route=str(d.auto_route).lower()).inc()

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

    if agentic_info is not None and issues:
        cands = A.issue_candidates(issues, s.agentic_issue_gap,
                                   s.agentic_n_candidates)
        if cands:
            ch = await A.ask(client, s.agentic_url, body.text, "issue", cands,
                             d.team, rid, s.agentic_timeout_s)
            agentic_info["issue_candidates"] = cands
            if ch is None:
                AGENTIC_CALLS.labels(task="issue", outcome="failed").inc()
            else:
                agentic_info["issue_choice"] = ch.choice
                AGENTIC_CALLS.labels(
                    task="issue",
                    outcome="kept" if ch.choice == cands[0] else "changed").inc()
                issues = A.reorder_issues(issues, ch.choice)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    log.info("predict_served", model=stack.model_name, team=d.team,
             gate_score=round(d.score, 4), threshold=d.threshold,
             auto_route=d.auto_route, scenario=d.scenario,
             model_teams=d.model_teams, primary_team=svm.team,
             missing_models=missing, prior_active=d.prior_active,
             agentic=agentic_info, n_issues=len(issues),
             latency_ms=latency_ms)

    out = PredictResponse(
        model_name=stack.model_name, model_version=stack.version,
        team=d.team, team_confidence=min(max(d.score, 0.0), 1.0),
        top_issues=issues, auto_route=d.auto_route, request_id=rid,
        latency_ms=latency_ms, scenario=d.scenario,
        team_probs=svm.team_probs,
        stack_team_probs=(None if d.stack_probs is None else
                          dict(zip(stack.teams,
                                   [float(x) for x in d.stack_probs]))),
        agentic=agentic_info or None,
    )
    headers = {"x-routing-scenario": d.scenario}
    if missing:
        headers["x-missing-models"] = ",".join(missing)

    # ---- Explanation (after latency_ms; never affects the response) ----
    task = None
    status = "unavailable"
    st = request.app.state
    if getattr(st, "explain_store", None) is not None:
        try:
            from src.explain.explainers.routing import PlatformDecisionContext
            ctx = PlatformDecisionContext(
                decision=d, bundle=stack, pre_llm_team=pre_llm_team,
                pre_llm_auto=pre_llm_auto, missing_models=tuple(missing),
                agentic_info=agentic_info, rescue_threshold=getattr(
                    st, "rescue_threshold", None),
                text=body.text, request_id=rid, probs=probs,
                served_issues=tuple(i.issue for i in issues),
                svm_team=svm.team)
            outcome = _outcome_dict(d, issues, agentic_info)
            versions = {"ensemble": stack.version}
            task = BackgroundTask(
                explain_platform_task, request.app, ctx, body.text, rid,
                outcome, versions, {"predict_ms": latency_ms})
            status = "pending"
        except Exception as exc:
            log.warning("explain_setup_failed", request_id=rid,
                        error=type(exc).__name__)
            task = None
    content = out.model_dump()
    content["explanation_status"] = status
    return JSONResponse(status_code=200, content=content, headers=headers,
                        background=task)
