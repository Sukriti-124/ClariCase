"""agentic-api HTTP routes."""
from __future__ import annotations

import time
import uuid

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from prometheus_client import Counter, Histogram

from claricase_platform_shared.contracts import ChooseRequest, ChooseResponse
from claricase_platform_shared.logging import bind_request_id

router = APIRouter()
log = structlog.get_logger()

CHOICES = Counter("agentic_choices_total",
                  "LLM choices by task and whether they kept the ML's first candidate.",
                  ["task", "kept_first"])
CACHE_HITS = Counter("agentic_cache_total", "Prompt cache outcome.", ["hit"])
UNPARSED = Counter("agentic_unparsed_total", "Answers with no candidate id.", ["task"])
LLM_LATENCY = Histogram("agentic_latency_seconds", "End-to-end choice latency.",
                        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8))


@router.get("/healthz")
def healthz(request: Request) -> dict:
    eng = getattr(request.app.state, "engine", None)
    if eng is None:
        raise HTTPException(status_code=503, detail="engine_not_loaded")
    return {"status": "ok", "model": eng.model_name, "pool_size": len(eng.pool),
            "cache_answers": len(eng.chooser.cache)}


@router.post("/choose", response_model=ChooseResponse)
async def choose(body: ChooseRequest, request: Request) -> ChooseResponse:
    rid = body.request_id or str(uuid.uuid4())
    bind_request_id(rid)
    eng = request.app.state.engine
    t0 = time.perf_counter()
    try:
        ch = await run_in_threadpool(eng.choose, body.text, body.task,
                                     body.candidates, body.team)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e))
    latency = time.perf_counter() - t0
    LLM_LATENCY.observe(latency)
    CHOICES.labels(task=body.task,
                   kept_first=str(ch.candidate == body.candidates[0]).lower()).inc()
    CACHE_HITS.labels(hit=str(ch.cached).lower()).inc()
    if not ch.parsed:
        UNPARSED.labels(task=body.task).inc()
    log.info("choice_served", task=body.task, candidates=body.candidates,
             choice=ch.candidate, parsed=ch.parsed, cached=ch.cached,
             latency_ms=int(latency * 1000))
    return ChooseResponse(model_name=eng.model_name, model_version=eng.version,
                          task=body.task, choice=ch.candidate, parsed=ch.parsed,
                          cached=ch.cached, request_id=rid,
                          latency_ms=int(latency * 1000))
