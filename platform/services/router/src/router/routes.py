"""router HTTP routes."""
from __future__ import annotations

import uuid
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
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


AGENT_TOKEN_HEADER = "X-ClariCase-Agent-Token"


@router.get("/explain/{request_id}")
async def explain(request_id: str, request: Request) -> Response:
    """Forward GET /explain/{request_id} to ensemble-api (Req 15.8, 17.9).

    The agent token header is forwarded unchanged and is never logged or
    echoed. The upstream status and body pass through unchanged.
    """
    client: httpx.AsyncClient = request.app.state.http
    settings = request.app.state.settings

    headers: dict[str, str] = {}
    token = request.headers.get(AGENT_TOKEN_HEADER)
    if token is not None:
        headers[AGENT_TOKEN_HEADER] = token

    url = f"{settings.ensemble_api_url}/explain/{quote(request_id, safe='')}"
    try:
        resp = await client.get(url, headers=headers, timeout=settings.primary_timeout_s)
    except httpx.HTTPError:
        return JSONResponse(
            status_code=503,
            content={
                "error_code": "explanation_service_unavailable",
                "message": "The explanation service is unavailable.",
                "request_id": request_id,
            },
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )
