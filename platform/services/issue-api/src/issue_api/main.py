"""Standalone issue service; no encoder, training data, or GPU required."""
import asyncio
import uuid
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram

from claricase_platform_shared.contracts import PredictResponse
from claricase_platform_shared.logging import bind_request_id, configure_logging
from claricase_platform_shared.telemetry import setup_observability
from issue_api.catalog import Catalog
from issue_api.classifier import BusyError, Classifier, InvalidDecision, PROMPT_VERSION
from issue_api.config import Settings
from issue_api.contracts import ClassifyRequest, ComplaintRequest, IssueResult, ProductIssueResponse

log = structlog.get_logger()
OUTCOMES = Counter("issue_classifications_total", "Issue decisions and failures", ["outcome"])
LATENCY = Histogram("issue_classification_seconds", "Issue classification latency")


def create_app(settings: Settings | None = None, *, transport=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        cfg = settings or Settings()
        if cfg.backend != "mock" and (not cfg.model or not cfg.api_key.get_secret_value()):
            raise RuntimeError("Set ISSUE_MODEL and ISSUE_API_KEY, or ISSUE_BACKEND=mock")
        configure_logging("issue-api")
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            app.state.settings = cfg
            app.state.client = client
            app.state.classifier = Classifier(cfg, Catalog(cfg.catalog_dir), client)
            yield

    app = FastAPI(title="ClariCase product-conditioned issue API", lifespan=lifespan)
    setup_observability(app, service_name="issue-api")

    @app.get("/healthz")
    def health(request: Request):
        engine = getattr(request.app.state, "classifier", None)
        if engine is None:
            raise HTTPException(503, "not_ready")
        return {"status": "ok", "backend": engine.settings.backend,
                "model": engine.settings.model or "mock", "teams": len(engine.catalog.teams),
                "issues": len(engine.catalog.issues), "catalog_version": engine.catalog.version,
                "prompt_version": PROMPT_VERSION}

    @app.get("/catalog")
    def catalog(request: Request, team: str | None = None):
        registry = request.app.state.classifier.catalog
        if team is not None and team not in registry.teams:
            raise HTTPException(422, "unknown_team")
        return {"catalog_version": registry.version,
                "products": [registry.view(t) for t in ([team] if team else sorted(registry.teams))]}

    async def classify(body: ClassifyRequest, request: Request) -> IssueResult:
        rid = body.request_id or str(uuid.uuid4())
        bind_request_id(rid)
        body = body.model_copy(update={"request_id": rid})
        try:
            result = await request.app.state.classifier.classify(body)
        except KeyError:
            raise HTTPException(422, "unknown_team")
        except BusyError:
            OUTCOMES.labels("busy").inc()
            raise HTTPException(503, "issue_service_busy", headers={"Retry-After": "1"})
        except (TimeoutError, httpx.TimeoutException):
            OUTCOMES.labels("timeout").inc()
            raise HTTPException(504, "issue_provider_timeout")
        except InvalidDecision:
            OUTCOMES.labels("invalid_response").inc()
            raise HTTPException(502, "invalid_issue_response")
        except httpx.HTTPError:
            OUTCOMES.labels("provider_error").inc()
            raise HTTPException(502, "issue_provider_unavailable")
        OUTCOMES.labels(result.status).inc()
        LATENCY.observe(result.latency_ms / 1000)
        log.info("issue_classified", request_id=rid, team=result.team,
                 issue_id=result.issue_id, status=result.status, latency_ms=result.latency_ms)
        return result

    app.post("/classify-issue", response_model=IssueResult)(classify)

    @app.post("/predict", response_model=ProductIssueResponse)
    async def predict(body: ComplaintRequest, request: Request):
        cfg = request.app.state.settings
        if not cfg.product_url:
            raise HTTPException(503, "product_service_not_configured")
        rid = body.request_id or str(uuid.uuid4())
        bind_request_id(rid)
        try:
            upstream = await asyncio.wait_for(request.app.state.client.post(
                cfg.product_url.rstrip("/") + "/predict",
                json={"text": body.text, "request_id": rid}, timeout=cfg.product_timeout_s),
                timeout=cfg.product_timeout_s)
            upstream.raise_for_status()
            prediction = PredictResponse.model_validate(upstream.json())
            if prediction.team not in request.app.state.classifier.catalog.teams:
                raise ValueError("unknown product")
            if prediction.request_id != rid:
                raise ValueError("upstream request correlation mismatch")
        except (TimeoutError, httpx.TimeoutException):
            raise HTTPException(504, "product_provider_timeout")
        except (httpx.HTTPError, ValueError):
            raise HTTPException(502, "invalid_product_response")
        result = await classify(ClassifyRequest(text=body.text, team=prediction.team,
                                               request_id=rid), request)
        return ProductIssueResponse(team_prediction=prediction, issue_result=result)

    return app


app = create_app()
