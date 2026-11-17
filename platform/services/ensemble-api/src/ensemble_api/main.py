"""ensemble-api FastAPI application factory."""
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from claricase_platform_shared.logging import configure_logging
from claricase_platform_shared.telemetry import setup_observability

from ensemble_api.agentic import load_rescue_threshold
from ensemble_api.config import get_settings
from ensemble_api.routes import router as http_router
from ensemble_api.stack_model import load_stack


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.service_name)
    app.state.settings = settings
    app.state.stack = load_stack(
        settings.stack_path, window=settings.prior_window,
        min_rows=settings.prior_min_rows, refresh=settings.prior_refresh,
        label_shift=settings.label_shift_enabled)
    app.state.rescue_threshold = (
        load_rescue_threshold(settings.agentic_policy_path)
        if settings.agentic_enabled else None)
    app.state.explain_tasks = set()
    app.state.explain_cfg = None
    app.state.explain_cache = None
    app.state.explain_narrator = None
    app.state.explain_store = None
    if settings.explanations_enabled:
        from src.explain.cache import ExplanationCache
        from src.explain.config import load_config, log_startup
        from src.explain.narrator import Narrator
        from src.explain.store import ExplanationStore

        cfg = load_config()
        log_startup(cfg, logging.getLogger("ensemble_api.explain"))
        app.state.explain_cfg = cfg
        app.state.explain_cache = ExplanationCache()
        app.state.explain_narrator = Narrator(
            cfg, explanation_cache=app.state.explain_cache)
        app.state.explain_store = ExplanationStore(
            ttl_s=settings.explain_store_ttl_s,
            max_ids=settings.explain_store_max)
    app.state.http = httpx.AsyncClient()
    try:
        yield
    finally:
        await app.state.http.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="ensemble-api", lifespan=lifespan)
    setup_observability(app, service_name="ensemble-api")
    app.include_router(http_router)
    return app


app = create_app()
