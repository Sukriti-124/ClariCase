"""ensemble-api FastAPI application factory."""
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from claricase_platform_shared.logging import configure_logging
from claricase_platform_shared.telemetry import setup_observability

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
