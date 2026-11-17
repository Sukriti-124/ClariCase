"""router FastAPI application factory."""
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from claricase_platform_shared.logging import configure_logging
from claricase_platform_shared.telemetry import setup_observability

from router.config import get_settings
from router.routes import router as http_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.service_name)
    app.state.settings = settings
    app.state.http = httpx.AsyncClient()
    try:
        yield
    finally:
        await app.state.http.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="router", lifespan=lifespan)
    setup_observability(app, service_name="router")
    app.include_router(http_router)
    return app


app = create_app()
