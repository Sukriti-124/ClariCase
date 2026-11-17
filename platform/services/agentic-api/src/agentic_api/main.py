"""agentic-api FastAPI application factory."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from claricase_platform_shared.logging import configure_logging
from claricase_platform_shared.telemetry import setup_observability

from agentic_api.config import get_settings
from agentic_api.engine import load_engine
from agentic_api.routes import router as http_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.service_name)
    app.state.settings = settings
    app.state.engine = load_engine(settings.pool_path, settings.embed_model_path,
                                   settings.llm_cache_path, settings.device)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="agentic-api", lifespan=lifespan)
    setup_observability(app, service_name="agentic-api")
    app.include_router(http_router)
    return app


app = create_app()
