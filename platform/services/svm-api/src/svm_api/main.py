"""svm-api FastAPI application factory."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from claricase_platform_shared.logging import configure_logging
from claricase_platform_shared.telemetry import setup_observability

from svm_api.config import get_settings
from svm_api.model import load_bundle
from svm_api.routes import router as http_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.service_name)
    app.state.settings = settings
    app.state.bundle = load_bundle(settings.model_path,
                                   settings.baseline2_model_path)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="svm-api", lifespan=lifespan)
    setup_observability(app, service_name="svm-api")
    app.include_router(http_router)
    return app


app = create_app()
