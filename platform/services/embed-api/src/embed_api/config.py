"""embed-api configuration (env-driven)."""
import os
from pathlib import Path
from typing import Mapping

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.explain.config import occlusion_settings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    service_name: str = Field(default="embed-api")
    port: int = Field(default=8000, alias="EMBED_API_PORT")
    model_path: Path = Field(
        default=Path("/app/docs/metrics/improved_model_2/model.joblib"),
        alias="EMBED_MODEL_PATH",
    )
    device: str | None = Field(default=None, alias="EMBED_DEVICE")
    # Nearest-neighbour index (written by src.ensemble.export_predictions,
    # mounted into the container). Optional: without it, no knn_embed vote.
    knn_index_path: Path | None = Field(
        default=Path("/app/data/cache/knn_embed/index.npz"),
        alias="KNN_INDEX_PATH",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def get_settings() -> Settings:
    return Settings()


def get_occlusion_settings(environ: Mapping[str, str] | None = None) -> dict:
    """IM2 occlusion settings from EXPLAIN_OCCLUSION_ENABLED / _K (Req 19.8).

    Read through ``src.explain.config.occlusion_settings`` so embed-api and
    the Streamlit path share one parser; the defaults (enabled, K = 20) live
    in the IM2_Explainer, not here.
    """
    return occlusion_settings(os.environ if environ is None else environ)
