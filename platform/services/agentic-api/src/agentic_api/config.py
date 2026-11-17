"""agentic-api configuration (env-driven). LLM settings come from
src.agentic.config (AGENTIC_BACKEND, AGENTIC_BASE_URL, AGENTIC_API_KEY,
AGENTIC_MODEL), shared with the offline runs so the cache is one store."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    service_name: str = Field(default="agentic-api")
    port: int = Field(default=8000, alias="AGENTIC_API_PORT")
    # Example pool (parquet + npy stem) written by src.agentic.examples.
    pool_path: Path = Field(default=Path("/app/data/cache/agentic/example_pool"),
                            alias="AGENTIC_POOL_PATH")
    # improved_model_2's bundle: names the encoder the pool was built with.
    embed_model_path: Path = Field(
        default=Path("/app/docs/metrics/improved_model_2/model.joblib"),
        alias="EMBED_MODEL_PATH")
    llm_cache_path: Path | None = Field(
        default=Path("/app/data/cache/agentic/llm_cache.sqlite"),
        alias="AGENTIC_LLM_CACHE")
    device: str | None = Field(default=None, alias="EMBED_DEVICE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def get_settings() -> Settings:
    return Settings()
