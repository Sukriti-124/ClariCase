"""router configuration (env-driven)."""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    service_name: str = Field(default="router")
    port: int = Field(default=8080, alias="ROUTER_PORT")
    primary_url: str = Field(
        default="http://svm-api:8000",
        alias="PRIMARY_URL",
    )
    primary_timeout_s: float = Field(default=5.0, alias="PRIMARY_TIMEOUT_S")
    # Optional second model service, tried once when the primary times out,
    # is unreachable or returns 5xx. Empty disables fallback (Phase 1
    # behaviour: the router returns 503).
    fallback_url: str = Field(default="", alias="FALLBACK_URL")
    fallback_timeout_s: float = Field(default=5.0, alias="FALLBACK_TIMEOUT_S")
    # Host of the Explain_Endpoint (GET /explain/{request_id}), Req 17.9.
    ensemble_api_url: str = Field(
        default="http://ensemble-api:8000",
        alias="ENSEMBLE_API_URL",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def get_settings() -> Settings:
    return Settings()
