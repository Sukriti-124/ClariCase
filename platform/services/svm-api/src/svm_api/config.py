"""svm-api configuration (env-driven)."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    service_name: str = Field(default="svm-api")
    port: int = Field(default=8000, alias="SVM_API_PORT")
    model_path: Path = Field(
        default=Path("/app/docs/metrics/improved_model_1/model.joblib"),
        alias="MODEL_PATH",
    )
    # Optional co-hosted second opinion (TF-IDF + Naive Bayes). Served in the
    # same response under extra_team_probs when the artifact is present.
    baseline2_model_path: Path | None = Field(
        default=Path("/app/docs/metrics/baseline_model_2/model.joblib"),
        alias="BASELINE2_MODEL_PATH",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def get_settings() -> Settings:
    return Settings()
