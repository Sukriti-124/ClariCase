"""ensemble-api configuration (env-driven)."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    service_name: str = Field(default="ensemble-api")
    port: int = Field(default=8000, alias="ENSEMBLE_API_PORT")
    stack_path: Path = Field(
        default=Path("/app/docs/metrics/ensemble_stack/stack.joblib"),
        alias="STACK_PATH",
    )
    # svm-api serves improved_model_1 (+ baseline_model_2) and the issue heads.
    svm_api_url: str = Field(default="http://svm-api:8000", alias="SVM_API_URL")
    # embed-api serves improved_model_2 (+ knn_embed).
    embed_api_url: str = Field(default="http://embed-api:8000",
                               alias="EMBED_API_URL")
    # Keep below the router's PRIMARY_TIMEOUT_S (5 s) so the router gets an
    # answer from us rather than timing out first.
    upstream_timeout_s: float = Field(default=4.0, alias="UPSTREAM_TIMEOUT_S")

    # Label-shift correction (EM over a rolling window of stacker outputs).
    label_shift_enabled: bool = Field(default=True, alias="LABEL_SHIFT_ENABLED")
    prior_window: int = Field(default=5000, alias="PRIOR_WINDOW")
    prior_min_rows: int = Field(default=500, alias="PRIOR_MIN_ROWS")
    prior_refresh: int = Field(default=100, alias="PRIOR_REFRESH")

    # Optional LLM stage (agentic-api). Empty = off. When on, the router's
    # PRIMARY_TIMEOUT_S should allow for one or two extra calls.
    agentic_url: str = Field(default="", alias="AGENTIC_URL")
    agentic_timeout_s: float = Field(default=3.0, alias="AGENTIC_TIMEOUT_S")
    # Rescue rule fitted by src.agentic.evaluate (evaluation.json). Missing
    # file or null threshold: the LLM's team is used but nothing is rescued.
    agentic_policy_path: Path | None = Field(
        default=Path("/app/docs/metrics/agentic/evaluation.json"),
        alias="AGENTIC_POLICY_PATH")
    agentic_issue_gap: float = Field(default=0.3, alias="AGENTIC_ISSUE_GAP")
    agentic_n_candidates: int = Field(default=3, alias="AGENTIC_N_CANDIDATES")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def agentic_enabled(self) -> bool:
        return bool(self.agentic_url.strip())


def get_settings() -> Settings:
    return Settings()
