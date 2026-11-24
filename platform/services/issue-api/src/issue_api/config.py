from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ISSUE_", extra="ignore")

    catalog_dir: Path = Path("data/processed")
    backend: Literal["openai_compat", "mock"] = "openai_compat"
    base_url: str = "https://api.groq.com/openai/v1"
    api_key: SecretStr = SecretStr("")
    model: str = ""
    response_format: Literal["json_schema", "json_object", "none"] = "json_object"
    timeout_s: float = Field(default=25, gt=0, le=120)
    max_output_tokens: int = Field(default=800, ge=128, le=4096)
    concurrency: int = Field(default=4, ge=1, le=64)
    # URL of the product/team model service, not this service itself.
    product_url: str = ""
    product_timeout_s: float = Field(default=10, gt=0, le=120)
