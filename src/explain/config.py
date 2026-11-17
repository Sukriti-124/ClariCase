"""Narrator_Config: the single configuration source for the Narrator (Req 19).

Every Narrator setting comes from exactly one ``EXPLAIN_*`` variable. The
lookup order is the environment, then st.secrets (Streamlit only), then the
default defined here (Req 19.2, 19.5). An empty or whitespace-only value counts
as unset. No ``AGENTIC_*``, ``AIC_*``, ``GROQ_API_KEY`` or ``OPENAI_API_KEY``
variable is ever read (Req 19.1).

Any invalid value disables the Narrator; the invalid settings are recorded as
``(setting, reason)`` pairs that never contain the value (Req 19.7).

The occlusion settings for the IM2_Explainer are deliberately kept out of
:class:`NarratorConfig` and are read by :func:`occlusion_settings` (Req 19.8).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Literal, Mapping

MAX_RETRIES = 5                    # retry limit (Req 11.12); constant, no env variable
PLATFORM_DEADLINE_S = 110.0        # Narrator_Deadline on the Platform (Req 11.14)
STREAMLIT_DEADLINE_PAD_S = 3.0     # Streamlit Narrator_Deadline = timeout_s + 3 s

# Part of the Record_Hash settings (Req 16.5). The Narrator's prompt builder
# uses this same constant; it lives here so config has no import on narrator.
PROMPT_VERSION = "explain-v1"

BACKENDS = ("openai_compat", "mock")
AUDIENCES = ("consumer", "agent")
MOCK_MODES = ("faithful", "unfaithful", "failure_parse", "failure_timeout")

# setting -> environment variable (exactly the list of Req 19.2)
VARIABLES: dict[str, str] = {
    "backend": "EXPLAIN_BACKEND",
    "base_url": "EXPLAIN_BASE_URL",
    "api_key": "EXPLAIN_API_KEY",
    "model": "EXPLAIN_MODEL",
    "timeout_s": "EXPLAIN_TIMEOUT_S",
    "audiences": "EXPLAIN_AUDIENCES",
    "include_text": "EXPLAIN_INCLUDE_TEXT",
    "agent_token": "EXPLAIN_AGENT_TOKEN",
    "max_tokens": "EXPLAIN_MAX_TOKENS",
    "mock_mode": "EXPLAIN_MOCK_MODE",
    "mock_delay_s": "EXPLAIN_MOCK_DELAY_S",
}

DEFAULTS: dict[str, Any] = {
    "backend": "openai_compat",
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": "",
    "model": "llama-3.3-70b-versatile",
    "timeout_s": 30.0,
    "audiences": AUDIENCES,
    "include_text": True,
    "agent_token": "",
    "max_tokens": 1024,
    "mock_mode": "faithful",
    "mock_delay_s": 0.0,
}

SECRET_SETTINGS = frozenset({"api_key", "agent_token"})

OCCLUSION_ENABLED_VAR = "EXPLAIN_OCCLUSION_ENABLED"
OCCLUSION_K_VAR = "EXPLAIN_OCCLUSION_K"

Source = Literal["env", "st.secrets", "default"]


@dataclass(frozen=True, kw_only=True)
class NarratorConfig:
    backend: str = DEFAULTS["backend"]
    base_url: str = DEFAULTS["base_url"]
    api_key: str = field(default="", repr=False)
    model: str = DEFAULTS["model"]
    timeout_s: float = DEFAULTS["timeout_s"]
    audiences: tuple[str, ...] = DEFAULTS["audiences"]
    include_text: bool = DEFAULTS["include_text"]
    agent_token: str = field(default="", repr=False)
    max_tokens: int = DEFAULTS["max_tokens"]
    mock_mode: str = DEFAULTS["mock_mode"]
    mock_delay_s: float = DEFAULTS["mock_delay_s"]
    sources: dict[str, str] = field(default_factory=lambda: {k: "default" for k in VARIABLES})
    enabled: bool = True
    invalid: tuple[tuple[str, str], ...] = ()

    max_retries: ClassVar[int] = MAX_RETRIES

    def deadline_s(self, serving_path: Literal["platform", "streamlit"]) -> float:
        """Narrator_Deadline length in seconds for a serving path (Req 11.14)."""
        if serving_path == "platform":
            return PLATFORM_DEADLINE_S
        if serving_path == "streamlit":
            return self.timeout_s + STREAMLIT_DEADLINE_PAD_S
        raise ValueError(f"unknown serving path: {serving_path!r}")

    def hash_settings(self) -> dict:
        """Narrator settings that are part of the Record_Hash (Req 16.5)."""
        return {
            "backend": self.backend,
            "model": self.model,
            "audiences": list(self.audiences),
            "include_text": self.include_text,
            "mock_mode": self.mock_mode,
            "prompt_version": PROMPT_VERSION,
        }


# ---------------------------------------------------------------------------
# Lookup and parsing
# ---------------------------------------------------------------------------

def _as_text(value: Any) -> str | None:
    """Normalise a raw env / st.secrets value; empty or blank means unset."""
    if value is None:
        return None
    if isinstance(value, bool):  # TOML booleans in st.secrets
        value = "true" if value else "false"
    text = str(value).strip()
    return text or None


def _secret_get(secrets: Mapping | None, name: str) -> Any:
    if secrets is None:
        return None
    try:
        return secrets.get(name)
    except Exception:  # st.secrets raises when no secrets file exists
        return None


def _lookup(name: str, environ: Mapping[str, str], secrets: Mapping | None) -> tuple[str | None, Source]:
    raw = _as_text(environ.get(name))
    if raw is not None:
        return raw, "env"
    raw = _as_text(_secret_get(secrets, name))
    if raw is not None:
        return raw, "st.secrets"
    return None, "default"


def _parse_number(raw: str) -> float | None:
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _parse_backend(raw: str) -> tuple[Any, str | None]:
    return (raw, None) if raw in BACKENDS else (None, "must be openai_compat or mock")


def _parse_timeout(raw: str) -> tuple[Any, str | None]:
    v = _parse_number(raw)
    if v is None or not 1 <= v <= 120:
        return None, "must be a number from 1 to 120 seconds"
    return v, None


def _parse_audiences(raw: str) -> tuple[Any, str | None]:
    items = {p.strip().lower() for p in raw.split(",") if p.strip()}
    if not items or not items <= set(AUDIENCES):
        return None, "must be a non-empty subset of consumer and agent"
    return tuple(a for a in AUDIENCES if a in items), None


def _parse_bool(raw: str) -> tuple[Any, str | None]:
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true", None
    return None, "must be true or false"


def _parse_max_tokens(raw: str) -> tuple[Any, str | None]:
    try:
        v = int(raw, 10)
    except ValueError:
        return None, "must be an integer from 256 to 4096"
    if not 256 <= v <= 4096:
        return None, "must be an integer from 256 to 4096"
    return v, None


def _parse_mock_mode(raw: str) -> tuple[Any, str | None]:
    if raw in MOCK_MODES:
        return raw, None
    return None, "must be one of faithful, unfaithful, failure_parse, failure_timeout"


def _parse_mock_delay(raw: str) -> tuple[Any, str | None]:
    v = _parse_number(raw)
    if v is None or v < 0:
        return None, "must be a number greater than or equal to 0"
    return v, None


def _parse_any(raw: str) -> tuple[Any, str | None]:
    return raw, None


_PARSERS: dict[str, Callable[[str], tuple[Any, str | None]]] = {
    "backend": _parse_backend,
    "base_url": _parse_any,
    "api_key": _parse_any,
    "model": _parse_any,
    "timeout_s": _parse_timeout,
    "audiences": _parse_audiences,
    "include_text": _parse_bool,
    "agent_token": _parse_any,
    "max_tokens": _parse_max_tokens,
    "mock_mode": _parse_mock_mode,
    "mock_delay_s": _parse_mock_delay,
}


def load_config(environ: Mapping[str, str] | None = None,
                secrets: Mapping | None = None) -> NarratorConfig:
    """Resolve every Narrator setting from ``EXPLAIN_*`` sources (Req 19.2, 19.5, 19.7).

    ``environ`` defaults to ``os.environ``. ``secrets`` is st.secrets (or any
    mapping) and is consulted only when the environment variable is unset.
    """
    env = os.environ if environ is None else environ
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    invalid: list[tuple[str, str]] = []

    for setting, var in VARIABLES.items():
        raw, source = _lookup(var, env, secrets)
        sources[setting] = source
        if raw is None:
            values[setting] = DEFAULTS[setting]
            continue
        value, reason = _PARSERS[setting](raw)
        if reason is not None:
            invalid.append((setting, reason))
            values[setting] = DEFAULTS[setting]
        else:
            values[setting] = value

    if values["backend"] == "openai_compat" and not values["api_key"] \
            and not any(s == "backend" for s, _ in invalid):
        invalid.append(("api_key", "required when backend is openai_compat"))

    return NarratorConfig(**values, sources=sources,
                          enabled=not invalid, invalid=tuple(invalid))


def log_startup(cfg: NarratorConfig, log) -> None:
    """Log the resolved settings and their sources (Req 19.4, 19.7).

    ``log`` is a ``logging.Logger`` (or anything with ``info`` and
    ``warning``). The API key and Agent_Token values are never logged; only
    whether they are configured and where they came from.
    """
    log.info(
        "narrator config: backend=%s (%s) base_url=%s (%s) model=%s (%s) timeout_s=%s (%s)",
        cfg.backend, cfg.sources.get("backend", "default"),
        cfg.base_url, cfg.sources.get("base_url", "default"),
        cfg.model, cfg.sources.get("model", "default"),
        cfg.timeout_s, cfg.sources.get("timeout_s", "default"),
    )
    log.info(
        "narrator config: api_key configured=%s (%s) agent_token configured=%s (%s) enabled=%s",
        "yes" if cfg.api_key else "no", cfg.sources.get("api_key", "default"),
        "yes" if cfg.agent_token else "no", cfg.sources.get("agent_token", "default"),
        cfg.enabled,
    )
    for setting, reason in cfg.invalid:
        log.warning("narrator disabled: invalid setting %s (%s): %s",
                    setting, VARIABLES.get(setting, setting), reason)


def occlusion_settings(environ: Mapping[str, str] | None = None) -> dict:
    """IM2_Explainer occlusion settings; not part of NarratorConfig (Req 19.8).

    Returns ``{"enabled": bool, "k": str | None, "invalid": tuple}``. ``enabled``
    defaults to true; a value other than true/false disables occlusion and is
    recorded in ``invalid`` (without the value). ``k`` is the raw
    ``EXPLAIN_OCCLUSION_K`` text (``None`` when unset) and is validated by the
    IM2_Explainer (Req 5.3).
    """
    env = os.environ if environ is None else environ
    invalid: list[tuple[str, str]] = []
    raw_enabled = _as_text(env.get(OCCLUSION_ENABLED_VAR))
    enabled = True
    if raw_enabled is not None:
        value, reason = _parse_bool(raw_enabled)
        if reason is not None:
            invalid.append((OCCLUSION_ENABLED_VAR, reason))
            enabled = False
        else:
            enabled = value
    return {"enabled": enabled, "k": _as_text(env.get(OCCLUSION_K_VAR)),
            "invalid": tuple(invalid)}
