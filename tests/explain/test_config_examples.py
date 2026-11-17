"""Unit tests for Narrator_Config (src/explain/config.py).

Requirements 11.3, 11.14, 19.1, 19.2, 19.4, 19.5, 19.7, 19.8.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

from src.explain import config as C


def test_defaults_with_empty_environment():
    cfg = C.load_config({})
    assert cfg.backend == "openai_compat"
    assert cfg.base_url == "https://api.groq.com/openai/v1"
    assert cfg.model == "llama-3.3-70b-versatile"
    assert cfg.timeout_s == 30.0
    assert cfg.audiences == ("consumer", "agent")
    assert cfg.include_text is True
    assert cfg.agent_token == ""
    assert cfg.max_tokens == 1024
    assert cfg.mock_mode == "faithful"
    assert cfg.mock_delay_s == 0.0
    assert set(cfg.sources.values()) == {"default"}
    # openai_compat without a key disables the Narrator
    assert not cfg.enabled
    assert cfg.invalid == (("api_key", "required when backend is openai_compat"),)


def test_mock_backend_without_key_is_enabled():
    cfg = C.load_config({"EXPLAIN_BACKEND": "mock"})
    assert cfg.enabled and cfg.invalid == ()
    assert cfg.sources["backend"] == "env"


def test_constants_and_deadlines():
    assert C.MAX_RETRIES == 5 and C.NarratorConfig.max_retries == 5
    cfg = C.load_config({"EXPLAIN_BACKEND": "mock", "EXPLAIN_TIMEOUT_S": "12.5"})
    assert cfg.deadline_s("platform") == 110.0
    assert cfg.deadline_s("streamlit") == 15.5
    with pytest.raises(ValueError):
        cfg.deadline_s("other")  # type: ignore[arg-type]


def test_frozen():
    cfg = C.load_config({"EXPLAIN_BACKEND": "mock"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model = "x"  # type: ignore[misc]


def test_empty_string_is_unset_and_precedence():
    secrets = {"EXPLAIN_MODEL": "from-secrets", "EXPLAIN_TIMEOUT_S": 45,
               "EXPLAIN_INCLUDE_TEXT": False, "EXPLAIN_BACKEND": "mock"}
    env = {"EXPLAIN_MODEL": "", "EXPLAIN_TIMEOUT_S": "20"}
    cfg = C.load_config(env, secrets)
    assert (cfg.model, cfg.sources["model"]) == ("from-secrets", "st.secrets")
    assert (cfg.timeout_s, cfg.sources["timeout_s"]) == (20.0, "env")
    assert (cfg.include_text, cfg.sources["include_text"]) == (False, "st.secrets")
    assert cfg.sources["base_url"] == "default"


def test_failing_secrets_mapping_is_ignored():
    class Broken:
        def get(self, name):
            raise FileNotFoundError("no secrets.toml")

    cfg = C.load_config({"EXPLAIN_BACKEND": "mock"}, Broken())
    assert cfg.enabled and cfg.sources["model"] == "default"


def test_foreign_variables_are_ignored():
    env = {"EXPLAIN_BACKEND": "mock", "AGENTIC_MODEL": "other", "AIC_BACKEND": "x",
           "GROQ_API_KEY": "k1", "OPENAI_API_KEY": "k2", "AGENTIC_TIMEOUT_S": "5"}
    assert C.load_config(env) == C.load_config({"EXPLAIN_BACKEND": "mock"})
    # GROQ_API_KEY does not satisfy the openai_compat key requirement
    assert not C.load_config({"GROQ_API_KEY": "k1"}).enabled


@pytest.mark.parametrize("var,value,setting", [
    ("EXPLAIN_BACKEND", "anthropic", "backend"),
    ("EXPLAIN_TIMEOUT_S", "0.5", "timeout_s"),
    ("EXPLAIN_TIMEOUT_S", "121", "timeout_s"),
    ("EXPLAIN_TIMEOUT_S", "nan", "timeout_s"),
    ("EXPLAIN_AUDIENCES", "consumer,admin", "audiences"),
    ("EXPLAIN_AUDIENCES", ",", "audiences"),
    ("EXPLAIN_INCLUDE_TEXT", "yes", "include_text"),
    ("EXPLAIN_MAX_TOKENS", "255", "max_tokens"),
    ("EXPLAIN_MAX_TOKENS", "4097", "max_tokens"),
    ("EXPLAIN_MAX_TOKENS", "1024.0", "max_tokens"),
    ("EXPLAIN_MOCK_MODE", "random", "mock_mode"),
    ("EXPLAIN_MOCK_DELAY_S", "-1", "mock_delay_s"),
])
def test_invalid_values_disable_and_never_record_the_value(var, value, setting):
    cfg = C.load_config({"EXPLAIN_BACKEND": "mock", var: value} if var != "EXPLAIN_BACKEND"
                        else {var: value})
    assert not cfg.enabled
    assert setting in [s for s, _ in cfg.invalid]
    for _, reason in cfg.invalid:
        assert value not in reason


def test_valid_boundaries_and_audience_subset():
    cfg = C.load_config({"EXPLAIN_BACKEND": "mock", "EXPLAIN_TIMEOUT_S": "120",
                         "EXPLAIN_MAX_TOKENS": "256", "EXPLAIN_AUDIENCES": "agent",
                         "EXPLAIN_MOCK_MODE": "failure_timeout",
                         "EXPLAIN_MOCK_DELAY_S": "30"})
    assert cfg.enabled
    assert (cfg.timeout_s, cfg.max_tokens, cfg.audiences) == (120.0, 256, ("agent",))
    assert (cfg.mock_mode, cfg.mock_delay_s) == ("failure_timeout", 30.0)


def test_hash_settings_keys():
    cfg = C.load_config({"EXPLAIN_BACKEND": "mock", "EXPLAIN_API_KEY": "sk-secret"})
    hs = cfg.hash_settings()
    assert set(hs) == {"backend", "model", "audiences", "include_text",
                       "mock_mode", "prompt_version"}
    assert hs["prompt_version"] == C.PROMPT_VERSION
    assert "sk-secret" not in repr(hs) and "sk-secret" not in repr(cfg)


def test_log_startup_never_logs_secrets(caplog):
    key, token = "sk-live-ABCDEF123456", "agent-token-XYZ987"
    cfg = C.load_config({"EXPLAIN_API_KEY": key, "EXPLAIN_AGENT_TOKEN": token,
                         "EXPLAIN_TIMEOUT_S": "999"})
    log = logging.getLogger("test.explain.config")
    with caplog.at_level(logging.INFO, logger="test.explain.config"):
        C.log_startup(cfg, log)
    text = caplog.text
    for needle in ("openai_compat", "https://api.groq.com/openai/v1",
                   "llama-3.3-70b-versatile", "timeout_s", "env", "default",
                   "EXPLAIN_TIMEOUT_S"):
        assert needle in text
    assert "999" not in text
    for secret in (key, token):
        for i in range(len(secret) - 5):
            assert secret[i:i + 6] not in text


def test_occlusion_settings_separate_from_narrator_config():
    assert C.occlusion_settings({}) == {"enabled": True, "k": None, "invalid": ()}
    s = C.occlusion_settings({"EXPLAIN_OCCLUSION_ENABLED": "false",
                              "EXPLAIN_OCCLUSION_K": "abc"})
    assert s["enabled"] is False and s["k"] == "abc"
    bad = C.occlusion_settings({"EXPLAIN_OCCLUSION_ENABLED": "maybe"})
    assert bad["enabled"] is False and bad["invalid"][0][0] == "EXPLAIN_OCCLUSION_ENABLED"
    fields = {f.name for f in dataclasses.fields(C.NarratorConfig)}
    assert not any("occlusion" in f for f in fields)
    # occlusion variables do not change the Narrator config
    base = {"EXPLAIN_BACKEND": "mock"}
    assert C.load_config({**base, "EXPLAIN_OCCLUSION_K": "7"}) == C.load_config(base)
