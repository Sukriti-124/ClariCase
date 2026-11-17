"""Unit tests for Narrator.narrate (task 20.2).

**Validates: Requirements 11.1, 11.8, 11.9, 11.11, 11.12, 13.9, 13.10, 16.1, 16.7, 20.1, 20.5**
"""
from __future__ import annotations

import time

import pytest

from src.agentic.llm import PromptCache
from src.explain.cache import ExplanationCache
from src.explain.config import NarratorConfig
from src.explain.narrator import Narrator
from tests.explain.test_validator_unit import TABLES, real_record


def _cfg(**kw) -> NarratorConfig:
    kw.setdefault("backend", "mock")
    return NarratorConfig(**kw)


def _rec():
    return real_record(sorted(TABLES.issues)[0])


def _narrate(cfg, rec=None, **kw):
    n = Narrator(cfg, explanation_cache=kw.pop("ec", None), prompt_cache=kw.pop("pc", None))
    return n, n.narrate(rec or _rec(), "masked text", deadline=time.monotonic() + 100)


def test_faithful_is_llm_and_cached():
    ec = ExplanationCache()
    n, out = _narrate(_cfg(mock_mode="faithful"), ec=ec)
    assert out.explanation.source == "llm" and out.explanation.fallback_reason is None
    assert out.backend_calls == 1 and len(out.audit) == 1 and len(ec) == 1
    out2 = n.narrate(_rec(), "masked text", deadline=time.monotonic() + 100)
    assert out2.backend_calls == 0 and out2.explanation.source == "llm"


def test_unfaithful_falls_back_and_deletes_prompt_cache():
    pc = PromptCache(None)
    deleted = []
    pc.delete = lambda m, p: deleted.append((m, p))
    ec = ExplanationCache()
    _, out = _narrate(_cfg(mock_mode="unfaithful"), ec=ec, pc=pc)
    assert out.explanation.source == "template"
    assert out.explanation.fallback_reason == "validator_rejection"
    assert out.explanation.validation.failed and len(ec) == 0 and len(deleted) == 1
    assert out.audit[0].raw_reply and out.audit[0].explanation_source == "template"


def test_failure_parse():
    _, out = _narrate(_cfg(mock_mode="failure_parse"))
    assert out.explanation.fallback_reason == "parse_error"
    assert out.audit[0].parse_error


def test_failure_timeout():
    _, out = _narrate(_cfg(mock_mode="failure_timeout"))
    assert out.explanation.fallback_reason == "timeout"
    assert out.audit[0].failure_reason.startswith("timeout") and out.audit[0].raw_reply is None


def test_no_api_key_and_invalid_config_skip_backend():
    _, out = _narrate(_cfg(backend="openai_compat", api_key=""))
    assert out.explanation.fallback_reason == "no_api_key" and out.backend_calls == 0
    _, out = _narrate(_cfg(enabled=False))
    assert out.explanation.fallback_reason == "invalid_config" and out.backend_calls == 0


def test_deadline_passed_is_timeout_without_call():
    n = Narrator(_cfg())
    out = n.narrate(_rec(), None, deadline=time.monotonic() - 1)
    assert out.explanation.fallback_reason == "timeout" and out.backend_calls == 0


def test_record_not_modified():
    rec = _rec()
    before = rec.model_dump_json()
    Narrator(_cfg()).narrate(rec, "t", deadline=time.monotonic() + 100)
    assert rec.model_dump_json() == before
