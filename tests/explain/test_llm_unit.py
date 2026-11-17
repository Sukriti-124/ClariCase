"""Unit tests for the free-text completion, deadline handling and
PromptCache.delete in src/agentic/llm.py (task 18.1).

**Validates: Requirements 11.2, 11.3, 11.8, 11.12, 11.14**
"""
from __future__ import annotations

import json

import httpx
import pytest

from src.agentic import llm as L


def _backend(handler, clock, max_retries=5, timeout_s=30.0):
    return L.OpenAICompatBackend(
        base_url="http://llm.test/v1", api_key="k", model="m",
        timeout_s=timeout_s, max_retries=max_retries,
        clock=clock.monotonic, sleep=clock.sleep,
        transport=httpx.MockTransport(handler))


def _ok(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_complete_text_payload_and_raw_reply(fake_clock):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return _ok('  {"a": 1}\n')

    b = _backend(handler, fake_clock)
    assert b.complete_text("hi", max_tokens=1024) == '  {"a": 1}\n'
    assert seen[0]["temperature"] == 0 and seen[0]["max_tokens"] == 1024
    assert seen[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_complete_payload_unchanged(fake_clock):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return _ok(" T02 ")

    b = _backend(handler, fake_clock)
    assert b.complete("p") == "T02"
    assert seen[0] == {"model": "m", "temperature": 0, "max_tokens": 16,
                       "messages": [{"role": "user", "content": "p"}]}


def test_http_timeout_capped_at_remaining(fake_clock):
    timeouts = []

    def handler(req):
        timeouts.append(req.extensions["timeout"]["read"])
        return _ok("x")

    b = _backend(handler, fake_clock)
    b.complete_text("p", max_tokens=256, deadline=fake_clock.now + 7.5)
    b.complete_text("p", max_tokens=256, deadline=fake_clock.now + 100)
    assert timeouts == [7.5, 30.0]


def test_deadline_already_passed_makes_no_call(fake_clock):
    calls = []
    b = _backend(lambda r: calls.append(r) or _ok("x"), fake_clock)
    with pytest.raises(TimeoutError):
        b.complete_text("p", max_tokens=256, deadline=fake_clock.now)
    assert calls == []


def test_backoff_that_would_cross_deadline_is_not_started(fake_clock):
    calls = []

    def handler(req):
        calls.append(1)
        fake_clock.advance(2.0)          # each attempt takes 2 s
        return httpx.Response(503)

    b = _backend(handler, fake_clock)
    deadline = fake_clock.now + 6.0
    with pytest.raises(TimeoutError):
        b.complete_text("p", max_tokens=256, deadline=deadline)
    # attempt (2s) + sleep 1 + attempt (2s) -> t=5; next back-off of 2s would
    # end at 7 > 6, so it is not started.
    assert fake_clock.sleeps == [1.0] and len(calls) == 2
    assert fake_clock.now < deadline


def test_retry_limit_without_deadline(fake_clock):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(429)

    b = _backend(handler, fake_clock, max_retries=5)
    with pytest.raises(httpx.HTTPStatusError):
        b.complete_text("p", max_tokens=256)
    assert len(calls) == 6 and len(fake_clock.sleeps) == 5


def test_transport_timeout_becomes_timeout_error(fake_clock):
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)

    b = _backend(handler, fake_clock)
    with pytest.raises(TimeoutError):
        b.complete_text("p", max_tokens=256, deadline=fake_clock.now + 50)


def test_prompt_cache_delete(tmp_path):
    cache = L.PromptCache(tmp_path / "c.sqlite")
    cache.put("m", "p", "raw")
    cache.put("m", "q", "other")
    cache.delete("m", "p")
    assert cache.get("m", "p") is None and cache.get("m", "q") == "other"
    cache.delete("m", "absent")          # no-op
    L.PromptCache(None).delete("m", "p")  # no-op without a file


def test_mock_backend_has_complete_text():
    assert isinstance(L.MockBackend().complete_text("x", max_tokens=256), str)


# --- Mock_Backend narration modes (task 18.2, Req 11.13, 21.1, 21.2) -------

def _mock_prompt():
    from src.explain.record import (SECTION_NAMES, DecisionRecord, canonical_json,
                                    hash_view, make_section)
    rec = DecisionRecord(
        request_id="r1", created_at="2025-01-01T00:00:00.000Z",
        serving_path="streamlit", model_versions={"improved_model_1": "sha256:x"},
        outcome=dict(chosen_team={"id": "T01", "name": "Team One"},
                     top_issues=[{"rank": 1, "issue_id": "T01_I001", "label": "a",
                                  "display_label": "A", "prob": 0.7}],
                     issue_order="probability", auto_route=True, route="auto",
                     scenario="primary_only"),
        sections={n: make_section(n, status="skipped", reason="not available",
                                  method="exact", technique="rules")
                  for n in SECTION_NAMES})
    view = json.loads(canonical_json(hash_view(rec)))
    prompt = ("rules...\n<<<DECISION_RECORD\n" + canonical_json(hash_view(rec))
              + "\nDECISION_RECORD>>>\nReply with one JSON object.")
    return prompt, view


def _at(obj, path):
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def test_mock_faithful_is_deterministic_and_cites_record_values():
    from src.explain.narration import parse_narration
    prompt, view = _mock_prompt()
    b = L.MockBackend(mode="faithful")
    raw = b.complete_text(prompt, max_tokens=1024)
    assert raw == L.MockBackend(mode="faithful").complete_text(prompt, max_tokens=1024)
    res = parse_narration(raw)
    assert res.ok and res.narration.cited_facts
    assert res.narration.consumer_summary and res.narration.agent_summary
    for f in res.narration.cited_facts:
        assert _at(view, f.path) == f.value


def test_mock_unfaithful_changes_a_number_and_the_team():
    from src.explain.narration import parse_narration
    prompt, view = _mock_prompt()
    n = parse_narration(L.MockBackend(mode="unfaithful")
                        .complete_text(prompt, max_tokens=1024)).narration
    faithful = parse_narration(L.MockBackend(mode="faithful")
                               .complete_text(prompt, max_tokens=1024)).narration
    diffs = [f for f in n.cited_facts if _at(view, f.path) != f.value]
    assert len(diffs) == 1 and diffs[0].value == _at(view, diffs[0].path) + 0.5
    import re
    team = re.compile(r"\bT01\b")
    assert team.search(faithful.agent_summary) and not team.search(n.agent_summary)
    assert "T01_I001" in n.agent_summary          # issue ids are left alone


def test_mock_failure_modes_and_delay():
    from src.explain.narration import parse_narration
    prompt, _ = _mock_prompt()
    assert not parse_narration(L.MockBackend(mode="failure_parse")
                               .complete_text(prompt, max_tokens=256)).ok
    with pytest.raises(TimeoutError):
        L.MockBackend(mode="failure_timeout").complete_text(prompt, max_tokens=256)
    slept = []
    b = L.get_backend("mock", mode="failure_parse", delay_s=30, sleep=slept.append)
    assert isinstance(b, L.MockBackend) and b.mode == "failure_parse"
    b.complete_text(prompt, max_tokens=256)
    assert slept == [30.0]
    with pytest.raises(ValueError):
        L.MockBackend(mode="bogus")


def test_mock_default_choice_mode_unchanged():
    b = L.get_backend("mock")
    assert b.mode == "choice" and b.model == "mock"
    p = "- T01: card fees\n- T06: mortgage escrow\n\nComplaint: escrow mortgage\n"
    assert b.complete(p) == "T06" and b.complete_text(p, max_tokens=256) == "T06"
