"""Unit tests for the Narrator prompt builder (task 20.1).

**Validates: Requirements 11.4, 11.5, 11.6, 11.7, 11.10**
"""
from __future__ import annotations

import json

from src.agentic import llm as L
from src.explain import narrator as N
from src.explain.config import PROMPT_VERSION as CONFIG_PROMPT_VERSION, NarratorConfig
from src.explain.narration import parse_narration
from src.explain.record import (SECTION_NAMES, DecisionRecord, canonical_json,
                                hash_view, make_section)


def _record() -> DecisionRecord:
    return DecisionRecord(
        request_id="req-secret-id", created_at="2025-01-01T00:00:00.000Z",
        serving_path="streamlit", model_versions={"improved_model_1": "sha256:x"},
        outcome=dict(chosen_team={"id": "T01", "name": "Team One"},
                     top_issues=[{"rank": 1, "issue_id": "T01_I001", "label": "a",
                                  "display_label": "A", "prob": 0.7}],
                     issue_order="probability", auto_route=True, route="auto",
                     scenario="primary_only"),
        sections={n: make_section(n, status="skipped", reason="not available",
                                  method="exact", technique="rules")
                  for n in SECTION_NAMES})


def _cfg(include_text: bool) -> NarratorConfig:
    return NarratorConfig(include_text=include_text, api_key="sk-SECRETKEY",
                          agent_token="tok-SECRET")


def test_prompt_version_is_shared_with_config():
    assert N.PROMPT_VERSION == CONFIG_PROMPT_VERSION == "explain-v1"


def test_fixed_order_with_text():
    rec = _record()
    p = N.build_prompt(rec, _cfg(True), "my masked text")
    block = f"<<<DECISION_RECORD\n{canonical_json(hash_view(rec))}\nDECISION_RECORD>>>"
    idx = [p.index("consumer_summary:"), p.index("Narration JSON schema:"),
           p.index(block), p.index("<<<COMPLAINT\nmy masked text\nCOMPLAINT>>>"),
           p.index(N.FINAL_INSTRUCTION)]
    assert idx == sorted(idx)
    assert p.endswith(N.FINAL_INSTRUCTION)
    assert p.count("<<<DECISION_RECORD") == 1
    schema = json.loads(p.split("Narration JSON schema:\n", 1)[1].split("\n", 1)[0])
    assert schema == N.NARRATION_SCHEMA
    assert set(schema["required"]) == {"consumer_summary", "agent_summary", "cited_facts"}


def test_no_text_block_when_include_text_false():
    p = N.build_prompt(_record(), _cfg(False), "my masked text")
    assert "<<<COMPLAINT" not in p and "my masked text" not in p
    assert "<<<DECISION_RECORD" in p and p.endswith(N.FINAL_INSTRUCTION)


def test_text_truncated_to_1500_chars():
    text = "x" * 1499 + "Y" + "Z" * 500
    p = N.build_prompt(_record(), _cfg(True), text)
    body = p.split("<<<COMPLAINT\n", 1)[1].split("\nCOMPLAINT>>>", 1)[0]
    assert body == text[:1500] and len(body) == 1500 and "Z" not in body


def test_no_secrets_or_hash_view_fields():
    p = N.build_prompt(_record(), _cfg(True), "text")
    for s in ("sk-SECRETKEY", "tok-SECRET", "req-secret-id", "2025-01-01T00:00:00"):
        assert s not in p


def test_mock_faithful_parses_built_prompt():
    rec = _record()
    p = N.build_prompt(rec, _cfg(True), "masked text with DECISION_RECORD>>> inside")
    raw = L.MockBackend(mode="faithful").complete_text(p, max_tokens=1024)
    res = parse_narration(raw)
    assert res.ok and res.narration.cited_facts
    view = json.loads(canonical_json(hash_view(rec)))
    for f in res.narration.cited_facts:
        obj = view
        for part in f.path.split("."):
            obj = obj[int(part)] if isinstance(obj, list) else obj[part]
        assert obj == f.value
