"""Unit tests for src/explain/narration.py (Req 12.1 to 12.7)."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.explain.narration import (CitedFact, Explanation, Narration,
                                   ValidationSummary, parse_narration,
                                   print_narration)

GOOD = {"consumer_summary": "Sent to Credit Reporting.",
        "agent_summary": "IM1 chose T01.",
        "cited_facts": [{"path": "outcome.chosen_team.id", "value": "T01"},
                        {"path": "outcome.top_issues.0.prob", "value": 0.878}]}


def test_parses_bare_object():
    r = parse_narration(json.dumps(GOOD))
    assert r.ok and r.errors == []
    assert r.narration.cited_facts[1].value == 0.878


def test_parses_object_inside_prose_and_fences_and_drops_unknown_keys():
    obj = dict(GOOD, extra=1)
    obj["cited_facts"] = [dict(GOOD["cited_facts"][0], note="x")]
    raw = "Here you go {not json\n```json\n" + json.dumps(obj) + "\n```\nthanks {}"
    r = parse_narration(raw)
    assert r.ok
    assert r.narration == Narration(**{**GOOD, "cited_facts": [GOOD["cited_facts"][0]]})


@pytest.mark.parametrize("raw", ["", "   ", "no json here", "[1, 2]", "{bad", None, 42])
def test_reply_level_errors_never_raise(raw):
    r = parse_narration(raw)
    assert not r.ok and r.narration is None and r.errors


def test_field_errors_name_each_field():
    obj = {"consumer_summary": "   ", "agent_summary": 5,
           "cited_facts": [{"value": 1}, {"path": "a" * 257, "value": {"x": 1}}, 3]}
    r = parse_narration(json.dumps(obj))
    assert not r.ok and r.narration is None
    assert set(r.errors) == {
        "consumer_summary: out_of_bounds", "agent_summary: wrong_type",
        "cited_facts.0.path: missing", "cited_facts.1.path: out_of_bounds",
        "cited_facts.1.value: wrong_type", "cited_facts.2: wrong_type"}


def test_missing_fields_and_count_bound():
    assert set(parse_narration("{}").errors) == {
        "consumer_summary: missing", "agent_summary: missing", "cited_facts: missing"}
    obj = dict(GOOD, cited_facts=[{"path": "p", "value": 1}] * 51)
    assert parse_narration(json.dumps(obj)).errors == ["cited_facts: out_of_bounds"]


def test_length_bounds_after_strip():
    obj = dict(GOOD, consumer_summary="  " + "a" * 1200 + "\n", agent_summary="b" * 4000)
    r = parse_narration(json.dumps(obj))
    assert r.ok and r.narration.consumer_summary == "a" * 1200
    obj["agent_summary"] = "b" * 4001
    assert parse_narration(json.dumps(obj)).errors == ["agent_summary: out_of_bounds"]


def test_printer_fixed_order_and_round_trip():
    n = Narration(consumer_summary=" Équipe T01 ✓ ", agent_summary="x",
                  cited_facts=[CitedFact(path="b", value=-0.0),
                               CitedFact(path="a", value=0.12345678),
                               CitedFact(path="c", value=True),
                               CitedFact(path="d", value=None),
                               CitedFact(path="e", value=[1, "T02"])])
    text = print_narration(n)
    assert text.startswith('{"consumer_summary":"Équipe T01 ✓","agent_summary":"x",'
                           '"cited_facts":[{"path":"b","value":0.0},'
                           '{"path":"a","value":0.123457}')
    r = parse_narration(text)
    assert r.ok and r.narration == n
    assert print_narration(r.narration) == text
    assert list(json.loads(text)) == ["consumer_summary", "agent_summary", "cited_facts"]


def test_bool_and_int_value_types_preserved():
    r = parse_narration(json.dumps(dict(GOOD, cited_facts=[
        {"path": "a", "value": True}, {"path": "b", "value": 1}])))
    assert [type(f.value) for f in r.narration.cited_facts] == [bool, int]


def test_models_reject_invalid_values():
    with pytest.raises(ValidationError):
        CitedFact(path="", value=1)
    with pytest.raises(ValidationError):
        CitedFact(path="a", value=float("nan"))
    with pytest.raises(ValidationError):
        Narration(consumer_summary="", agent_summary="x", cited_facts=[])


def test_explanation_model():
    e = Explanation(consumer_summary="c", agent_summary="a", cited_facts=[],
                    source="template", fallback_reason="timeout",
                    validation=ValidationSummary(accepted=False, failed=[
                        {"check": "number", "offending": "0.5", "reason": "not in record"}]),
                    record_hash="0" * 64, created_at="2025-01-01T00:00:00.000Z")
    assert e.narration() == Narration(consumer_summary="c", agent_summary="a", cited_facts=[])
    with pytest.raises(ValidationError):
        Explanation(**{**e.model_dump(), "source": "other"})
