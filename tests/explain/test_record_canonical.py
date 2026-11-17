"""Unit tests for canonical_json, parse_record, hash_view and record_hash (task 2.3)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from src.explain.record import (NonFiniteError, RecordParseError,
                                canonical_json, hash_view, make_section,
                                parse_record, record_hash)
from tests.explain.test_record_schema import _outcome, _record


@dataclass
class _Cfg:
    include_text: bool = False
    settings: dict = field(default_factory=lambda: {
        "backend": "mock", "model": "m", "audiences": ["agent", "consumer"],
        "include_text": False, "mock_mode": "ok", "prompt_version": "explain-v1"})

    def hash_settings(self) -> dict:
        return dict(self.settings, include_text=self.include_text)


def _gate_record(**details):
    sec = make_section("gate", status="ok", method="approximate",
                       technique="ablation", details=details)
    rec = _record()
    sections = dict(rec.sections, gate=sec)
    return _record(serving_path="platform", outcome=_outcome(scenario="auto"),
                   sections=sections)


# --- canonical_json ---------------------------------------------------------

def test_canonical_format():
    out = canonical_json({"b": 1, "a": {"é": "ü", "Z": [0.1234565, -0.0, 2.5e-7]}})
    # 0.1234565 is just below the half in binary, 2.5e-7 rounds to 0
    assert out == '{"a":{"Z":[0.123456,0.0,0.0],"é":"ü"},"b":1}'
    assert canonical_json({"x": 0.5000005}) == '{"x":0.5}'  # binary value < half
    assert canonical_json({"t": True, "n": None, "i": -0}) == '{"i":0,"n":null,"t":true}'


def test_canonical_rejects_every_non_finite_path():
    bad = {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": float("-inf")}}
    with pytest.raises(NonFiniteError) as ei:
        canonical_json(bad)
    assert ei.value.paths == ["a", "b.1", "c.d"]


def test_canonical_rejects_non_str_keys():
    with pytest.raises(TypeError):
        canonical_json({1: "x"})


def test_round_trip_is_byte_identical():
    rec = _gate_record(score=0.71234567, threshold=0.5, constant=-0.0,
                       residual=1e-9, gate_kind="xgb")
    first = canonical_json(rec)
    parsed = parse_record(first.encode("utf-8"))
    assert canonical_json(parsed) == first
    assert parsed.sections["gate"].details.score == 0.712346
    assert parse_record(first) == parsed          # str input too


# --- parse_record -----------------------------------------------------------

def test_parse_int_for_float_and_absent():
    d = json.loads(canonical_json(_gate_record(score=0.5)))
    d["sections"]["gate"]["details"]["threshold"] = 1
    rec = parse_record(json.dumps(d))
    assert rec.sections["gate"].details.threshold == 1.0
    assert isinstance(rec.sections["gate"].details.threshold, float)
    d["sections"]["gate"]["details"]["threshold"] = {"absent": True, "reason": "none"}
    assert parse_record(json.dumps(d)).sections["gate"].details.threshold.absent


def test_parse_invalid_utf8_reports_byte_offset():
    good = canonical_json(_record()).encode("utf-8")
    data = good[:10] + b"\xff" + good[10:]
    with pytest.raises(RecordParseError) as ei:
        parse_record(data)
    assert ei.value.kind == "decode" and ei.value.offset == 10


def test_parse_malformed_json_reports_byte_offset():
    # "é" is 2 bytes, so the character position 8 is byte offset 9
    with pytest.raises(RecordParseError) as ei:
        parse_record('{"é": 1 x}'.encode("utf-8"))
    assert ei.value.kind == "json" and ei.value.offset == 9


def test_parse_rejects_duplicate_keys_and_nan_tokens():
    with pytest.raises(RecordParseError, match="duplicate key 'a'"):
        parse_record('{"a": 1, "a": 2}')
    with pytest.raises(RecordParseError) as ei:
        parse_record('{"s": "NaN", "x": NaN}')
    assert ei.value.kind == "json" and ei.value.offset == 18


def test_parse_unsupported_version():
    d = json.loads(canonical_json(_record()))
    d["schema_version"] = "2"
    with pytest.raises(RecordParseError) as ei:
        parse_record(json.dumps(d))
    assert ei.value.kind == "version" and ei.value.paths == ["schema_version"]


def test_parse_reports_every_failing_path():
    d = json.loads(canonical_json(_gate_record(score=0.5)))
    del d["request_id"]
    d["bogus"] = 1
    d["outcome"]["top_issues"][0]["prob"] = True
    d["outcome"]["auto_route"] = "yes"
    d["sections"]["gate"]["details"]["score"] = "high"
    with pytest.raises(RecordParseError) as ei:
        parse_record(json.dumps(d))
    assert ei.value.kind == "schema"
    assert set(ei.value.paths) == {
        "request_id", "bogus", "outcome.top_issues.0.prob", "outcome.auto_route",
        "sections.gate.details.score"}


def test_parse_rejects_non_object():
    with pytest.raises(RecordParseError) as ei:
        parse_record("[1, 2]")
    assert ei.value.paths == [""]


# --- hash_view and record_hash ----------------------------------------------

def test_hash_view_drops_volatile_fields():
    view = hash_view(_record(timings={"total": 5}))
    assert not {"request_id", "created_at", "timings"} & view.keys()
    assert "outcome" in view and "sections" in view


def test_record_hash_stable_and_ignores_volatile_fields():
    cfg = _Cfg()
    h = record_hash(_record(), cfg, "masked")
    assert len(h) == 64 and h == h.lower() and int(h, 16) >= 0
    assert record_hash(_record(), cfg, "masked") == h
    other = _record(request_id="r2", created_at="2026-01-01T00:00:00.000Z",
                    timings={"total": 9})
    assert record_hash(other, cfg, "masked") == h
    # masked text is ignored while include_text is false
    assert record_hash(_record(), cfg, "different") == h


def test_record_hash_sensitive_to_record_settings_and_text():
    base = _Cfg()
    h = record_hash(_record(), base, "masked")
    assert record_hash(_record(model_versions={"improved_model_1": "sha256:y"}),
                       base, "masked") != h
    changed = _Cfg(settings=dict(base.settings, model="m2"))
    assert record_hash(_record(), changed, "masked") != h
    with_text = _Cfg(include_text=True)
    ht = record_hash(_record(), with_text, "masked")
    assert ht != h
    assert record_hash(_record(), with_text, "other") != ht
