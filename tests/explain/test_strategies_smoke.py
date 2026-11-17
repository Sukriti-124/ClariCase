"""Smoke tests for the record strategies in tests/explain/strategies.py (task 2.4)."""
from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from src.explain.record import DecisionRecord, canonical_json
from tests.explain import strategies as S

_QUIET = [HealthCheck.too_slow, HealthCheck.data_too_large]


@settings(max_examples=60, deadline=None, suppress_health_check=_QUIET)
@given(S.st_record_dict)
def test_drawn_records_pass_every_validator(d):
    rec = DecisionRecord.model_validate(d)
    # The dump also passes strict validation, and serialises canonically.
    DecisionRecord.model_validate(rec.model_dump(), strict=True)
    assert canonical_json(rec)


_PINS = st.one_of(
    st.tuples(st.just("streamlit"), st.just(None), st.sampled_from(S.ISSUE_ORDERS),
              st.sampled_from(S.STATUSES)),
    st.tuples(st.just("platform"), st.sampled_from(S.SCENARIOS),
              st.sampled_from(S.ISSUE_ORDERS), st.sampled_from(S.STATUSES)))


@settings(max_examples=30, deadline=None, suppress_health_check=_QUIET)
@given(st.data(), _PINS)
def test_pinned_dimensions_are_respected(data, pins):
    path, scenario, order, status = pins
    rec = data.draw(S.records(path, scenario, order, status))
    assert rec.serving_path == path
    assert rec.outcome.scenario == (scenario or "primary_only")
    assert rec.outcome.issue_order == order
    assert {s.status for s in rec.sections.values()} == {status}


def test_streamlit_rejects_other_scenarios():
    with pytest.raises(ValueError):
        _exists(S.records(serving_path="streamlit", scenario="auto"),
                lambda r: True)


# --- coverage: each kind of value shows up in unpinned draws -----------------

class _Found(Exception):
    pass


def _exists(strategy, pred, n=300):
    """Raise unless ``pred`` holds for some draw (no shrinking, so it is fast)."""
    @settings(max_examples=n, phases=[Phase.generate], database=None,
              deadline=None, suppress_health_check=list(HealthCheck))
    @given(strategy)
    def probe(x):
        if pred(x):
            raise _Found

    try:
        probe()
    except _Found:
        return
    raise AssertionError("no draw satisfied the predicate")


def _floats(obj):
    if isinstance(obj, float):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _floats(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _floats(v)


def _details_values(d):
    for s in d["sections"].values():
        yield from (s.get("details") or {}).values()


_COVERAGE = {
    "negative zero": lambda d: any(f == 0.0 and math.copysign(1, f) < 0
                                   for f in _floats(d)),
    "subnormal": lambda d: any(0 < abs(f) < 2.2250738585072014e-308
                               for f in _floats(d)),
    "rounding boundary": lambda d: any(
        abs(abs(f) * 1e6 % 1 - 0.5) < 1e-6 and 1e-6 < abs(f) < 10
        for f in _floats(d)),
    "Absent marker": lambda d: any(isinstance(v, dict) and v.get("absent") is True
                                   for v in _details_values(d)),
    "redacted driver": lambda d: any(dr["kind"] == "redacted"
                                     for s in d["sections"].values()
                                     for dr in s.get("drivers", [])),
    "astral unicode": lambda d: any(ord(c) > 0xFFFF for c in canonical_json(d)),
    "mixed statuses": lambda d: len({s["status"] for s in d["sections"].values()}) == 3,
    "agentic order against prob": lambda d: (
        d["outcome"]["issue_order"] == "agentic"
        and any(a["prob"] < b["prob"] for a, b in
                zip(d["outcome"]["top_issues"], d["outcome"]["top_issues"][1:]))),
}


@pytest.mark.parametrize("name", sorted(_COVERAGE))
def test_unpinned_draws_cover(name):
    _exists(S.st_record_dict, _COVERAGE[name])
