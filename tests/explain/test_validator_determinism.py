"""Property test: the validator is deterministic (Req 13.11)."""
from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.explain.narration import CitedFact, Narration
from src.explain.template import template_narration
from src.explain.validator import validate
from tests.explain.test_validator_unit import TABLES, ctx, real_record

ISSUES = sorted(TABLES.issues)

_text = st.text(min_size=1, max_size=300).filter(lambda s: s.strip())
_fact = st.builds(
    CitedFact,
    path=st.sampled_from(["outcome.chosen_team.id", "outcome.top_issues.0.prob",
                          "routing.details.score", "nope.path", "sections.gate.status"]),
    value=st.one_of(st.none(), st.booleans(), st.integers(-10**6, 10**6),
                    st.floats(allow_nan=False, allow_infinity=False, width=32),
                    st.text(max_size=20)))


@st.composite
def _case(draw):
    rec = real_record(draw(st.sampled_from(ISSUES)), auto=draw(st.booleans()),
                      path=draw(st.sampled_from(["platform", "streamlit"])))
    base = template_narration(rec)
    if draw(st.booleans()):
        n = base
    else:
        n = Narration(
            consumer_summary=draw(st.one_of(_text, st.just(base.consumer_summary))),
            agent_summary=draw(st.one_of(_text, st.just(base.agent_summary))),
            cited_facts=draw(st.lists(_fact, max_size=4)))
    return rec, n, draw(st.booleans())


# Feature: model-explainability, Property 26: Validation is deterministic
# **Validates: Requirements 13.11**
@settings(max_examples=100, deadline=None, suppress_health_check=list(HealthCheck))
@given(_case())
def test_validation_is_deterministic(case):
    rec, n, include_text = case
    c = ctx(include_text=include_text)
    first = validate(n, rec, c)
    second = validate(n, rec, c)
    assert first == second
    assert first.accepted == second.accepted
    assert [(f.check, f.offending, f.reason) for f in first.failed] == \
           [(f.check, f.offending, f.reason) for f in second.failed]
