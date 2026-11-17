"""Property test for number matching in the Faithfulness_Validator (design §16).

Property 24: Number matching accepts faithful renderings.
Requirements 13.4, 13.5.

Records come from ``real_record`` (names and labels from the real label
tables). The generated value v is stored as the routing threshold and the
generated p as the routing score, so both are leaves of a schema-valid record.
A "faithful rendering" of a record value is its canonical value (floats are
rounded to 6 dp by canonical JSON, Req 2.1) rounded half-even from
``Decimal(repr(v))`` to d decimals, which is the matching rule of design §16
and the rule ``template.format_fixed`` uses.
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.explain.labels import load_tables
from src.explain.narration import Narration
from src.explain.record import DecisionRecord, canonical_json, hash_view
from src.explain.template import template_narration
from src.explain.validator import default_context, validate
from tests.explain.test_validator_unit import real_record

TABLES = load_tables()
ISSUES = sorted(TABLES.issues)


# --------------------------------------------------------------------------
# Helpers (an independent oracle; no validator internals)
# --------------------------------------------------------------------------

def _record(issue_id: str, auto: bool, path: str, v: float | int, p: float) -> DecisionRecord:
    d = real_record(issue_id, auto=auto, path=path).model_dump(mode="python")
    d["sections"]["routing"]["details"]["threshold"] = v
    d["sections"]["routing"]["details"]["score"] = p
    return DecisionRecord.model_validate(d)


def _canonical(rec: DecisionRecord) -> dict:
    return json.loads(canonical_json(hash_view(rec)))


def _leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield obj


def _round(x: Decimal, d: int) -> Decimal:
    with localcontext() as c:
        c.prec = max(60, x.adjusted() + d + 8)
        return x.quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_EVEN)


def _render(v: float | int, d: int) -> str:
    """``v`` with d decimals, rounded half-even from ``repr(v)``."""
    return format(_round(Decimal(repr(v)), d), "f")


def _render_pct(p: float, d: int) -> str:
    with localcontext() as c:
        c.prec = 60
        x = Decimal(repr(p)) * 100
    return format(_round(x, d), "f") + "%"


def _rounded_values(canon: dict, d: int, pct: bool) -> set[Decimal]:
    out = set()
    for v in _leaves(canon):
        x = Decimal(repr(v))
        if pct:
            if not 0 <= x <= 1:
                continue
            with localcontext() as c:
                c.prec = 60
                x = x * 100
        out.add(_round(x, d))
    return out


def _is_exempt(token: str) -> bool:
    return "." not in token and "%" not in token and 0 <= int(token) <= 11


def _with_agent(rec: DecisionRecord, extra: str) -> Narration:
    base = template_narration(rec)
    return Narration(consumer_summary=base.consumer_summary,
                     agent_summary=f"{base.agent_summary} Value {extra} noted.",
                     cited_facts=list(base.cited_facts))


def _number_failures(rec: DecisionRecord, extra: str) -> list[str]:
    res = validate(_with_agent(rec, extra), rec, default_context(masked_text=None))
    return [f.offending for f in res.failed if f.check == "number"]


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------

# Floats over several magnitudes, integers, and short decimals that hit
# half-way cases such as 2.675 or 0.125.
values = st.one_of(
    st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-10**9, max_value=10**9),
    st.builds(lambda n, k: float(Decimal(n).scaleb(-k)),
              st.integers(min_value=-10**7, max_value=10**7), st.integers(1, 6)),
)
probs = st.one_of(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.builds(lambda n, k: float(Decimal(n).scaleb(-k)),
              st.integers(min_value=0, max_value=10**6), st.integers(1, 6))
    .filter(lambda x: 0.0 <= x <= 1.0),
)
records = st.builds(_record, st.sampled_from(ISSUES), st.booleans(),
                    st.sampled_from(["platform", "streamlit"]), values, probs)
places = st.integers(min_value=0, max_value=6)


# --------------------------------------------------------------------------
# Property 24
# --------------------------------------------------------------------------

# Feature: model-explainability, Property 24: Number matching accepts faithful renderings
@settings(max_examples=100, deadline=None)
@given(rec=records, d=places, pd=places)
def test_faithful_renderings_pass(rec, d, pd):
    """**Validates: Requirements 13.4, 13.5**"""
    canon = _canonical(rec)
    v = canon["sections"]["routing"]["details"]["threshold"]
    p = canon["sections"]["routing"]["details"]["score"]
    plain, pct = _render(v, d), _render_pct(p, pd)
    assert _number_failures(rec, plain) == [], (v, d, plain)
    assert _number_failures(rec, pct) == [], (p, pd, pct)
    assert validate(_with_agent(rec, f"{plain} and {pct}"), rec,
                    default_context(masked_text=None)).accepted


# Feature: model-explainability, Property 24: Number matching accepts faithful renderings
@settings(max_examples=100, deadline=None)
@given(rec=records, d=places, n=st.integers(min_value=-10**6, max_value=10**6),
       percent=st.booleans())
def test_unmatched_numbers_fail_unless_exempt(rec, d, n, percent):
    """**Validates: Requirements 13.4, 13.5**"""
    canon = _canonical(rec)
    x = Decimal(n).scaleb(-d)
    token = format(x, "f") + ("%" if percent else "")
    assume(x not in _rounded_values(canon, d, percent))
    failures = _number_failures(rec, token)
    if _is_exempt(token):                                  # Req 13.5a
        assert failures == [], token
    else:                                                  # Req 13.4
        assert failures == [token], (token, failures)
    # Req 13.5c: inside a quoted phrase the number is not checked as a number
    assert _number_failures(rec, f'"{token}"') == [], token
