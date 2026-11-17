"""Property test: Cited_Facts must match stored values (design §16 step 5).

Feature: model-explainability, Property 25.

For any record and any existing leaf path, a Cited_Fact with the stored value
passes, and so does any number within 1e-6 of it. A Cited_Fact with a
non-existent path, a different string or bool, a type change, or a number more
than 1e-6 away fails ``cited_fact_path`` or ``cited_fact_value``.

**Validates: Requirements 13.6**
"""
from __future__ import annotations

import json
from typing import Any, Iterator

from hypothesis import given, settings
from hypothesis import strategies as st

from src.explain.labels import load_tables
from src.explain.narration import PATH_MAX, CitedFact, Narration
from src.explain.record import canonical_json, hash_view
from src.explain.template import template_narration
from src.explain.validator import validate
from tests.explain.strategies import st_record
from tests.explain.test_validator_unit import ctx, real_record

ISSUE_IDS = sorted(load_tables().issues)
NEAR_LIMIT = 1e6          # below this magnitude a 1e-6 step survives float addition
PLAIN = Narration(consumer_summary="Placeholder.", agent_summary="Placeholder.")


def _view(record) -> dict:
    """The record as the validator sees it (canonical hash view)."""
    return json.loads(canonical_json(hash_view(record)))


def _nodes(obj: Any, segs: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Every node with its path segments; paths through keys holding '.' are skipped."""
    yield segs, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "." not in k:
                yield from _nodes(v, segs + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _nodes(v, segs + (str(i),))


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _usable(segs: tuple[str, ...]) -> bool:
    return bool(segs) and len(".".join(segs)) <= PATH_MAX - 40


def _leaves(view: dict) -> list[tuple[str, Any]]:
    return [(".".join(s), v) for s, v in _nodes(view)
            if _usable(s) and not isinstance(v, (dict, list))]


def _containers(view: dict) -> list[tuple[tuple[str, ...], Any]]:
    return [(s, v) for s, v in _nodes(view)
            if isinstance(v, (dict, list)) and (not s or _usable(s))]


GOOD_KINDS = ("same", "near")
BAD_VALUE_KINDS = ("different", "type_change")


@st.composite
def _cited_fact(draw, view: dict) -> tuple[str, CitedFact]:
    """A (kind, Cited_Fact) pair; kind says whether the fact should pass."""
    kind = draw(st.sampled_from(GOOD_KINDS + BAD_VALUE_KINDS + ("missing",)))
    if kind == "missing":
        segs, node = draw(st.sampled_from(_containers(view)))
        if isinstance(node, list):
            seg = draw(st.sampled_from((str(len(node)), str(len(node) + 7), "x", "-1")))
        else:
            seg = draw(st.text("abcxyz_", min_size=1, max_size=6).map(
                lambda s: "zz_missing_" + s).filter(lambda s: s not in node))
        if draw(st.booleans()):
            # a path that runs past an existing scalar leaf
            path, _ = draw(st.sampled_from(_leaves(view)))
            return kind, CitedFact(path=path + "." + seg, value=0)
        return kind, CitedFact(path=".".join(segs + (seg,)), value=0)

    path, stored = draw(st.sampled_from(_leaves(view)))
    if kind == "near" and not _is_num(stored):
        kind = "same"
    if kind == "same":
        return kind, CitedFact(path=path, value=stored)
    if kind == "near":
        if abs(stored) >= NEAR_LIMIT:
            return kind, CitedFact(path=path, value=stored)
        d = draw(st.one_of(st.sampled_from((-1e-6, -5e-7, 0.0, 4e-7, 1e-6)),
                           st.floats(-1e-6, 1e-6)))
        return kind, CitedFact(path=path, value=stored + d)
    if kind == "different":
        if isinstance(stored, bool):
            value: Any = not stored
        elif isinstance(stored, str):
            value = stored + draw(st.sampled_from(("x", " ", "\u00e9", "0")))
        elif _is_num(stored):
            if abs(stored) >= NEAR_LIMIT:
                value = -stored
            else:
                sign = draw(st.sampled_from((-1, 1)))
                value = stored + sign * draw(st.floats(2e-6, 1e6))
        else:                                   # None: a different value is a type change
            value = draw(st.sampled_from((0, "", False)))
        return kind, CitedFact(path=path, value=value)
    # type_change
    if isinstance(stored, bool):
        value = draw(st.sampled_from((int(stored), float(stored), str(stored).lower(), None)))
    elif _is_num(stored):
        value = draw(st.sampled_from((repr(stored), True, False, None)))
    elif isinstance(stored, str):
        value = draw(st.sampled_from((0, 1.5, True, None)))
    else:
        value = draw(st.sampled_from((0, "", False, "null")))
    return kind, CitedFact(path=path, value=value)


def _expected(kind: str) -> set[str]:
    if kind in GOOD_KINDS:
        return set()
    return {"cited_fact_path"} if kind == "missing" else {"cited_fact_value"}


def _cited_checks(res) -> set[str]:
    return {f.check for f in res.failed if f.check.startswith("cited_fact")}


# Feature: model-explainability, Property 25: Cited facts must match stored values
@settings(max_examples=100, deadline=None)
@given(data=st.data(), record=st_record)
def test_cited_facts_on_generated_records(data, record):
    view = _view(record)
    kind, cf = data.draw(_cited_fact(view))
    res = validate(PLAIN.model_copy(update={"cited_facts": [cf]}), record, ctx())
    assert "internal_error" not in {f.check for f in res.failed}, res.failed
    assert _cited_checks(res) == _expected(kind), (kind, cf, res.failed)
    for f in res.failed:
        if f.check.startswith("cited_fact"):
            assert f.offending == cf.path


# Feature: model-explainability, Property 25: Cited facts must match stored values
@settings(max_examples=100, deadline=None)
@given(data=st.data(), issue_id=st.sampled_from(ISSUE_IDS), auto=st.booleans(),
       path=st.sampled_from(("platform", "streamlit")))
def test_cited_facts_on_real_records(data, issue_id, auto, path):
    # The template narration is accepted on real records, so one added
    # Cited_Fact decides the result on its own.
    record = real_record(issue_id, auto=auto, path=path)
    base = template_narration(record)
    kind, cf = data.draw(_cited_fact(_view(record)))
    n = Narration(consumer_summary=base.consumer_summary, agent_summary=base.agent_summary,
                  cited_facts=list(base.cited_facts) + [cf])
    res = validate(n, record, ctx())
    assert res.accepted == (kind in GOOD_KINDS), (kind, cf, res.failed)
    assert {f.check for f in res.failed} == _expected(kind), (kind, cf, res.failed)
