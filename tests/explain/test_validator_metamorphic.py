"""Metamorphic tests for the Faithfulness_Validator (design §16, Property 28).

Each test starts from a record built from the real label tables and its
template Narration, which the validator accepts. It then applies one
unfaithful edit and checks that the validator rejects the result and names
the matching check:

* a team mention replaced by a team id or name absent from the record
* an issue mention replaced by an absent issue id or label
* a non-exempt number replaced by a value that matches no record value
* a quoted phrase replaced by text that is neither a Driver nor in the text
* the outcome phrase replaced by the opposite outcome
* a Cited_Fact given a wrong value or a path that does not exist
"""
from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any, Iterator

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.explain.labels import load_tables
from src.explain.narration import CitedFact, Narration
from src.explain.record import canonical_json, hash_view
from src.explain.template import template_narration
from src.explain.validator import (AUTO_PHRASES, NUM_RE, QUOTE_RE, REVIEW_PHRASES,
                                   ValidationResult, norm, number_matches, validate)
from src.utils.text import clean_text
from tests.explain.test_validator_unit import MASKED, ctx, real_record

TABLES = load_tables()
ISSUE_IDS = sorted(TABLES.issues)
TEAM_IDS = sorted(TABLES.teams)


def _fold(s: str) -> str:
    return norm(s)


def _leaves(obj: Any) -> Iterator[Any]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v)
    else:
        yield obj


class Base:
    """A record, its accepted template Narration and the record's values."""

    def __init__(self, issue_id: str, auto: bool, path: str, include_text: bool):
        self.rec = real_record(issue_id, auto=auto, path=path)
        self.ctx = ctx(include_text=include_text)
        self.n = template_narration(self.rec)
        canon = json.loads(canonical_json(hash_view(self.rec)))
        self.canon = canon
        leaves = list(_leaves(canon))
        self.strings = {x for x in leaves if isinstance(x, str)}
        self.folded = {_fold(x) for x in self.strings}
        self.numbers = [x for x in leaves
                        if isinstance(x, (int, float)) and not isinstance(x, bool)]
        out = canon["outcome"]
        self.team_id = out["chosen_team"]["id"]
        self.team_name = out["chosen_team"]["name"]
        self.rank1 = out["top_issues"][0]
        self.auto = auto
        res = validate(self.n, self.rec, self.ctx)
        assert res == ValidationResult(True, ()), res.failed

    def foreign_teams(self) -> list[str]:
        return [t for t in TEAM_IDS if t not in self.strings
                and _fold(TABLES.teams[t]) not in self.folded]

    def foreign_issues(self) -> list[str]:
        return [i for i in ISSUE_IDS if i not in self.strings
                and _fold(TABLES.issues[i].label) not in self.folded]


@st.composite
def bases(draw) -> Base:
    return Base(draw(st.sampled_from(ISSUE_IDS)), draw(st.booleans()),
                draw(st.sampled_from(("platform", "streamlit"))), draw(st.booleans()))


def edit(n: Narration, field: str, old: str, new: str) -> Narration:
    """Replace the first occurrence of ``old`` in one summary."""
    text = getattr(n, field)
    assert old in text, (field, old, text)
    text = text.replace(old, new, 1)
    consumer = text if field == "consumer_summary" else n.consumer_summary
    agent = text if field == "agent_summary" else n.agent_summary
    return Narration(consumer_summary=consumer, agent_summary=agent,
                     cited_facts=list(n.cited_facts))


def failed(b: Base, n: Narration) -> set[tuple[str, str]]:
    res = validate(n, b.rec, b.ctx)
    assert not res.accepted and res.failed
    return {(f.check, f.offending) for f in res.failed}


FIELDS = st.sampled_from(("consumer_summary", "agent_summary"))


# Feature: model-explainability, Property 28: Unfaithful edits are rejected (metamorphic)
@settings(max_examples=100, deadline=None)
@given(b=bases(), data=st.data())
def test_foreign_team_id_rejected(b, data):
    """**Validates: Requirements 13.1, 13.9**"""
    foreign = data.draw(st.sampled_from(b.foreign_teams()))
    n = edit(b.n, "agent_summary", f"chosen team {b.team_id}", f"chosen team {foreign}")
    assert ("team_mention", foreign) in failed(b, n)


# Feature: model-explainability, Property 28: Unfaithful edits are rejected (metamorphic)
@settings(max_examples=100, deadline=None)
@given(b=bases(), field=FIELDS, data=st.data())
def test_foreign_team_name_rejected(b, field, data):
    """**Validates: Requirements 13.1, 13.9**"""
    foreign = TABLES.teams[data.draw(st.sampled_from(b.foreign_teams()))]
    n = edit(b.n, field, b.team_name, foreign)
    assert ("team_mention", _fold(foreign)) in failed(b, n)


# Feature: model-explainability, Property 28: Unfaithful edits are rejected (metamorphic)
@settings(max_examples=100, deadline=None)
@given(b=bases(), data=st.data())
def test_foreign_issue_id_rejected(b, data):
    """**Validates: Requirements 13.2, 13.9**"""
    foreign = data.draw(st.sampled_from(b.foreign_issues()))
    n = edit(b.n, "agent_summary", b.rank1["issue_id"], foreign)
    assert ("issue_mention", foreign) in failed(b, n)


# Feature: model-explainability, Property 28: Unfaithful edits are rejected (metamorphic)
@settings(max_examples=100, deadline=None)
@given(b=bases(), field=FIELDS, data=st.data())
def test_foreign_issue_label_rejected(b, field, data):
    """**Validates: Requirements 13.2, 13.9**"""
    foreign = TABLES.issues[data.draw(st.sampled_from(b.foreign_issues()))].label
    old = b.rank1["display_label"] if field == "consumer_summary" else b.rank1["label"]
    n = edit(b.n, field, old, foreign)
    assert ("issue_mention", _fold(foreign)) in failed(b, n)


def _number_tokens(text: str) -> list[re.Match]:
    """Numbers outside quoted spans that are not exempt small integers."""
    quoted = [m.span() for m in QUOTE_RE.finditer(text)]
    out = []
    for m in NUM_RE.finditer(text):
        if any(a <= m.start() < z for a, z in quoted):
            continue
        tok = m.group()
        if "." not in tok and "%" not in tok and 0 <= int(tok) <= 11:
            continue
        out.append(m)
    return out


# Feature: model-explainability, Property 28: Unfaithful edits are rejected (metamorphic)
@settings(max_examples=100, deadline=None)
@given(b=bases(), data=st.data())
def test_perturbed_number_rejected(b, data):
    """**Validates: Requirements 13.9, 13.13**"""
    text = b.n.agent_summary
    tokens = _number_tokens(text)
    assert tokens, text
    m = data.draw(st.sampled_from(tokens))
    body = m.group().rstrip("%")
    places = len(body.split(".", 1)[1]) if "." in body else 0
    step = Decimal(1).scaleb(-places)
    k = data.draw(st.integers(-500, 500).filter(lambda x: x != 0))
    new = format((Decimal(body) + k * step).quantize(step), "f")
    if new.startswith("-") and Decimal(new) == 0:
        new = new[1:]
    assume(not number_matches(new, b.numbers))
    assume(places or not 0 <= int(new) <= 11)
    n = Narration(consumer_summary=b.n.consumer_summary,
                  agent_summary=text[:m.start()] + new + text[m.end():],
                  cited_facts=list(b.n.cited_facts))
    assert ("number", new) in failed(b, n)


_WORDS = st.lists(st.text("abcdefghijklmnopqrstuvwxyzQXZ", min_size=1, max_size=10),
                  min_size=1, max_size=6).map(" ".join)


# Feature: model-explainability, Property 28: Unfaithful edits are rejected (metamorphic)
@settings(max_examples=100, deadline=None)
@given(b=bases(), data=st.data())
def test_invented_quote_rejected(b, data):
    """**Validates: Requirements 13.3, 13.9**"""
    text = b.n.agent_summary
    quotes = list(QUOTE_RE.finditer(text))
    assert quotes, text
    m = data.draw(st.sampled_from(quotes))
    invented = data.draw(_WORDS)
    nq = norm(invented)
    drivers = {norm(q.group(1)) for q in quotes}
    assume(nq not in drivers)
    assume(nq not in norm(MASKED) and nq not in norm(clean_text(MASKED)))
    n = Narration(consumer_summary=b.n.consumer_summary,
                  agent_summary=text[:m.start()] + f'"{invented}"' + text[m.end():],
                  cited_facts=list(b.n.cited_facts))
    assert ("quoted_phrase", invented) in failed(b, n)


# Feature: model-explainability, Property 28: Unfaithful edits are rejected (metamorphic)
@settings(max_examples=100, deadline=None)
@given(b=bases(), field=FIELDS)
def test_opposite_outcome_rejected(b, field):
    """**Validates: Requirements 13.7, 13.9**"""
    auto_verb, review_verb = "auto-routed to", "sent to human review for"
    old, new = (auto_verb, review_verb) if b.auto else (review_verb, auto_verb)
    got = failed(b, edit(b.n, field, old, new))
    opposite = REVIEW_PHRASES if b.auto else AUTO_PHRASES
    assert ("required_outcome", field) in got
    assert any(("opposite_outcome", p) in got for p in opposite)


def _perturb(value: Any, delta: float) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + delta
    if isinstance(value, str):
        return value + "_x"
    if isinstance(value, list):
        return value + ["x"]
    return "x"


# Feature: model-explainability, Property 28: Unfaithful edits are rejected (metamorphic)
@settings(max_examples=100, deadline=None)
@given(b=bases(), data=st.data())
def test_perturbed_cited_fact_rejected(b, data):
    """**Validates: Requirements 13.9, 13.13**"""
    facts = list(b.n.cited_facts)
    assert facts
    i = data.draw(st.integers(0, len(facts) - 1))
    cf = facts[i]
    if data.draw(st.booleans()):
        delta = data.draw(st.one_of(st.floats(1e-5, 1e3), st.floats(-1e3, -1e-5)))
        facts[i] = CitedFact(path=cf.path, value=_perturb(cf.value, delta))
        expected = ("cited_fact_value", cf.path)
    else:
        facts[i] = CitedFact(path=cf.path + ".nope", value=cf.value)
        expected = ("cited_fact_path", cf.path + ".nope")
    n = Narration(consumer_summary=b.n.consumer_summary,
                  agent_summary=b.n.agent_summary, cited_facts=facts)
    assert expected in failed(b, n)
