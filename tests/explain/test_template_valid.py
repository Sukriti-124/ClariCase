"""Property 27: the template output is always valid and consumer-safe.

Requirements 13.7, 13.12, 14.1, 14.4, 14.5, 15.3, 15.4, 15.9, 21.1.

Two record populations are covered:

* ``records()`` from ``tests/explain/strategies.py``, whose team names, issue
  labels and ids are arbitrary unicode text. These are validated twice: with a
  ``ValidatorContext`` whose name tables come from the record itself, and with
  the production context (the committed label tables).
* records whose Chosen_Team, served issues, model top teams and kNN neighbours
  use real names and labels from ``src/explain/label_tables.json``, validated
  with the production context, so the neighbour-label and table-name checks
  meet real collisions.

Every example is timed (1 s bound, Req 14.1), and the autouse socket guard in
``conftest.py`` fails the test if the template opens a connection (Req 21.1);
the guard's attempt list is also asserted empty after every example.
"""
from __future__ import annotations

import json
import re
import time

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.explain.labels import load_tables
from src.explain.narration import Narration
from src.explain.record import DecisionRecord, canonical_json
from src.explain.template import (_one_line, _quotable, _safe_free_text,
                                  _safe_team_id, _strip_prefix, _tables,
                                  _top_key, format_fixed, template_narration)
from src.explain.validator import (AUTO_PHRASES, BANNED_WORD_RE, CONSUMER_WORD_MAX,
                                   MODEL_NAMES, REVIEW_PHRASES, ValidatorContext,
                                   default_context, validate)
from tests.explain.strategies import SECTION_NAMES, record_dicts, records, st_prob

TABLES = load_tables()
TEAM_IDS = sorted(TABLES.teams)
ISSUES_BY_TEAM = {t: sorted(i for i, e in TABLES.issues.items() if e.team_id == t)
                  for t in TEAM_IDS}
NARRATOR_MODEL = "llama-3.3-70b-versatile"
TIME_LIMIT_S = 1.0
_WS = re.compile(r"\s+")

_tables()                       # warm the label-table cache outside the timed call

_SETTINGS = settings(max_examples=100, deadline=None,
                     suppress_health_check=[HealthCheck.too_slow,
                                            HealthCheck.function_scoped_fixture])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().casefold()


def _guard(term: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")


def _has(text: str, term: str) -> bool:
    return bool(term) and _guard(term).search(text) is not None


def _fmt(v) -> str:
    return format_fixed(v, 3)


def _canon(rec: DecisionRecord) -> dict:
    """The form the template reads (6-decimal rounding of ``canonical_json``)."""
    return json.loads(canonical_json(rec))


def _record_context(rec: DecisionRecord) -> ValidatorContext:
    """Name tables taken from the record: the Chosen_Team and the served issues."""
    out = rec.outcome
    return ValidatorContext(
        masked_text=None, include_text=False,
        team_names={out.chosen_team.id: out.chosen_team.name},
        issue_labels={i.issue_id: i.label for i in out.top_issues},
        narrator_model=NARRATOR_MODEL)


def _production_context() -> ValidatorContext:
    return default_context(masked_text=None, include_text=False,
                           narrator_model=NARRATOR_MODEL)


def _timed_template(rec: DecisionRecord) -> Narration:
    start = time.perf_counter()
    n = template_narration(rec)
    elapsed = time.perf_counter() - start
    assert elapsed < TIME_LIMIT_S, f"template took {elapsed:.3f} s"
    return n


# ---------------------------------------------------------------------------
# Consumer_Summary (Req 13.7, 14.4, 15.1 to 15.3, 15.9)
# ---------------------------------------------------------------------------

def _outcome_values(c: dict) -> list[str]:
    out = c["outcome"]
    vals = {out["chosen_team"]["id"], out["chosen_team"]["name"]}
    for i in out.get("top_issues") or []:
        vals |= {i["issue_id"], i["label"], i["display_label"]}
    return sorted({_norm(v) for v in vals if _norm(v)}, key=lambda s: (-len(s), s))


def _outside_outcome(text: str, c: dict) -> str:
    """The normalised summary with the always-allowed outcome values blanked."""
    s = _norm(text)
    for v in _outcome_values(c):
        s = _guard(v).sub(" ", s)
    return s


def _neighbour_terms(c: dict, ctx: ValidatorContext) -> set[str]:
    terms: set[str] = set()
    knn = (c.get("sections") or {}).get("knn_embed") or {}
    for nb in (knn.get("details") or {}).get("neighbours") or []:
        terms.add(nb["complaint_id"])
        iid = nb.get("issue_id")
        if isinstance(iid, str):
            terms.add(iid)
            label = ctx.issue_labels.get(iid)
            if isinstance(label, str):
                terms |= {label, label.split(" :: ", 1)[-1]}
    return {_norm(t) for t in terms if _norm(t)}


def _check_consumer(n: Narration, c: dict, ctx: ValidatorContext) -> None:
    text = n.consumer_summary
    whole = _norm(text)
    out = c["outcome"]
    assert len(text.split()) <= CONSUMER_WORD_MAX

    name = _norm(out["chosen_team"]["name"])
    if name:
        assert _has(whole, name), "Chosen_Team name missing"
    issues = out.get("top_issues") or []
    if issues:
        label, disp = _norm(issues[0]["label"]), _norm(issues[0]["display_label"])
        if label or disp:
            assert _has(whole, label) or _has(whole, disp), "rank-1 issue label missing"

    rest = _outside_outcome(text, c)
    own, other = ((AUTO_PHRASES, REVIEW_PHRASES) if out["auto_route"]
                  else (REVIEW_PHRASES, AUTO_PHRASES))
    if any(v in ph for v in _outcome_values(c) for ph in (*AUTO_PHRASES, *REVIEW_PHRASES)):
        return  # a generated name is part of a routing phrase; blanking it hides the phrase
    assert ("auto-routed" if out["auto_route"] else "sent to human review") in rest
    assert not any(p in rest for p in other), "opposite routing outcome stated"
    assert any(p in rest for p in own)

    assert "%" not in rest
    assert BANNED_WORD_RE.search(rest) is None
    for m in [*MODEL_NAMES, ctx.narrator_model]:
        assert not _has(rest, _norm(m)), f"model name {m!r} in Consumer_Summary"
    gate = (c.get("sections") or {}).get("gate") or {}
    for d in gate.get("drivers") or []:
        if isinstance(d.get("label"), str):
            assert not _has(rest, _norm(d["label"])), "gate feature name leaked"
    for t in _neighbour_terms(c, ctx):
        assert not _has(rest, t), f"neighbour value {t!r} leaked"


# ---------------------------------------------------------------------------
# Agent_Summary (Req 13.7, 14.5, 15.4)
# ---------------------------------------------------------------------------

def _team_ref(v) -> str | None:
    """How the template writes a team value, or None when it cannot be stated."""
    if not _safe_team_id(v):
        return None
    return v if v in _tables()[0] else v.strip()


def _section_line(lines: list[str], name: str) -> str:
    hits = [l for l in lines if l.startswith(f"{name} (")]
    assert len(hits) == 1, f"ok section {name} has {len(hits)} lines"
    return hits[0]


def _check_drivers(line: str, drivers: list[dict]) -> None:
    idx = sorted(range(len(drivers)),
                 key=lambda i: (-abs(drivers[i]["contribution"]), i))[:3]
    for i in idx:
        d = drivers[i]
        c = _fmt(d["contribution"])
        assert f" {c}" in line, f"Driver contribution {c} missing"
        if d["kind"] == "ngram" and isinstance(d.get("label"), str):
            gram = _one_line(_strip_prefix(d["label"]))
            if _quotable(gram):
                assert f'"{gram}" {c}' in line
        elif d["kind"] == "sentence" and isinstance(d.get("text"), str):
            text = _one_line(d["text"])
            if _quotable(text):
                assert f'"{text}" {c}' in line


def _check_score(line: str, details: dict) -> None:
    s, t = details.get("score"), details.get("threshold")
    if not isinstance(s, (int, float)) or isinstance(s, bool):
        return
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        op = ">=" if s >= t else "<"
        assert f"score {_fmt(s)} {op} threshold {_fmt(t)}" in line
    else:
        assert f"score {_fmt(s)}, threshold absent" in line


def _check_routing(line: str, details: dict, chosen: str) -> None:
    teams = details.get("model_teams")
    if not isinstance(teams, dict):
        return
    shown, disagree = 0, []
    for model in sorted(teams):
        team = teams[model]
        if not _safe_free_text(model):
            continue
        if isinstance(team, dict):            # an Absent marker
            assert f"{model} absent" in line
            shown += 1
            continue
        ref = _team_ref(team)
        if ref is None:
            continue
        assert f"{model} {ref}" in line, f"top team of {model} missing"
        shown += 1
        if team != chosen:
            disagree.append(model)
    if shown:
        want = ", ".join(disagree) if disagree else "none"
        assert f"models disagreeing with the chosen team: {want}" in line


def _check_top_team(line: str, name: str, details: dict) -> None:
    if name == "improved_model_1":
        ref = _team_ref(details.get("top_team"))
        if ref:
            assert f"top team {ref}" in line
    elif name == "baseline_model_2":
        ref = _team_ref(details.get("chosen_team"))
        if ref:
            assert f"explains team {ref}" in line
    elif name in ("improved_model_2", "knn_embed"):
        key = "team_probs" if name == "improved_model_2" else "team_distribution"
        top = _top_key(details.get(key))
        if top is not None and _safe_team_id(top):
            assert f"top team {top.strip()}" in line
    elif name == "stacker":
        ref = _team_ref(details.get("top_after"))
        ranking = details.get("ranking")
        if details.get("top_after") is None and isinstance(ranking, list) and ranking:
            ref = _team_ref(ranking[0])
        if ref:
            assert f"top team {ref}" in line


def _check_agent(n: Narration, c: dict) -> None:
    lines = n.agent_summary.split("\n")
    out = c["outcome"]
    assert lines[0].startswith("Outcome: the complaint was ")
    verb = "auto-routed to" if out["auto_route"] else "sent to human review for"
    assert verb in lines[0]
    assert f"; scenario {out['scenario']};" in lines[0]
    sections = c.get("sections") or {}
    ok = [s for s in SECTION_NAMES if sections[s]["status"] == "ok"]
    assert len(lines) == 1 + len(ok)          # skipped and error sections left out
    for name in ok:
        sec = sections[name]
        line = _section_line(lines, name)
        details = sec.get("details") or {}
        _check_drivers(line, sec.get("drivers") or [])
        _check_top_team(line, name, details)
        if name in ("gate", "routing"):
            _check_score(line, details)
        if name == "routing":
            _check_routing(line, details, out["chosen_team"]["id"])


def _check_all(rec: DecisionRecord, contexts: list[ValidatorContext], attempts) -> None:
    n = _timed_template(rec)
    c = _canon(rec)
    for ctx in contexts:
        res = validate(n, rec, ctx)
        assert res.accepted and res.failed == (), res.failed
        _check_consumer(n, c, ctx)
    _check_agent(n, c)
    assert attempts == [], attempts


# ---------------------------------------------------------------------------
# Records with real label-table names
# ---------------------------------------------------------------------------

_real_team = st.sampled_from(TEAM_IDS)
_complaint_id = st.integers(10**5, 10**8 - 1).map(str)
_ROUTING_MODELS = ("improved_model_1", "baseline_model_2", "improved_model_2",
                   "knn_embed", "stacker")


@st.composite
def real_records(draw) -> DecisionRecord:
    """``record_dicts()`` with every team, issue and neighbour value made real."""
    d = draw(record_dicts())
    out = d["outcome"]
    team = draw(_real_team)
    pool = ISSUES_BY_TEAM[team]
    served = draw(st.lists(st.sampled_from(pool), min_size=1,
                           max_size=min(3, len(pool)), unique=True))
    probs = draw(st.lists(st_prob, min_size=len(served), max_size=len(served)))
    if out["issue_order"] == "probability":
        probs.sort(reverse=True)
    out["chosen_team"] = {"id": team, "name": TABLES.teams[team]}
    out["top_issues"] = [{"rank": r, "issue_id": i, "label": TABLES.issues[i].label,
                          "display_label": TABLES.issues[i].display_label, "prob": p}
                         for r, (i, p) in enumerate(zip(served, probs), start=1)]

    secs = d["sections"]
    det = {n: secs[n].setdefault("details", {}) for n in SECTION_NAMES}
    for key in ("top_team", "runner_up"):
        det["improved_model_1"][key] = draw(_real_team)
    for n in ("baseline_model_2", "improved_model_2", "stacker"):
        det[n]["chosen_team"] = draw(_real_team)
    det["improved_model_2"]["team_probs"] = draw(st.dictionaries(_real_team, st_prob,
                                                                 max_size=3))
    knn = det["knn_embed"]
    knn["team_distribution"] = draw(st.dictionaries(_real_team, st_prob, max_size=3))
    knn["neighbours"] = [
        {"rank": r, "similarity": draw(st_prob), "team_id": draw(_real_team),
         "issue_id": draw(st.sampled_from(sorted(TABLES.issues))),
         "complaint_id": draw(_complaint_id)}
        for r in range(1, draw(st.integers(0, 3)) + 1)]
    st_ = det["stacker"]
    st_["top_after"] = draw(_real_team)
    st_["ranking"] = draw(st.lists(_real_team, max_size=4))
    st_["comparison_team"] = draw(_real_team)
    det["routing"]["model_teams"] = {
        m: draw(st.one_of(_real_team, st.just({"absent": True, "reason": "missing"})))
        for m in draw(st.lists(st.sampled_from(_ROUTING_MODELS), unique=True))}
    det["agentic"]["team_choice"] = draw(_real_team)
    det["agentic"]["issue_choice"] = draw(st.sampled_from(sorted(TABLES.issues)))
    return DecisionRecord.model_validate(d)


# ---------------------------------------------------------------------------
# Property 27
# ---------------------------------------------------------------------------

# Feature: model-explainability, Property 27: Template output is always valid and consumer-safe
@_SETTINGS
@given(rec=records())
def test_template_valid_for_arbitrary_records(rec: DecisionRecord, _no_network):
    """**Validates: Requirements 13.7, 13.12, 14.1, 14.4, 14.5, 15.3, 15.4, 15.9, 21.1**"""
    _check_all(rec, [_record_context(rec), _production_context()], _no_network)


# Feature: model-explainability, Property 27: Template output is always valid and consumer-safe
@_SETTINGS
@given(rec=real_records())
def test_template_valid_for_real_label_records(rec: DecisionRecord, _no_network):
    """**Validates: Requirements 13.7, 13.12, 14.1, 14.4, 14.5, 15.3, 15.4, 15.9, 21.1**"""
    _check_all(rec, [_production_context()], _no_network)


# Feature: model-explainability, Property 27: Template output is always valid and consumer-safe
@_SETTINGS
@given(rec=st.one_of(records(serving_path="streamlit"), records(serving_path="platform", scenario="degraded"),
                     records(status="ok")))
def test_template_valid_streamlit_degraded_and_all_ok(rec: DecisionRecord, _no_network):
    """**Validates: Requirements 13.7, 13.12, 14.1, 14.4, 14.5, 15.3, 15.4, 15.9, 21.1**"""
    _check_all(rec, [_record_context(rec), _production_context()], _no_network)
