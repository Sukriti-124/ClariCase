"""Property 1: Explaining never changes the decision (tasks 13.2 and 20.8).

Clauses covered here (task 13.2, the Record_Builder):

* Builder clause: for any served outcome on either serving path and any mix
  of section successes, skips and failures, ``build_record`` returns a record
  whose ``outcome`` equals the served decision, resolves every team and issue
  value to exactly one label-table entry, keeps probabilities in [0, 1] and
  follows the Issue_Order rules (Req 1.1, 1.5, 1.7, 1.11).
* Mismatch clause: a record whose decision differs from the served one is
  never returned (``BuildError("outcome_mismatch")``).
* Streamlit clause: the served prediction of the synthetic IM1 model
  (``src.improved_model_1.predict.predict``) equals the record outcome, and
  the scenario is ``primary_only`` (Req 1.11).
* No-mutation clause: the served outcome, the sections mapping, the
  prediction dict and the fitted model passed in are unchanged afterwards.

Task 20.8 adds the narrate clause (Req 11.9) at the end of this file: reuse
``built_records()`` and ``assert_decision_unchanged`` and compare
``canonical_json(record.outcome)`` before and after ``Narrator.narrate``.

**Validates: Requirements 1.1, 1.5, 1.7, 1.11, 11.9**
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.app.pii import mask
from src.explain.builder import STREAMLIT_RUN, BuildError, build_record
from src.explain.explainers import error_section, run_explainer, skipped_section
from src.explain.explainers.im1 import explain_im1
from src.explain.explainers.routing import explain_routing_streamlit
from src.explain.labels import display_label, issue_labels, load_tables
from src.explain.record import (SECTION_NAMES, DecisionRecord, Outcome,
                                canonical_json, make_section)
from src.improved_model_1.predict import predict as im1_predict
from tests.explain.fixtures import st_text as st_complaint
from tests.explain.fixtures import synthetic_im1
from tests.explain.strategies import section_dicts, st_nonblank, st_prob

TABLES = load_tables()
TEAM_IDS = sorted(TABLES.teams)
ISSUES_BY_TEAM = {t: sorted(i for i, e in TABLES.issues.items() if e.team_id == t)
                  for t in TEAM_IDS}
PLATFORM_SCENARIOS = ("auto", "assisted", "contested", "degraded")
TEXT_DRIVER_KINDS = ("ngram", "sentence")


# ---------------------------------------------------------------------------
# The property, as one reusable assertion
# ---------------------------------------------------------------------------

def assert_decision_unchanged(record: DecisionRecord, served: Outcome) -> None:
    """Every bullet of Property 1 except the narrate clause."""
    o = record.outcome
    # Chosen_Team, top-3 issues (ids, probabilities, order), auto-route, scenario.
    assert o.chosen_team == served.chosen_team
    assert [(i.rank, i.issue_id, i.prob) for i in o.top_issues] == \
        [(i.rank, i.issue_id, i.prob) for i in served.top_issues]
    assert (o.auto_route, o.route) == (served.auto_route, served.route)
    assert o.scenario == served.scenario
    assert o.issue_order == served.issue_order
    assert canonical_json(o) == canonical_json(served)

    # Every value resolves to exactly one label-table entry (Req 1.5).
    team = (o.chosen_team.id, o.chosen_team.name)
    assert sum(1 for t in TABLES.teams.items() if t == team) == 1
    for i in o.top_issues:
        hits = [iid for iid, e in TABLES.issues.items()
                if (iid, e.label, e.display_label)
                == (i.issue_id, i.label, i.display_label)]
        assert len(hits) == 1, i.issue_id
        assert 0.0 <= i.prob <= 1.0

    # Issue_Order rules (Req 1.1).
    assert o.issue_order in ("probability", "agentic")
    if o.issue_order == "probability":
        probs = [i.prob for i in o.top_issues]
        assert all(a >= b for a, b in zip(probs, probs[1:]))
    else:
        ag = record.sections["agentic"]
        assert ag.status == "ok"
        pick = ag.details.issue_choice
        assert o.top_issues and o.top_issues[0].issue_id == pick
        assert [i.issue_id for i in o.top_issues[1:]] == \
            [i.issue_id for i in served.top_issues if i.issue_id != pick]

    if record.serving_path == "streamlit":                       # Req 1.11
        assert o.scenario == "primary_only"


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _issue_ref(rank: int, issue_id: str, prob: float) -> dict:
    e = TABLES.issues[issue_id]
    return {"rank": rank, "issue_id": issue_id, "label": e.label,
            "display_label": e.display_label, "prob": prob}


@st.composite
def served_outcomes(draw, serving_path: str, issue_order: str) -> Outcome:
    """A served decision over the real label tables."""
    team = draw(st.sampled_from(TEAM_IDS))
    pool = ISSUES_BY_TEAM[team] or sorted(TABLES.issues)
    lo = 1 if issue_order == "agentic" else 0
    ids = draw(st.lists(st.sampled_from(pool), min_size=lo,
                        max_size=min(3, len(pool)), unique=True))
    probs = sorted(draw(st.lists(st_prob, min_size=len(ids), max_size=len(ids))),
                   reverse=True)
    pairs = list(zip(ids, probs))
    if issue_order == "agentic":
        # The agentic issue pick goes first; the rest keep their served order.
        j = draw(st.integers(0, len(pairs) - 1))
        pairs = [pairs[j]] + pairs[:j] + pairs[j + 1:]
    auto = draw(st.booleans())
    scenario = ("primary_only" if serving_path == "streamlit"
                else draw(st.sampled_from(PLATFORM_SCENARIOS)))
    return Outcome.model_validate({
        "chosen_team": {"id": team, "name": TABLES.teams[team]},
        "top_issues": [_issue_ref(r, i, p) for r, (i, p) in enumerate(pairs, 1)],
        "issue_order": issue_order, "auto_route": auto,
        "route": "auto" if auto else "review", "scenario": scenario})


def _strip_text_drivers(sec: dict) -> dict:
    """Drop ngram/sentence Drivers: with no complaint text in this clause none
    could be derivable (that check is Property 3's business)."""
    if "drivers" in sec:
        sec["drivers"] = [d for d in sec["drivers"]
                          if d["kind"] not in TEXT_DRIVER_KINDS]
    issue = (sec.get("details") or {}).get("issue")
    if isinstance(issue, dict) and "drivers" in issue:
        issue["drivers"] = [d for d in issue["drivers"]
                            if d["kind"] not in TEXT_DRIVER_KINDS]
    return sec


@st.composite
def _section_input(draw, name: str, serving_path: str):
    """What a caller may pass for one section: nothing, None, a skip, an
    error, or an ``ok`` section (as a Section model or a plain dict)."""
    if serving_path == "streamlit" and name not in STREAMLIT_RUN:
        kinds = ("omit", "none", "skipped")
    else:
        kinds = ("omit", "none", "skipped", "error", "ok", "ok_dict")
    kind = draw(st.sampled_from(kinds))
    if kind == "omit":
        return "omit"
    if kind == "none":
        return None
    if kind == "skipped":
        return skipped_section(name, draw(st_nonblank))
    if kind == "error":
        return error_section(name, RuntimeError(draw(st.text(max_size=40))))
    sec = _strip_text_drivers(draw(section_dicts(name, "ok")))
    if kind == "ok_dict":
        return sec
    return make_section(name, **{k: v for k, v in sec.items() if k != "explainer"})


@st.composite
def built_records(draw, serving_path: str | None = None):
    """``(record, served, inputs)``: a record from ``build_record`` together with
    the served outcome and the exact objects passed to the builder."""
    path = serving_path or draw(st.sampled_from(("streamlit", "platform")))
    order = ("probability" if path == "streamlit"
             else draw(st.sampled_from(("probability", "agentic"))))
    served = draw(served_outcomes(path, order))
    sections: dict = {}
    for name in SECTION_NAMES:
        sec = draw(_section_input(name, path))
        if sec != "omit":
            sections[name] = sec
    if order == "agentic":
        # The agentic chooser ran and picked the first issue.
        pick = served.top_issues[0].issue_id
        details = {"issue_choice": pick, "issue_status": "chosen",
                   "issue_candidates": [i.issue_id for i in served.top_issues]}
        sections["agentic"] = make_section("agentic", status="ok", method="exact",
                                           technique="rules", details=details)
    outcome_in = (served.model_dump(mode="python") if draw(st.booleans())
                  else served)
    inputs = {"outcome": outcome_in, "sections": sections,
              "model_versions": {"improved_model_1": "v1"}}
    snapshot = copy.deepcopy(inputs)
    record = build_record(request_id="r-prop1", serving_path=path,
                          model_versions=inputs["model_versions"],
                          outcome=outcome_in, sections=sections, served=served,
                          created_at="2026-01-01T00:00:00+00:00")
    return record, served, inputs, snapshot


# ---------------------------------------------------------------------------
# Builder clauses
# ---------------------------------------------------------------------------

# Feature: model-explainability, Property 1: Explaining never changes the decision
@settings(max_examples=100, deadline=None)
@given(built_records())
def test_builder_keeps_served_decision(case):
    record, served, inputs, snapshot = case
    assert_decision_unchanged(record, served)
    assert tuple(record.sections) == SECTION_NAMES
    # The objects passed in are not mutated.
    assert inputs == snapshot


def _perturb(served: Outcome, field: str) -> Outcome:
    d = served.model_dump(mode="python")
    if field == "auto_route":
        d["auto_route"] = not d["auto_route"]
        d["route"] = "auto" if d["auto_route"] else "review"
    elif field == "chosen_team":
        other = next(t for t in TEAM_IDS if t != d["chosen_team"]["id"])
        d["chosen_team"] = {"id": other, "name": TABLES.teams[other]}
    elif field == "prob":
        p = d["top_issues"][-1]["prob"]
        d["top_issues"][-1]["prob"] = p / 2 if p > 0 else 0.5
        if d["issue_order"] == "probability":
            d["top_issues"].sort(key=lambda i: -i["prob"])
            for r, i in enumerate(d["top_issues"], 1):
                i["rank"] = r
    elif field == "scenario":
        d["scenario"] = next(s for s in PLATFORM_SCENARIOS if s != d["scenario"])
    return Outcome.model_validate(d)


# Feature: model-explainability, Property 1: Explaining never changes the decision
@settings(max_examples=100, deadline=None)
@given(data=st.data(), path=st.sampled_from(("streamlit", "platform")),
       field=st.sampled_from(("auto_route", "chosen_team", "prob", "scenario")))
def test_builder_never_returns_a_different_decision(data, path, field):
    served = data.draw(served_outcomes(path, "probability"))
    if field == "prob" and not served.top_issues:
        field = "auto_route"
    if field == "scenario" and path == "streamlit":
        field = "chosen_team"
    other = _perturb(served, field)
    with pytest.raises(BuildError) as ei:
        build_record(request_id="r", serving_path=path, model_versions={},
                     outcome=other, sections={}, served=served)
    assert ei.value.error_type == "outcome_mismatch"


# ---------------------------------------------------------------------------
# Streamlit clause: synthetic IM1, served prediction vs record outcome
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def im1_bundle():
    m = synthetic_im1()
    return {"vectorizer": m.vectorizer, "model": m.hsvm,
            "team_classes": [str(c) for c in m.hsvm.calibrated_team_.classes_],
            "global_threshold": 0.5, "issue_labels": dict(issue_labels()),
            "model_name": "improved_model_1"}


def _streamlit_outcome(pred: dict) -> Outcome:
    """The served prediction dict in Decision_Record form (no re-ranking)."""
    return Outcome.model_validate({
        "chosen_team": {"id": pred["predicted_team_id"],
                        "name": pred["predicted_team_name"]},
        "top_issues": [{"rank": r, "issue_id": i["issue_id"],
                        "label": i["issue_label"],
                        "display_label": display_label(i["issue_label"]),
                        "prob": i["confidence"]}
                       for r, i in enumerate(pred["issue_top_k"], 1)],
        "issue_order": "probability", "auto_route": pred["route"] == "auto",
        "route": pred["route"], "scenario": "primary_only"})


def _boom(*_args, **_kwargs):
    raise RuntimeError("explainer failure")


def _model_state(bundle: dict) -> list[np.ndarray]:
    hsvm = bundle["model"]
    clfs = [hsvm.team_clf] + [c for c in hsvm.issue_clfs_.values()
                              if hasattr(c, "coef_")]
    return [a.copy() for c in clfs for a in (c.coef_, c.intercept_)]


# Feature: model-explainability, Property 1: Explaining never changes the decision
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=st_complaint, threshold=st.one_of(st.none(), st.floats(0.0, 1.0)),
       im1_fails=st.booleans())
def test_streamlit_record_matches_served_im1_prediction(im1_bundle, text,
                                                        threshold, im1_fails):
    bundle = dict(im1_bundle, global_threshold=threshold)
    pred = im1_predict(text, model=bundle)
    pred_snapshot = copy.deepcopy(pred)
    state = _model_state(bundle)
    served = _streamlit_outcome(pred)

    vec, hsvm = bundle["vectorizer"], bundle["model"]
    masked = mask(text)
    served_issues = [i["issue_id"] for i in pred["issue_top_k"]]
    im1 = (run_explainer("improved_model_1", _boom) if im1_fails else
           run_explainer("improved_model_1", explain_im1, vec, hsvm,
                         bundle["team_classes"], text, masked,
                         pred["predicted_team_id"], served_issues))
    routing = explain_routing_streamlit(pred["confidence"], threshold)
    record = build_record(request_id="r-st", serving_path="streamlit",
                          model_versions={"improved_model_1": "synthetic"},
                          outcome=served, served=served,
                          sections={"improved_model_1": im1, "routing": routing},
                          raw_text=text, masked_text=masked,
                          vectorizers={"improved_model_1": vec},
                          created_at="2026-01-01T00:00:00+00:00")

    assert_decision_unchanged(record, served)
    assert record.serving_path == "streamlit"
    assert record.outcome.scenario == "primary_only"
    assert record.sections["routing"].status == "ok"   # rule held as served
    # Explaining did not touch the served prediction or the fitted model.
    assert pred == pred_snapshot
    assert all(np.array_equal(a, b) for a, b in zip(state, _model_state(bundle)))
    assert im1_predict(text, model=bundle) == pred


# ---------------------------------------------------------------------------
# Narrate clause (Req 11.9): added by task 20.8.
# ---------------------------------------------------------------------------

import time

from src.explain.cache import ExplanationCache
from src.explain.config import NarratorConfig
from src.explain.narrator import Narrator

MOCK_MODES = ("faithful", "unfaithful", "failure_parse", "failure_timeout")


# Feature: model-explainability, Property 1: Explaining never changes the decision
@settings(max_examples=100, deadline=None)
@given(built_records())
def test_narrate_never_changes_the_decision(case):
    record, served, _inputs, _snapshot = case
    before = canonical_json(record.outcome)
    for mode in MOCK_MODES:
        narrator = Narrator(NarratorConfig(backend="mock", mock_mode=mode),
                            explanation_cache=ExplanationCache(),
                            prompt_cache=None)
        narrator.narrate(record, "masked text", deadline=time.monotonic() + 100)
        assert canonical_json(record.outcome) == before
        assert canonical_json(record.outcome) == canonical_json(served)
