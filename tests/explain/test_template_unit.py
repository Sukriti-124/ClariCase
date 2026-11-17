"""Unit tests for the Template_Explainer (src/explain/template.py).

Requirements 14.1, 14.2, 14.4, 14.5, 15.3, 15.4, 15.9.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings

from src.explain.narration import AGENT_MAX, print_narration
from src.explain.record import (SECTION_NAMES, DecisionRecord, canonical_json,
                                parse_record)
from src.explain.template import (CONSUMER_WORD_MAX, format_fixed,
                                  summary_facts, template_explanation,
                                  template_narration)
from tests.explain.strategies import records

_SETTINGS = settings(max_examples=60, deadline=None,
                     suppress_health_check=[HealthCheck.too_slow])


def _section(name: str, status: str = "skipped", **kw) -> dict:
    techniques = {"improved_model_1": "linear", "baseline_model_2": "nb_llr",
                  "improved_model_2": "occlusion", "knn_embed": "neighbours",
                  "stacker": "linear", "gate": "shap", "routing": "rules",
                  "agentic": "rules"}
    sec = {"explainer": name, "status": status, "method": "exact",
           "technique": techniques[name]}
    if status != "ok":
        sec["reason"] = "not run on this path"
    sec.update(kw)
    return sec


def _example(auto: bool = True, **sections) -> DecisionRecord:
    secs = {n: _section(n) for n in SECTION_NAMES}
    secs.update(sections)
    return DecisionRecord.model_validate({
        "request_id": "r-1", "created_at": "2025-01-01T00:00:00.000Z",
        "serving_path": "platform", "model_versions": {},
        "outcome": {
            "chosen_team": {"id": "T06", "name": "Mortgage Team"},
            "top_issues": [
                {"rank": 1, "issue_id": "T06_I001", "label": "Mortgage :: Trouble during payment process",
                 "display_label": "Trouble during payment process", "prob": 0.8781234},
                {"rank": 2, "issue_id": "T06_I002", "label": "Mortgage :: Struggling to pay mortgage",
                 "display_label": "Struggling to pay mortgage", "prob": 0.1}],
            "issue_order": "probability", "auto_route": auto,
            "route": "auto" if auto else "review", "scenario": "auto" if auto else "contested"},
        "sections": secs,
    })


IM1_OK = _section(
    "improved_model_1", "ok",
    drivers=[
        {"kind": "ngram", "label": "word__escrow", "value": 0.2, "contribution": 0.4321},
        {"kind": "ngram", "label": "char__ mor", "value": 0.1, "contribution": -0.0123},
        {"kind": "ngram", "label": 'word__say "hi"', "value": 0.1, "contribution": 0.9},
        {"kind": "ngram", "label": "word__small", "value": 0.1, "contribution": 0.001},
    ],
    details={"top_team": "T06", "runner_up": "T01"})
ROUTING_OK = _section(
    "routing", "ok",
    details={"score": 0.91, "threshold": 0.8,
             "model_teams": {"improved_model_1": "T06", "baseline_model_2": "T01"}})


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

def test_consumer_summary_states_team_outcome_and_issue():
    n = template_narration(_example(auto=True))
    assert n.consumer_summary == ("Your complaint was auto-routed to the Mortgage Team. "
                                  "The main issue we identified is: Trouble during payment process.")
    n = template_narration(_example(auto=False))
    assert n.consumer_summary.startswith(
        "Your complaint was sent to human review for the Mortgage Team.")


def test_agent_summary_lines_drivers_and_routing():
    n = template_narration(_example(improved_model_1=IM1_OK, routing=ROUTING_OK))
    lines = n.agent_summary.split("\n")
    assert lines[0].startswith("Outcome: the complaint was auto-routed to chosen team T06 (Mortgage Team)")
    assert "0.878" in lines[0] and "scenario auto" in lines[0]
    assert lines[1].startswith("improved_model_1 (exact linear): top team T06; runner-up T01")
    # top 3 by |contribution|; the n-gram with a quote character is not quoted
    assert 'an unquotable n-gram 0.900, "escrow" 0.432, " mor" -0.012' in lines[1]
    assert "small" not in lines[1]
    assert lines[2].startswith("routing (exact rules)")
    assert "baseline_model_2 T01, improved_model_1 T06" in lines[2]
    assert "models disagreeing with the chosen team: baseline_model_2" in lines[2]
    assert "score 0.910 >= threshold 0.800" in lines[2]
    assert "human review" not in n.agent_summary
    facts = {f.path: f.value for f in n.cited_facts}
    assert facts["outcome.chosen_team.id"] == "T06"
    assert facts["sections.improved_model_1.drivers.0.contribution"] == 0.4321
    assert facts["outcome.top_issues.0.prob"] == 0.878123      # canonical 6 dp value
    assert facts["sections.routing.details.threshold"] == 0.8


def test_sentence_driver_with_quote_uses_offsets():
    im2 = _section("improved_model_2", "ok", drivers=[
        {"kind": "sentence", "text": 'He said "no".', "span": {"index": 2, "start": 40, "end": 53},
         "contribution": -0.25}])
    n = template_narration(_example(improved_model_2=im2))
    assert "sentence 2 (characters 40 to 53) -0.250" in n.agent_summary
    assert '"no"' not in n.agent_summary


def test_skipped_and_error_sections_are_left_out():
    gate_err = _section("gate", "error", details={"score": 0.5, "threshold": 0.4})
    n = template_narration(_example(gate=gate_err))
    assert n.agent_summary.count("\n") == 0
    assert "gate" not in n.agent_summary
    assert not any(f.path.startswith("sections.") for f in n.cited_facts)


def test_summary_facts_from_outcome():
    f = summary_facts(_example(auto=False))
    assert (f.team, f.team_name, f.route, f.auto_route, f.scenario) == (
        "T06", "Mortgage Team", "review", False, "contested")
    assert [i["issue_id"] for i in f.issues] == ["T06_I001", "T06_I002"]


def test_template_explanation_fields():
    rec = _example()
    e = template_explanation(rec, "validator rejection",
                             [{"check": "number", "offending": "0.5", "reason": "not in record"}])
    assert e.source == "template" and e.fallback_reason == "validator_rejection"
    assert not e.validation.accepted and e.validation.failed[0].check == "number"
    assert re.fullmatch(r"[0-9a-f]{64}", e.record_hash)
    assert e.created_at == rec.created_at
    assert template_explanation(rec, "no_api_key").fallback_reason == "no_api_key"
    with pytest.raises(ValueError):
        template_explanation(rec, "bogus")


@pytest.mark.parametrize("v,places,want", [
    (0.4321, 3, "0.432"), (-0.0001, 3, "0.000"), (2.0045, 3, "2.004"),
    (7, 0, "7"), (1e300, 3, "1" + "0" * 300 + ".000")])
def test_format_fixed(v, places, want):
    assert format_fixed(v, places) == want


def test_template_has_no_backend_clock_or_randomness():
    import src.explain.template as t
    tree = ast.parse(Path(t.__file__).read_text(encoding="utf-8"))
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    banned = ("src.agentic", "src.explain.narrator", "time", "datetime", "random",
              "locale", "socket", "requests", "httpx", "urllib")
    assert not [m for m in mods if m.split(".")[0] in banned or m in banned
                or m.startswith("src.agentic")], mods


# ---------------------------------------------------------------------------
# Generated records: bounds and determinism
# ---------------------------------------------------------------------------

@_SETTINGS
@given(records())
def test_bounds_on_any_record(rec: DecisionRecord):
    n = template_narration(rec)
    assert len(n.consumer_summary.split()) <= CONSUMER_WORD_MAX
    assert len(n.agent_summary) <= AGENT_MAX
    assert len(n.cited_facts) <= 50
    # the consumer summary adds no numbers beyond what the outcome names contain
    out = rec.outcome
    rest = n.consumer_summary.replace(out.chosen_team.name.strip(), "")
    for iss in out.top_issues[:1]:
        for s in (iss.display_label.strip(), iss.label.strip()):
            if s:
                rest = rest.replace(s, "")
    assert not re.search(r"\d|%", rest)
    verb = "auto-routed" if out.auto_route else "sent to human review"
    assert verb in n.consumer_summary and verb in n.agent_summary.split("\n")[0]
    # one line after the outcome per ok section, and routing is never dropped
    ok = [s for s in SECTION_NAMES if rec.sections[s].status == "ok"]
    heads = [l.split(" (", 1)[0] for l in n.agent_summary.split("\n")[1:]]
    assert heads == [s for s in ok if s in heads]
    if "routing" in ok:
        assert "routing" in heads


@_SETTINGS
@given(records())
def test_deterministic_and_input_form_independent(rec: DecisionRecord):
    a = print_narration(template_narration(rec))
    b = print_narration(template_narration(rec))
    c = print_narration(template_narration(parse_record(canonical_json(rec))))
    assert a == b == c
