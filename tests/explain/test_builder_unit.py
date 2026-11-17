"""Unit tests for src/explain/builder.py (task 13.1, Req 1.1-1.12)."""
from __future__ import annotations

import pytest

from src.explain.builder import (BuildError, build_failure, build_record,
                                 check_no_raw_text)
from src.explain.explainers import error_section
from src.explain.explainers.routing import explain_routing_streamlit
from src.explain.labels import load_tables
from src.explain.record import SECTION_NAMES, Driver, Outcome, make_section

TABLES = load_tables()
TEAM = sorted(TABLES.teams)[0]
ISSUES = sorted(i for i, e in TABLES.issues.items() if e.team_id == TEAM)[:2]

RAW = "My card 4111 1111 1111 1111 was charged twice. The bank refused a refund."


def _outcome(auto: bool = True, scenario: str = "primary_only", **kw) -> Outcome:
    issues = [{"rank": r, "issue_id": i, "label": TABLES.issues[i].label,
               "display_label": TABLES.issues[i].display_label, "prob": p}
              for r, (i, p) in enumerate(zip(ISSUES, (0.7, 0.2)), start=1)]
    d = {"chosen_team": {"id": TEAM, "name": TABLES.teams[TEAM]},
         "top_issues": issues, "issue_order": "probability", "auto_route": auto,
         "route": "auto" if auto else "review", "scenario": scenario}
    d.update(kw)
    return Outcome.model_validate(d)


def _im1(*drivers: Driver):
    return make_section("improved_model_1", status="ok", method="exact",
                        technique="linear", drivers=list(drivers))


def _build(**kw):
    args = dict(request_id="r1", serving_path="streamlit",
                model_versions={"improved_model_1": "v1"}, outcome=_outcome(),
                sections={"improved_model_1": _im1(
                              Driver(kind="ngram", label="word__refund", contribution=0.5)),
                          "routing": explain_routing_streamlit(0.9, 0.5)},
                created_at="2026-01-01T00:00:00+00:00", raw_text=RAW)
    args.update(kw)
    return build_record(**args)


def test_streamlit_fills_skips_and_keeps_run_sections():
    rec = _build()
    assert tuple(rec.sections) == SECTION_NAMES
    assert rec.outcome == _outcome()
    for name in SECTION_NAMES:
        sec = rec.sections[name]
        if name in ("improved_model_1", "routing"):
            assert sec.status == "ok"
        else:
            assert (sec.status, sec.reason) == ("skipped",
                                                "not available on this serving path")
            assert sec.drivers == []


def test_platform_missing_agentic_is_not_consulted():
    rec = _build(serving_path="platform", outcome=_outcome(scenario="auto"))
    assert rec.sections["agentic"].reason == "not consulted"
    assert rec.sections["stacker"].reason == "not available on this serving path"


def test_error_section_is_kept_and_record_completes():
    sec = error_section("improved_model_1", RuntimeError("boom"))
    rec = _build(sections={"improved_model_1": sec,
                           "routing": explain_routing_streamlit(0.9, 0.5)})
    assert rec.sections["improved_model_1"].status == "error"
    assert rec.sections["improved_model_1"].reason == "RuntimeError: boom"


def test_broken_routing_rule_becomes_error_section():
    # Served auto_route=False, but the routing section says score >= threshold.
    rec = _build(outcome=_outcome(auto=False))
    assert rec.sections["routing"].status == "error"
    assert rec.sections["routing"].reason.startswith("RoutingRuleError")


def test_outcome_mismatch_raises():
    with pytest.raises(BuildError) as ei:
        _build(served=_outcome(auto=False, scenario="primary_only"))
    assert ei.value.error_type == "outcome_mismatch"


def test_wrong_team_name_raises_label_mismatch():
    bad = _outcome(chosen_team={"id": TEAM, "name": "Not the team"})
    with pytest.raises(BuildError) as ei:
        _build(outcome=bad, served=bad)
    assert ei.value.error_type == "label_mismatch"


def test_streamlit_scenario_must_be_primary_only():
    with pytest.raises(BuildError) as ei:
        _build(outcome=_outcome(scenario="auto"))
    assert ei.value.error_type == "schema"


def test_streamlit_rejects_non_skipped_platform_section():
    stacker = make_section("stacker", status="ok", method="exact", technique="linear")
    with pytest.raises(BuildError) as ei:
        _build(sections={"stacker": stacker})
    assert ei.value.error_type == "invalid_input"


@pytest.mark.parametrize("driver", [
    Driver(kind="ngram", label="word__4111", contribution=0.3),          # PII digit run
    Driver(kind="sentence", text="My card 4111 1111 1111 1111 was charged twice.",
           contribution=0.3),
    Driver(kind="ngram", label="word__mortgage", contribution=0.3),      # not in text
])
def test_non_derivable_or_raw_text_driver_raises(driver):
    with pytest.raises(BuildError) as ei:
        _build(sections={"improved_model_1": _im1(driver),
                         "routing": explain_routing_streamlit(0.9, 0.5)})
    assert ei.value.error_type == "raw_text"
    assert "4111" not in str(ei.value)


def test_masked_sentence_driver_is_accepted():
    from src.app.pii import mask
    sentence = mask(RAW).split(". ")[1]
    rec = _build(sections={"improved_model_1": _im1(
        Driver(kind="sentence", text=sentence, contribution=0.2)),
        "routing": explain_routing_streamlit(0.9, 0.5)})
    check_no_raw_text(rec, RAW, mask(RAW))


def test_build_failure_entry():
    f = build_failure("r9", BuildError("outcome_mismatch", "x"))
    assert set(f) == {"request_id", "error_type", "at"}
    assert (f["request_id"], f["error_type"]) == ("r9", "outcome_mismatch")
    assert build_failure("r9", KeyError("k"))["error_type"] == "KeyError"
