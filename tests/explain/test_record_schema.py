"""Unit tests for the Decision_Record schema validators (task 2.2)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.explain.record import (SECTION_NAMES, Absent, AgenticDetails,
                                DecisionRecord, Driver, IM1Section,
                                IssueSubsection, Outcome, Section,
                                make_section)


def _skip(name: str, reason: str = "not available on this serving path"):
    return make_section(name, status="skipped", reason=reason,
                        method="exact", technique="rules")


def _outcome(**kw):
    base = dict(chosen_team={"id": "T01", "name": "Team One"},
                top_issues=[{"rank": 1, "issue_id": "T01_I001", "label": "a",
                             "display_label": "A", "prob": 0.7},
                            {"rank": 2, "issue_id": "T01_I002", "label": "b",
                             "display_label": "B", "prob": 0.2}],
                issue_order="probability", auto_route=True, route="auto",
                scenario="primary_only")
    base.update(kw)
    return base


def _record(**kw):
    base = dict(request_id="r1", created_at="2025-01-01T00:00:00.000Z",
                serving_path="streamlit", model_versions={"improved_model_1": "sha256:x"},
                outcome=_outcome(), sections={n: _skip(n) for n in SECTION_NAMES})
    base.update(kw)
    return DecisionRecord(**base)


REDACTED = {"kind": "redacted", "contribution": 0.25}
NGRAM = {"kind": "ngram", "label": "word__fee", "value": 0.3, "contribution": 0.1}


def test_valid_record_and_dict_round_trip():
    rec = _record()
    assert rec.schema_version == "1"
    again = DecisionRecord.model_validate(rec.model_dump())
    assert again == rec
    assert isinstance(again.sections["agentic"].details, AgenticDetails)


def test_models_are_frozen_and_forbid_extra():
    rec = _record()
    with pytest.raises(ValidationError):
        rec.request_id = "other"
    with pytest.raises(ValidationError):
        Absent(reason="x", extra_field=1)


def test_reason_non_empty_exactly_when_not_ok():
    with pytest.raises(ValidationError):
        make_section("gate", status="skipped", reason="", method="approximate",
                     technique="ablation")
    with pytest.raises(ValidationError):
        make_section("gate", status="ok", reason="why", method="approximate",
                     technique="ablation")
    make_section("gate", status="ok", method="approximate", technique="ablation")


def test_drivers_empty_when_not_ok():
    with pytest.raises(ValidationError):
        make_section("improved_model_1", status="error", reason="ValueError: x",
                     method="exact", technique="linear", drivers=[NGRAM])


def test_bare_section_and_mismatched_details_rejected():
    with pytest.raises(ValidationError):
        Section(explainer="gate", status="ok", method="exact", technique="linear")
    with pytest.raises(ValidationError):
        make_section("gate", status="ok", method="exact", technique="linear",
                     details={"top_team": "T01"})


def test_exactly_eight_sections():
    secs = {n: _skip(n) for n in SECTION_NAMES[:-1]}
    with pytest.raises(ValidationError):
        _record(sections=secs)
    with pytest.raises(ValidationError):
        _record(sections={**secs, "agentic": _skip("agentic"), "extra": _skip("gate")})
    with pytest.raises(ValidationError):
        _record(sections={**secs, "agentic": _skip("gate")})


def test_redacted_driver_rules():
    Driver(**REDACTED)
    with pytest.raises(ValidationError):
        Driver(kind="redacted", contribution=0.0)
    for field, val in [("label", "x"), ("text", "x"), ("value", 1.0),
                       ("span", {"index": 0, "start": 0, "end": 1})]:
        with pytest.raises(ValidationError):
            Driver(**{**REDACTED, field: val})
    ok = make_section("improved_model_1", status="ok", method="exact",
                      technique="linear", drivers=[NGRAM, REDACTED])
    assert ok.drivers[1].kind == "redacted"
    with pytest.raises(ValidationError):
        make_section("improved_model_1", status="ok", method="exact",
                     technique="linear", drivers=[REDACTED, REDACTED])


def test_issue_subsection_redacted_and_constant_rules():
    sub = dict(kind="linear_multiclass", method="exact", technique="linear",
               issue_id="T01_I001", compared_issue="T01_I002")
    IssueSubsection(**sub, drivers=[NGRAM, REDACTED])
    with pytest.raises(ValidationError):
        IssueSubsection(**sub, drivers=[REDACTED, REDACTED])
    with pytest.raises(ValidationError):
        IssueSubsection(kind="constant", method="exact", technique="constant",
                        issue_id="T01_I001", drivers=[NGRAM])
    with pytest.raises(ValidationError):
        IM1Section(explainer="improved_model_1", status="error", reason="E: x",
                   method="exact", technique="linear",
                   details={"issue": {**sub, "drivers": [NGRAM]}})


def test_probabilities_bounded_and_finite():
    bad = _outcome(top_issues=[{"rank": 1, "issue_id": "i", "label": "a",
                                "display_label": "A", "prob": 1.5}])
    with pytest.raises(ValidationError):
        Outcome(**bad)
    with pytest.raises(ValidationError):
        make_section("improved_model_2", status="skipped", reason="occlusion disabled",
                     method="approximate", technique="occlusion",
                     details={"team_probs": {"T01": -0.1}})
    with pytest.raises(ValidationError):
        Driver(kind="ngram", label="word__x", contribution=float("nan"))


def test_outcome_rules():
    with pytest.raises(ValidationError):
        Outcome(**_outcome(issue_order="random"))
    swapped = _outcome()["top_issues"][::-1]
    swapped = [{**d, "rank": i + 1} for i, d in enumerate(swapped)]
    with pytest.raises(ValidationError):
        Outcome(**_outcome(top_issues=swapped))
    assert Outcome(**_outcome(top_issues=swapped, issue_order="agentic"))
    with pytest.raises(ValidationError):
        Outcome(**_outcome(route="review"))


def test_scenario_includes_primary_only_and_streamlit_requires_it():
    assert _record().outcome.scenario == "primary_only"
    with pytest.raises(ValidationError):
        _record(outcome=_outcome(scenario="auto"))
    rec = _record(serving_path="platform", outcome=_outcome(scenario="auto"))
    assert rec.outcome.scenario == "auto"


def test_absent_values_in_details():
    sec = make_section("agentic", status="ok", method="exact", technique="rules",
                       details={"team_choice": {"absent": True, "reason": "not returned"}})
    assert isinstance(sec.details.team_choice, Absent)
    assert sec.details.rationale == "the agentic chooser provides no rationale"
