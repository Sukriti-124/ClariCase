"""Unit tests for the Agentic_Recorder (Req 10.1-10.7)."""
import pytest

from src.explain.explainers.agentic import record_agentic
from src.explain.record import AGENTIC_RATIONALE, Absent, AgenticSection


def test_streamlit_path_is_skipped():
    s = record_agentic({"team_candidates": ["a", "b"]}, "streamlit", "a")
    assert isinstance(s, AgenticSection)
    assert (s.status, s.reason) == ("skipped", "not available on this serving path")
    assert s.details.team_candidates is None
    assert s.details.rationale == AGENTIC_RATIONALE


def test_platform_not_consulted():
    s = record_agentic(None, "platform", "a")
    assert (s.status, s.reason) == ("skipped", "not consulted")
    d = s.details
    assert d.team_candidates is d.team_choice is d.rescued is None
    assert d.issue_candidates is d.issue_choice is None
    assert d.rationale == AGENTIC_RATIONALE


def test_team_chosen_and_changed_issue_not_asked():
    ag = {"team_candidates": ["t2", "t1", "t3"], "team_choice": "t1", "rescued": False}
    s = record_agentic(ag, "platform", "t2")
    d = s.details
    assert s.status == "ok" and s.reason is None
    assert d.team_candidates == ["t2", "t1", "t3"]
    assert (d.team_choice, d.rescued) == ("t1", False)
    assert d.team_status == "chosen" and d.team_status_reason is None
    assert d.team_changed is True and d.team_confidence_refers_to == "t2"
    assert d.issue_status == "not_asked"
    assert d.issue_candidates == Absent(reason="not returned")
    assert d.issue_choice == Absent(reason="not returned")
    assert d.rationale == AGENTIC_RATIONALE


def test_team_unchanged_has_no_confidence_note():
    s = record_agentic({"team_candidates": ["t1", "t2"], "team_choice": "t1",
                        "rescued": True}, "platform", "t1")
    assert s.details.team_changed is False
    assert s.details.team_confidence_refers_to is None
    assert s.details.rescued is True


def test_failed_task_records_no_choice_returned():
    s = record_agentic({"team_candidates": ["t1", "t2"],
                        "issue_candidates": ["i1", "i2"], "issue_choice": "i2"},
                       "platform", "t1")
    d = s.details
    assert (d.team_status, d.team_status_reason) == ("failed", "no choice returned")
    assert d.team_choice == Absent(reason="not returned")
    assert d.rescued == Absent(reason="not returned")
    assert d.team_changed is None
    assert d.issue_status == "chosen" and d.issue_choice == "i2"


def test_unknown_serving_path_raises():
    with pytest.raises(ValueError):
        record_agentic({}, "elsewhere", None)
