"""Agentic_Recorder (design §9, Req 10).

Copies what ensemble-api's LLM chooser was asked and chose (the
PredictResponse ``agentic`` dict) into the agentic section. Nothing is
invented: a key that was not returned is recorded as ``Absent``. The chooser
gives no rationale, and every section says so (``AgenticDetails.rationale``
defaults to ``AGENTIC_RATIONALE``).
"""
from __future__ import annotations

from typing import Any, Mapping

from src.explain.explainers import NOMINAL, skipped_section
from src.explain.record import Absent, Section, make_section

__all__ = ["COPIED_KEYS", "NOT_RETURNED", "NO_CHOICE", "SKIP_STREAMLIT",
           "SKIP_NOT_CONSULTED", "record_agentic"]

COPIED_KEYS: tuple[str, ...] = ("team_candidates", "team_choice", "rescued",
                                "issue_candidates", "issue_choice")
NOT_RETURNED = "not returned"
NO_CHOICE = "no choice returned"
SKIP_STREAMLIT = "not available on this serving path"
SKIP_NOT_CONSULTED = "not consulted"

_NAME = "agentic"


def _copy(value: Any) -> Any:
    """Verbatim copy; lists keep their order and values."""
    return list(value) if isinstance(value, (list, tuple)) else value


def _task_status(agentic: Mapping[str, Any], cands_key: str,
                 choice_key: str) -> tuple[str, str | None]:
    """``chosen`` / ``failed`` (no choice returned) / ``not_asked`` (Req 10.4-10.6)."""
    cands = agentic.get(cands_key)
    if not cands:                       # missing, null or empty: never asked
        return "not_asked", None
    if agentic.get(choice_key) is None:
        # The task's decision stayed with the ML (Req 10.4).
        return "failed", NO_CHOICE
    return "chosen", None


def record_agentic(agentic: Mapping[str, Any] | None, serving_path: str,
                   pre_llm_team: str | None) -> Section:
    """Build the agentic section.

    ``pre_llm_team`` is the team chosen before the LLM stage
    (``Decision.candidates[0]``), which ``team_confidence`` refers to (F9).
    """
    if serving_path == "streamlit":
        return skipped_section(_NAME, SKIP_STREAMLIT)
    if serving_path != "platform":
        raise ValueError(f"unknown serving path {serving_path!r}")
    if agentic is None:
        return skipped_section(_NAME, SKIP_NOT_CONSULTED)
    if not isinstance(agentic, Mapping):
        raise TypeError(f"agentic must be a mapping, got {type(agentic).__name__}")

    details: dict[str, Any] = {}
    for key in COPIED_KEYS:
        if key in agentic and agentic[key] is not None:
            details[key] = _copy(agentic[key])
        else:
            details[key] = Absent(reason=NOT_RETURNED)

    team_status, team_reason = _task_status(agentic, "team_candidates", "team_choice")
    issue_status, issue_reason = _task_status(agentic, "issue_candidates", "issue_choice")
    details.update(team_status=team_status, team_status_reason=team_reason,
                   issue_status=issue_status, issue_status_reason=issue_reason)

    choice = agentic.get("team_choice")
    if choice is not None and pre_llm_team is not None:
        changed = choice != pre_llm_team
        details["team_changed"] = changed
        if changed:
            # team_confidence belongs to the pre-LLM team, not the Chosen_Team.
            details["team_confidence_refers_to"] = pre_llm_team

    method, technique = NOMINAL[_NAME]
    return make_section(_NAME, status="ok", method=method, technique=technique,
                        details=details)
