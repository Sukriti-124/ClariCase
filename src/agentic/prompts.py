"""Prompt builders. Both end with the instruction to answer with one id."""
from __future__ import annotations

from src.agentic import config as C
from src.agentic.examples import Example, clip
from src.jev_model.config import EXTRA_TEAMS, STRUCTURED_TEAMS, _team_description
from src.utils.labels import issue_labels, team_names

_BOUNDARIES = {**STRUCTURED_TEAMS, **EXTRA_TEAMS}


def team_card(team: str) -> str:
    b = _BOUNDARIES.get(team)
    if b is None:
        return _team_description(team)
    return (f"{team_names[team]}. Covers: {b['covers']} "
            f"Not for: {b['not_for']}")


def issue_card(issue: str) -> str:
    label = issue_labels.get(issue, issue)
    return label.split(" :: ", 1)[-1] if " :: " in label else label


def _examples_block(examples: list[Example]) -> str:
    if not examples:
        return "    (no examples)"
    return "\n".join(f'    * "{e.text}"' for e in examples)


def team_prompt(complaint: str, candidates: list[str],
                examples: dict[str, list[Example]]) -> str:
    lines = [
        "A consumer filed this complaint about a financial product. Decide "
        "which support team should handle it. The options below are the "
        "only possible answers; the first is the current model's choice. "
        "Redacted details appear as XXXX.",
        "",
        "Candidate teams:",
    ]
    for t in candidates:
        lines.append(f"- {t}: {team_card(t)}")
        lines.append("  Similar complaints handled by this team:")
        lines.append(_examples_block(examples.get(t, [])))
    lines += ["", f"Complaint: {clip(complaint, C.MAX_COMPLAINT_CHARS)}", "",
              f"Answer with the team id only ({', '.join(candidates)})."]
    return "\n".join(lines)


def issue_prompt(complaint: str, team: str, candidates: list[str],
                 examples: dict[str, list[Example]]) -> str:
    lines = [
        f"A consumer filed this complaint, handled by the {team_names[team]}. "
        "Decide which issue best describes it. The options below are the "
        "only possible answers; the first is the current model's choice. "
        "Redacted details appear as XXXX.",
        "",
        "Candidate issues:",
    ]
    for i in candidates:
        lines.append(f"- {i}: {issue_card(i)}")
        lines.append("  Similar complaints filed under this issue:")
        lines.append(_examples_block(examples.get(i, [])))
    lines += ["", f"Complaint: {clip(complaint, C.MAX_COMPLAINT_CHARS)}", "",
              f"Answer with the issue id only ({', '.join(candidates)})."]
    return "\n".join(lines)
