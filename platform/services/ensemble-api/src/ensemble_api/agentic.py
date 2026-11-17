"""ensemble-api's use of agentic-api: ask the LLM only where the ML is
unsure, and never let a failed or slow call change the ML answer.

Team: on complaints the gate sent to review, ask among the stacker's top
candidates. The LLM's pick becomes the team. If it agrees with the stacker
and the gate score clears the rescue threshold fitted offline, the complaint
is auto-routed after all.

Issue: when svm-api's top-2 issue probabilities are within `gap`, ask among
its top candidates for the final team; the pick moves to the front.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import structlog

from claricase_platform_shared.contracts import ChooseResponse, IssueProb

log = structlog.get_logger()


def load_rescue_threshold(path: Path | None) -> float | None:
    if path is None or not Path(path).exists():
        log.warning("agentic_policy_missing", path=str(path))
        return None
    data = json.loads(Path(path).read_text())
    return (data.get("rescue_rule") or {}).get("threshold")


async def ask(client: httpx.AsyncClient, url: str, text: str, task: str,
              candidates: list[str], team: str | None, rid: str,
              timeout: float) -> ChooseResponse | None:
    try:
        resp = await client.post(f"{url.rstrip('/')}/choose", json={
            "text": text, "task": task, "candidates": candidates,
            "team": team, "request_id": rid}, timeout=timeout)
        if resp.status_code != 200:
            return None
        out = ChooseResponse(**resp.json())
        return out if out.choice in candidates else None
    except (httpx.HTTPError, ValueError):
        return None


def issue_candidates(top_issues: list[IssueProb], gap: float,
                     n: int) -> list[str] | None:
    """The ids to ask about, or None when the ML is sure enough (or has
    fewer than two issues)."""
    if len(top_issues) < 2:
        return None
    ranked = sorted(top_issues, key=lambda r: -r.prob)
    if ranked[0].prob - ranked[1].prob >= gap:
        return None
    return [r.issue for r in ranked[:n]]


def reorder_issues(top_issues: list[IssueProb], choice: str) -> list[IssueProb]:
    first = [r for r in top_issues if r.issue == choice]
    rest = [r for r in top_issues if r.issue != choice]
    return first + rest
