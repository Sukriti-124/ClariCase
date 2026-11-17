"""Adapters from other pipeline outputs to Decision_Record parts (Req 14.3).

``record_from_trace`` turns a ``scripts/trace_prediction.py`` results dict
into a partial record holding only ``outcome`` and the ``routing`` section, so
that ``trace_prediction.summarize`` and the Template_Explainer read the chosen
team, top-3 issues, route and scenario through the same
``src.explain.template.summary_facts``.
"""
from __future__ import annotations

from typing import Any

from .record import Outcome, make_section

TRACE_PRIMARY_ONLY = "primary-only"
MAX_TOP_ISSUES = 3


def _scenario(value: Any) -> str:
    """Trace scenario -> record scenario; ``primary-only`` gets an underscore."""
    s = str(value)
    return "primary_only" if s == TRACE_PRIMARY_ONLY else s


def _opt_float(x: Any) -> float | None:
    return None if x is None else float(x)


def _trace_decision(stages: dict) -> dict:
    """Team, route, scenario, score and threshold as ``summarize`` derives them."""
    im1 = stages["improved_model_1"]
    ens = stages.get("ensemble") or {}
    if ens.get("status") == "ok":
        return {"team": str(ens["team"]), "auto": bool(ens.get("auto_route")),
                "scenario": _scenario(ens.get("scenario")),
                "score": _opt_float(ens.get("score")),
                "threshold": _opt_float(ens.get("threshold")),
                "model_teams": {str(k): str(v)
                                for k, v in (ens.get("model_teams") or {}).items()},
                "missing": [str(m) for m in ens.get("missing") or []],
                "path": ens.get("path")}
    return {"team": str(im1["team"]), "auto": im1["route"] == "auto",
            "scenario": _scenario(TRACE_PRIMARY_ONLY),
            "score": _opt_float(im1.get("confidence")),
            "threshold": _opt_float(im1.get("global_threshold")),
            "model_teams": {"improved_model_1": str(im1["team"])},
            "missing": [], "path": None}


def record_from_trace(results: dict, names: dict | None = None) -> dict:
    """Partial Decision_Record (``outcome`` + ``sections.routing``) for a trace.

    ``names`` maps team_id to team name; missing names become ``""``. Only the
    first three issues of the issues stage are recorded (``Outcome`` holds at
    most three).
    """
    names = names or {}
    stages = results["stages"]
    d = _trace_decision(stages)
    route = "auto" if d["auto"] else "review"

    trace_issues = (stages.get("issues") or {}).get("issues") or []
    top_issues = []
    for rank, i in enumerate(trace_issues[:MAX_TOP_ISSUES], start=1):
        label = str(i.get("issue_label") or "")
        top_issues.append({"rank": rank, "issue_id": str(i["issue_id"]),
                           "label": label,
                           "display_label": label or str(i["issue_id"]),
                           "prob": float(i["prob"])})

    outcome = Outcome(
        chosen_team={"id": d["team"], "name": str(names.get(d["team"], ""))},
        top_issues=top_issues, issue_order="probability",
        auto_route=d["auto"], route=route, scenario=d["scenario"])

    details: dict[str, Any] = {"score": d["score"], "threshold": d["threshold"],
                               "scenario": d["scenario"],
                               "model_teams": d["model_teams"]}
    if d["path"] == "fallback":
        details.update(missing_models=d["missing"],
                       fallback_threshold=d["threshold"],
                       primary_score=d["score"])
    routing = make_section("routing", status="ok", method="exact",
                           technique="rules", details=details)
    return {"outcome": outcome, "sections": {"routing": routing}}
