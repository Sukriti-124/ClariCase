"""Routing_Explainer (design §8, Req 9).

Records the routing rule as it was applied: the score, the threshold (and
whether it was the base or a per-team threshold), the rule text and the
comparison result before any agentic rescue. On the Platform path it also
records the scenario, each model's top team, the pre-LLM team (F11), the
label-shift status, the degraded-mode facts and the agentic rescue facts.

Values that are not available are recorded as ``Absent`` with a reason; no
default is substituted (Req 9.8). Nothing here changes the served decision.

``routing_rule_holds`` is the pure invariant check of Req 9.7, used by the
Record_Builder (a failure becomes an ``error`` section) and by Property 18.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.explain.explainers import NOMINAL
from src.explain.record import Absent, RoutingSection, make_section

__all__ = ["RULE", "PlatformDecisionContext", "explain_routing_platform",
           "explain_routing_streamlit", "routing_rule_holds"]

RULE = "auto-route if and only if score >= threshold"
NAME = "routing"

R_NEVER = "team is never auto-routed"
R_NO_BASE = "bundle has no base threshold"
R_NO_FALLBACK = "no fallback threshold is configured"
R_NO_GLOBAL = "model bundle has no global threshold"
R_NO_RESCUE = "no rescue threshold is configured"
R_NO_TEAM = "model top team was not returned"
R_NO_N_SEEN = "decision did not record the rolling-prior row count"
R_NO_PRE_LLM = "pre-LLM team was not recorded"


@dataclass(frozen=True)
class PlatformDecisionContext:
    """Everything ``routes.predict`` knows about one Platform decision.

    ``decision`` is the ``stack_model.Decision`` after the LLM stage (the
    LLM stage may change ``team``, ``auto_route`` and ``scenario``), so the
    pre-LLM team and auto-route flag are copied before it (F11). ``bundle`` is
    the ``StackBundle``. ``rescue_threshold`` is ``app.state.rescue_threshold``.
    The remaining fields are for the other explainers.
    """
    decision: Any
    bundle: Any
    pre_llm_team: str | None
    pre_llm_auto: bool
    missing_models: Sequence[str] = ()
    agentic_info: Mapping[str, Any] | None = None
    rescue_threshold: float | None = None
    text: str = ""
    request_id: str = ""
    probs: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    served_issues: Sequence[str] = ()
    svm_team: str | None = None
    response: Mapping[str, Any] | None = None


def _num(x: Any) -> float | None:
    """``x`` as a finite float, or None when it is not a number."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _threshold(team: str | None, base: Any, teams: Mapping | None,
               no_base_reason: str) -> dict:
    """Threshold fields for ``team`` under a ``route_mask`` policy.

    Mirrors ``src.ensemble.thresholds.route_mask``: a ``None`` base routes
    nothing; a team listed in ``teams`` uses its own threshold, where
    ``None`` means "never".
    """
    teams = teams or {}
    if base is None:
        return {"threshold": Absent(reason=no_base_reason),
                "threshold_reason": no_base_reason, "threshold_kind": "base"}
    if team is not None and team in teams:
        info = teams[team]
        # Bundles store ``{"threshold": t, "status": ...}``; a bare value is
        # accepted too.
        t = info.get("threshold") if isinstance(info, Mapping) else info
        out = {"threshold_kind": "per-team", "threshold_team": team}
        if t is None:
            return {**out, "threshold": None, "threshold_reason": R_NEVER}
        return {**out, "threshold": float(t)}
    return {"threshold": float(base), "threshold_kind": "base"}


def _rule_result(score: Any, threshold: Any) -> bool:
    s, t = _num(score), _num(threshold)
    return s is not None and t is not None and s >= t


def _section(details: dict) -> RoutingSection:
    method, technique = NOMINAL[NAME]
    return make_section(NAME, status="ok", method=method, technique=technique,
                        details=details)


def explain_routing_streamlit(confidence: float,
                              global_threshold: float | None) -> RoutingSection:
    """Streamlit path: IM1 confidence against ``bundle["global_threshold"]``.

    The caller passes the threshold read from the loaded bundle (Req 9.5).
    """
    thr = _threshold(None, global_threshold, None, R_NO_GLOBAL)
    return _section({"score": float(confidence), **thr, "rule": RULE,
                     "pre_rescue_auto": _rule_result(confidence, thr["threshold"])})


def _label_shift(d: Any, bundle: Any) -> dict:
    out: dict = {"label_shift_active": bool(d.prior_active)}
    n_seen = getattr(d, "prior_n_seen", None)
    min_rows = getattr(d, "prior_min_rows", None)
    if min_rows is None:
        prior = getattr(bundle, "prior", None)
        min_rows = getattr(prior, "min_rows", None)
    if n_seen is None:
        out["prior_n_seen"] = Absent(reason=R_NO_N_SEEN)
        return out
    out["prior_n_seen"] = int(n_seen)
    if not d.prior_active and min_rows is not None and int(n_seen) < int(min_rows):
        out["prior_min_rows"] = int(min_rows)
    return out


def _agentic(ctx: PlatformDecisionContext, score: float) -> dict:
    info = ctx.agentic_info or {}
    if "team_candidates" not in info:      # the team chooser did not run
        return {}
    out: dict = {"pre_llm_auto": bool(ctx.pre_llm_auto)}
    thr = _num(ctx.rescue_threshold)
    out["rescue_threshold"] = thr if thr is not None else Absent(reason=R_NO_RESCUE)
    if "team_choice" in info:
        out["llm_kept_team"] = info["team_choice"] == ctx.pre_llm_team
    # A failed call leaves the auto-route flag unchanged, so no rescue happened.
    out["rescued"] = bool(info.get("rescued", False))
    return out


def explain_routing_platform(ctx: PlatformDecisionContext) -> RoutingSection:
    """Platform path: the bundle's routing rule as applied to ``ctx.decision``."""
    d, bundle = ctx.decision, ctx.bundle
    missing = list(ctx.missing_models or ())
    degraded = bool(missing) or d.scenario == "degraded"
    team = ctx.pre_llm_team
    score = float(d.score)

    if degraded:
        policy = bundle.fallback_policy
        thr = _threshold(team, policy.get("base"), policy.get("teams"), R_NO_FALLBACK)
    else:
        thr = _threshold(team, bundle.base_threshold, bundle.team_thresholds, R_NO_BASE)

    model_teams: dict = {}
    recorded = d.model_teams or {}
    for m in list(bundle.models) + [m for m in recorded if m not in bundle.models]:
        t = recorded.get(m)
        model_teams[m] = str(t) if t is not None else Absent(reason=R_NO_TEAM)

    details: dict = {
        "score": score, **thr, "rule": RULE,
        "pre_rescue_auto": bool(ctx.pre_llm_auto),
        "scenario": d.scenario,
        "model_teams": model_teams,
        "pre_llm_team": team if team is not None else Absent(reason=R_NO_PRE_LLM),
        **_label_shift(d, bundle),
    }
    if degraded:
        fb = _num(bundle.fallback_policy.get("base"))
        details["missing_models"] = missing
        details["fallback_threshold"] = (fb if fb is not None
                                         else Absent(reason=R_NO_FALLBACK))
        details["primary_score"] = score
    details.update(_agentic(ctx, score))
    return _section(details)


def routing_rule_holds(section: RoutingSection, final_auto: bool | None = None) -> bool:
    """The routing invariant of Req 9.7 (and the scenario checks of Req 9.2).

    * "threshold present and score >= threshold" reproduces ``pre_rescue_auto``;
    * a rescue implies the LLM kept the pre-LLM team, the score is >= the
      rescue threshold and (when ``final_auto`` is given) the final flag is true;
    * without a rescue, ``final_auto`` (when given) equals ``pre_rescue_auto``;
    * ``assisted`` implies every recorded model team equals the pre-LLM team
      and ``pre_rescue_auto`` is false; ``contested`` implies one differs.

    Only ``ok`` sections are checked; other sections trivially hold.
    """
    if section.status != "ok":
        return True
    r = section.details
    if r.pre_rescue_auto is None:
        return False
    if _rule_result(r.score, r.threshold) != r.pre_rescue_auto:
        return False
    if r.rescued:
        if r.llm_kept_team is not True:
            return False
        if not _rule_result(r.score, r.rescue_threshold):
            return False
        if final_auto is not None and final_auto is not True:
            return False
    elif final_auto is not None and final_auto != r.pre_rescue_auto:
        return False
    if r.scenario in ("assisted", "contested"):
        pre = r.pre_llm_team
        if not isinstance(pre, str) or r.model_teams is None:
            return False
        teams = [t for t in r.model_teams.values() if isinstance(t, str)]
        if r.scenario == "assisted":
            if r.pre_rescue_auto or any(t != pre for t in teams):
                return False
        elif not any(t != pre for t in teams):
            return False
    return True
