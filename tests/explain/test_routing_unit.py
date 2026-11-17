"""Unit tests for the Routing_Explainer (Req 9)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.ensemble.thresholds import route_mask
from src.explain.explainers.routing import (RULE, PlatformDecisionContext,
                                            explain_routing_platform,
                                            explain_routing_streamlit,
                                            routing_rule_holds)
from src.explain.record import Absent

MODELS = ["improved_model_1", "baseline_model_2", "improved_model_2", "knn_embed"]
TEAM_THR = {"T01": {"threshold": 0.9, "status": "raised"},
            "T11": {"threshold": None, "status": "never"}}


def bundle(base=0.8, team_thr=TEAM_THR, fb_base=0.7, min_rows=500):
    return SimpleNamespace(models=list(MODELS), base_threshold=base,
                           team_thresholds=team_thr,
                           fallback_policy={"base": fb_base, "teams": None},
                           prior=SimpleNamespace(min_rows=min_rows))


def decision(team, score, scenario, model_teams=None, prior_active=False,
             **extra):
    mt = model_teams if model_teams is not None else {m: team for m in MODELS}
    return SimpleNamespace(team=team, score=score, scenario=scenario,
                           model_teams=mt, prior_active=prior_active, **extra)


def served_auto(b, team, score):
    return bool(route_mask(np.array([score]), np.array([team]),
                           b.base_threshold, b.team_thresholds)[0])


def ctx(d, b, missing=(), agentic=None, rescue=None, pre_auto=None):
    pre = served_auto(b, d.team, d.score) if pre_auto is None else pre_auto
    return PlatformDecisionContext(decision=d, bundle=b, pre_llm_team=d.team,
                                   pre_llm_auto=pre, missing_models=missing,
                                   agentic_info=agentic, rescue_threshold=rescue)


def test_streamlit_reads_threshold_from_bundle():
    s = explain_routing_streamlit(0.85, 0.818)
    r = s.details
    assert (s.status, s.method, s.technique) == ("ok", "exact", "rules")
    assert r.score == 0.85 and r.threshold == 0.818 and r.threshold_kind == "base"
    assert r.rule == RULE and r.pre_rescue_auto is True
    assert explain_routing_streamlit(0.85, 0.9).details.pre_rescue_auto is False
    assert routing_rule_holds(s, final_auto=True)


def test_streamlit_missing_threshold_is_absent():
    r = explain_routing_streamlit(0.99, None).details
    assert isinstance(r.threshold, Absent) and r.pre_rescue_auto is False


@pytest.mark.parametrize("team,score,kind,thr", [
    ("T05", 0.85, "base", 0.8),
    ("T01", 0.85, "per-team", 0.9),
    ("T11", 0.99, "per-team", None),
])
def test_platform_threshold_kinds(team, score, kind, thr):
    b = bundle()
    auto = served_auto(b, team, score)
    d = decision(team, score, "auto" if auto else "assisted", prior_n_seen=10)
    r = explain_routing_platform(ctx(d, b)).details
    assert r.threshold_kind == kind and r.threshold == thr
    assert r.threshold_team == (team if kind == "per-team" else None)
    if thr is None:
        assert r.threshold_reason == "team is never auto-routed"
    assert r.pre_rescue_auto is auto
    assert r.pre_llm_team == team and r.model_teams == {m: team for m in MODELS}
    assert r.prior_n_seen == 10 and r.prior_min_rows == 500
    assert r.label_shift_active is False and r.missing_models is None
    assert routing_rule_holds(explain_routing_platform(ctx(d, b)), final_auto=auto)


def test_platform_contested_and_unknown_prior_count():
    mt = {m: "T05" for m in MODELS} | {"knn_embed": "T03"}
    d = decision("T05", 0.5, "contested", model_teams=mt, prior_active=True)
    s = explain_routing_platform(ctx(d, bundle()))
    assert isinstance(s.details.prior_n_seen, Absent)
    assert s.details.prior_min_rows is None
    assert routing_rule_holds(s, final_auto=False)


def test_degraded_records_fallback_and_missing_teams():
    b = bundle(fb_base=None)
    d = decision("T05", 0.95, "degraded", model_teams={"improved_model_1": "T05"},
                 prior_n_seen=0)
    s = explain_routing_platform(ctx(d, b, missing=["improved_model_2", "knn_embed"],
                                     pre_auto=False))
    r = s.details
    assert r.missing_models == ["improved_model_2", "knn_embed"]
    assert isinstance(r.fallback_threshold, Absent) and isinstance(r.threshold, Absent)
    assert r.primary_score == 0.95 and r.pre_rescue_auto is False
    assert isinstance(r.model_teams["knn_embed"], Absent)
    assert r.model_teams["improved_model_1"] == "T05"
    assert routing_rule_holds(s, final_auto=False)

    r2 = explain_routing_platform(ctx(d, bundle(fb_base=0.7),
                                      missing=["knn_embed"], pre_auto=True)).details
    assert r2.fallback_threshold == 0.7 and r2.threshold == 0.7 and r2.pre_rescue_auto


def test_agentic_rescue_and_change():
    b = bundle()
    d = decision("T05", 0.7, "auto", prior_n_seen=3)   # scenario after rescue
    info = {"team_candidates": ["T05", "T03"], "team_choice": "T05", "rescued": True}
    s = explain_routing_platform(ctx(d, b, agentic=info, rescue=0.6))
    r = s.details
    assert (r.rescued, r.llm_kept_team, r.pre_llm_auto, r.rescue_threshold) == \
        (True, True, False, 0.6)
    assert routing_rule_holds(s, final_auto=True)
    assert not routing_rule_holds(s, final_auto=False)

    d2 = decision("T03", 0.7, "assisted", model_teams={m: "T05" for m in MODELS})
    c2 = PlatformDecisionContext(decision=d2, bundle=b, pre_llm_team="T05",
                                 pre_llm_auto=False, rescue_threshold=None,
                                 agentic_info={"team_candidates": ["T05", "T03"],
                                               "team_choice": "T03", "rescued": False})
    r2 = explain_routing_platform(c2).details
    assert r2.llm_kept_team is False and r2.rescued is False
    assert isinstance(r2.rescue_threshold, Absent)
    assert routing_rule_holds(explain_routing_platform(c2), final_auto=False)


def test_rule_violations_are_detected():
    b = bundle()
    d = decision("T05", 0.85, "auto")
    assert not routing_rule_holds(explain_routing_platform(ctx(d, b, pre_auto=False)))
    bad = decision("T05", 0.5, "assisted", model_teams={m: "T03" for m in MODELS})
    assert not routing_rule_holds(explain_routing_platform(ctx(bad, b)))
    rescue_low = decision("T05", 0.5, "auto")
    info = {"team_candidates": ["T05"], "team_choice": "T05", "rescued": True}
    assert not routing_rule_holds(explain_routing_platform(
        ctx(rescue_low, b, agentic=info, rescue=0.6)))
