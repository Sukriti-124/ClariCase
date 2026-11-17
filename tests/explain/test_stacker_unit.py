"""Unit tests for the Stacker_Explainer (Req 7.1-7.8)."""
from __future__ import annotations

import numpy as np
import pytest

from src.ensemble.stack import stack_features, stacker_proba
from src.explain.explainers import run_explainer
from src.explain.explainers.stacker import REASON_NO_STACKER, explain_stacker
from src.explain.record import StackerSection
from tests.explain.fixtures import synthetic_stack_bundle


@pytest.fixture(scope="module")
def st():
    return synthetic_stack_bundle("logreg")


def _call(st, row=0, chosen=None, prior_active=False, p_adj=None, **kw):
    b = st.bundle
    models, teams = b["models"], b["teams"]
    mp = {m: st.probs[m][row] for m in models}
    p_raw = stacker_proba(b["stacker"], stack_features([st.probs[m][row:row + 1]
                                                        for m in models]), len(teams))[0]
    if chosen is None:
        chosen = teams[int(p_raw.argmax())]
    args = dict(stacker=b["stacker"], models=models, teams=teams, model_probs=mp,
                chosen_team=chosen, p_raw=p_raw, p_adj=p_adj,
                prior_active=prior_active, uses_stacker=True)
    args.update(kw)
    return explain_stacker(**args), p_raw


def test_agrees_case_is_exact_and_ranked(st):
    sec, p_raw = _call(st)
    assert isinstance(sec, StackerSection) and sec.status == "ok"
    assert (sec.method, sec.technique) == ("exact", "linear")
    d = sec.details
    teams = st.bundle["teams"]
    order = np.argsort(-p_raw, kind="stable")
    assert d.case == "stacker_agrees"
    assert d.comparison_team == teams[order[1]]
    contribs = [x.contribution for x in sec.drivers]
    assert [x.label for x in sec.drivers] == d.ranking
    assert contribs == sorted(contribs, reverse=True)
    assert abs(sum(contribs) + d.intercept_diff - d.logit_diff) <= 1e-6
    # Multinomial softmax: the logit gap is the log-probability ratio.
    assert d.logit_diff == pytest.approx(np.log(p_raw[order[0]] / p_raw[order[1]]), abs=1e-6)
    assert d.carrying_model == (d.ranking[0] if contribs[0] > 0 else None)
    assert d.label_shift_active is False and d.p_before is None


def test_disagrees_case_uses_stacker_top(st):
    teams = st.bundle["teams"]
    _, p_raw = _call(st)
    other = teams[int(np.argsort(-p_raw, kind="stable")[2])]
    sec, _ = _call(st, chosen=other)
    d = sec.details
    assert d.case == "stacker_disagrees"
    assert d.comparison_team == teams[int(p_raw.argmax())]
    assert d.logit_diff < 0
    assert abs(sum(x.contribution for x in sec.drivers) + d.intercept_diff - d.logit_diff) <= 1e-6


def test_label_shift_fields(st):
    teams = st.bundle["teams"]
    _, p_raw = _call(st)
    order = np.argsort(-p_raw, kind="stable")
    chosen = teams[order[1]]
    p_adj = p_raw.copy()
    p_adj[order[1]] = p_raw[order[0]] + 0.01     # correction lifts the chosen team
    p_adj /= p_adj.sum()
    sec, _ = _call(st, chosen=chosen, prior_active=True, p_adj=p_adj)
    d = sec.details
    assert d.label_shift_active is True
    assert len(d.p_before) == len(d.p_after) == 11
    assert d.top_before == teams[order[0]] and d.top_after == chosen
    assert d.shift_changed_top is True
    assert d.case == "stacker_disagrees"


def test_skip_without_stacker(st):
    sec, _ = _call(st, uses_stacker=False)
    assert sec.status == "skipped" and sec.reason == REASON_NO_STACKER
    assert sec.drivers == []


def test_skip_missing_model_vector(st):
    m = st.bundle["models"][1]
    mp = {k: st.probs[k][0] for k in st.bundle["models"] if k != m}
    sec, _ = _call(st, model_probs=mp)
    assert sec.status == "skipped" and m in sec.reason and sec.drivers == []


def test_skip_team_outside_fitted_classes(st):
    sec, _ = _call(st, chosen="NOT_A_TEAM")
    assert sec.status == "skipped" and "NOT_A_TEAM" in sec.reason


def test_bad_input_becomes_error_section(st):
    sec = run_explainer("stacker", _call_bad, st)
    assert sec.status == "error" and sec.drivers == []


def _call_bad(st):
    return _call(st, p_raw=np.ones(3))[0]
