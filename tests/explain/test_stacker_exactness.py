"""Property test for the Stacker_Explainer (design Property 15).

**Validates: Requirements 7.1, 7.2, 7.3, 7.6**
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.ensemble.stack import make_stacker, stack_features, stacker_proba
from src.explain.explainers.stacker import explain_stacker
from tests.explain.fixtures import MODELS, TEAMS, synthetic_stack_bundle

T = len(TEAMS)


@lru_cache(maxsize=None)
def _stacker(m: int):
    """A multinomial stacker over the first ``m`` fixture models."""
    s = synthetic_stack_bundle("logreg")
    models = MODELS[:m]
    est = make_stacker(1.0).fit(stack_features([s.probs[k] for k in models]), s.y)
    return est, models


# A probability vector over the 11 teams; exact zeros are allowed, and the
# exponent makes some vectors peaked and others flat.
st_prob = st.builds(
    lambda w, k: (lambda v: v / v.sum())(np.asarray(w, np.float64) ** k + 1e-12),
    st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=T, max_size=T),
    st.sampled_from([0.5, 1.0, 3.0, 8.0]))


@st.composite
def st_case(draw):
    m = draw(st.integers(1, len(MODELS)))
    vecs = [draw(st_prob) for _ in range(m)]
    chosen = draw(st.sampled_from(TEAMS))
    return m, vecs, chosen


# Feature: model-explainability, Property 15: Stacker exactness and ranking
@settings(max_examples=100, deadline=None)
@given(st_case())
def test_stacker_exactness_and_ranking(case):
    m, vecs, chosen = case
    stacker, models = _stacker(m)
    p_raw = stacker_proba(stacker, stack_features([v[None, :] for v in vecs]), T)[0]

    sec = explain_stacker(stacker=stacker, models=models, teams=TEAMS,
                          model_probs=dict(zip(models, vecs)), chosen_team=chosen,
                          p_raw=p_raw, p_adj=None, prior_active=False,
                          uses_stacker=True)
    assert sec.status == "ok"
    assert (sec.method, sec.technique) == ("exact", "linear")
    d = sec.details

    # 7.1: agree/disagree rule (ties broken by lower team index).
    order = np.argsort(-p_raw, kind="stable")
    a = TEAMS.index(chosen)
    if order[0] == a:
        assert d.case == "stacker_agrees"
        b = int(order[1])
    else:
        assert d.case == "stacker_disagrees"
        b = int(order[0])
    assert d.chosen_team == chosen and d.comparison_team == TEAMS[b]

    # 7.2 / 7.3: one contribution per model, summing exactly to the logit gap.
    assert sorted(x.label for x in sec.drivers) == sorted(models)
    contribs = [x.contribution for x in sec.drivers]
    assert abs(sum(contribs) + d.intercept_diff - d.logit_diff) <= 1e-6
    assert d.residual <= 1e-6
    assert abs(d.logit_diff - np.log(p_raw[a] / p_raw[b])) <= 1e-6

    # 7.6: ranking non-increasing; carrying model only when positive.
    assert [x.label for x in sec.drivers] == d.ranking
    assert all(contribs[i] >= contribs[i + 1] for i in range(len(contribs) - 1))
    assert d.carrying_model == (d.ranking[0] if contribs[0] > 0 else None)
