"""Property test for IM1 exactness for team and issue heads (task 5.3).

Every expected value is recomputed here in float64 from the fitted model's
coefficients and the vectorizer output, independently of ``linear_diff``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.explain.explainers.im1 import explain_im1
from tests.explain.fixtures import (BINARY_TEAM, CONSTANT_TEAM, TEAMS,
                                    st_text, synthetic_im1)

TOL = 1e-6
N_MAX = 100        # the largest N the IM1 explainer accepts per sign

# Built once per module: synthetic_im1 deep-copies, which is too slow per example.
MODELS = {False: synthetic_im1(compact=False), True: synthetic_im1(compact=True)}
FEATURE_NAMES = {k: [str(f) for f in m.vectorizer.get_feature_names_out()]
                 for k, m in MODELS.items()}

# Texts that vectorise to the empty vector, mixed in with ordinary complaints.
st_empty = st.sampled_from(["", " ", "\n", ".", "?!"])
st_any_text = st.one_of(st_text, st_text, st_text, st_empty)
# None means "the chosen team is IM1's own top team".
st_chosen = st.one_of(st.none(),
                      st.sampled_from([CONSTANT_TEAM, BINARY_TEAM]),
                      st.sampled_from(TEAMS))


def _dense(m, text: str) -> np.ndarray:
    return m.vectorizer.transform([text]).toarray()[0].astype(np.float64)


def _top_and_runner_up(probs, classes, chosen):
    order = sorted(range(len(classes)), key=lambda i: (-float(probs[i]), i))
    top = classes[order[0]]
    return top, (classes[order[1]] if top == chosen else chosen)


def _check_identity(d, method, drivers, names, x, w_diff, beta_diff,
                    served_diff):
    """Shared checks for one linear decomposition ``x · w_diff + beta_diff``.

    ``d`` holds raw_diff/constant/residual/served_*; ``method`` is the
    section's (team) or sub-section's (issue) method."""
    nz = np.flatnonzero(x)
    contribs = x[nz] * w_diff[nz]
    expected_raw = math.fsum(contribs.tolist()) + beta_diff

    assert d.constant == pytest.approx(beta_diff, abs=1e-12)
    assert d.raw_diff == pytest.approx(expected_raw, abs=1e-9)
    # The identity over all non-zero features (reported, unreported, redacted).
    assert abs(math.fsum(contribs.tolist()) + d.constant - d.raw_diff) <= TOL
    assert d.n_nonzero == len(nz)
    # method depends only on the float64 residual, never on served_diff.
    assert d.residual == pytest.approx(
        abs(math.fsum(contribs.tolist()) + d.constant - d.raw_diff), abs=1e-9)
    assert method == ("exact" if d.residual <= TOL else "approximate")
    assert method == "exact"
    # Served (possibly float32) difference is recorded for information.
    assert d.served_diff == pytest.approx(served_diff, abs=1e-9)
    assert d.served_decision_deviation == pytest.approx(
        abs(d.served_diff - d.raw_diff), abs=1e-12)

    if len(nz) == 0:
        assert drivers == []
        assert d.constant == d.raw_diff
        return
    lookup = {names[f]: (x[f], w_diff[f]) for f in nz}
    ngram = [dr for dr in drivers if dr.kind == "ngram"]
    for dr in ngram:
        v, w = lookup[dr.label]
        assert dr.value == pytest.approx(v, abs=1e-12)
        assert dr.contribution == pytest.approx(v * w, abs=1e-12)
    n_pos = sum(dr.contribution > 0 for dr in ngram)
    n_neg = sum(dr.contribution < 0 for dr in ngram)
    if n_pos < N_MAX and n_neg < N_MAX:   # nothing truncated: drivers cover all
        total = math.fsum(dr.contribution for dr in drivers)
        assert total + d.constant == pytest.approx(d.raw_diff, abs=TOL)


# Feature: model-explainability, Property 8: IM1 exactness for team and issue heads
@settings(max_examples=100, deadline=None)
@given(compact=st.booleans(), text=st_any_text, chosen=st_chosen)
def test_im1_exactness_team_and_issue(compact, text, chosen):
    """**Validates: Requirements 3.1, 3.4, 3.6, 3.7, 3.8, 3.9, 3.12**"""
    m = MODELS[compact]
    names = FEATURE_NAMES[compact]
    hsvm = m.hsvm
    x_served = m.vectorizer.transform([text])
    x = _dense(m, text)
    if not text.strip(" \n.?!"):
        assert x_served.nnz == 0          # the empty-vector case is exercised

    cal_classes = [str(c) for c in hsvm.calibrated_team_.classes_]
    probs = hsvm.team_proba(x_served)[0]
    if chosen is None:
        chosen = cal_classes[int(np.argmax(probs))]

    sec = explain_im1(m.vectorizer, hsvm, cal_classes, text, text, chosen, [],
                      n=N_MAX)
    d = sec.details
    assert sec.status == "ok"

    # ---- team section (Req 3.1, 3.4, 3.6, 3.12) ----
    top, ru = _top_and_runner_up(probs, cal_classes, chosen)
    assert (d.top_team, d.runner_up) == (top, ru)
    svc = hsvm.team_clf
    svc_classes = [str(c) for c in svc.classes_]
    a, b = svc_classes.index(top), svc_classes.index(ru)
    W = np.asarray(svc.coef_, dtype=np.float64)
    beta = np.asarray(svc.intercept_, dtype=np.float64)
    dec = np.asarray(svc.decision_function(x_served), dtype=np.float64)[0]
    _check_identity(d, sec.method, sec.drivers, names, x, W[a] - W[b],
                    float(beta[a] - beta[b]), float(dec[a] - dec[b]))
    assert sec.technique == "linear"

    # ---- issue sub-section (Req 3.7, 3.8, 3.9, 3.12) ----
    iss = d.issue
    clf = hsvm.issue_clfs_[chosen]
    if not hasattr(clf, "coef_"):
        assert chosen == CONSTANT_TEAM
        assert (iss.kind, iss.technique, iss.method) == ("constant", "constant", "exact")
        assert iss.issue_id == str(clf.issue_id)
        assert list(iss.drivers) == []
        return

    served = [i for i, _ in hsvm.predict_issue_topk(x_served, [chosen], k=3)[0]]
    classes = [str(c) for c in clf.classes_]
    Wi = np.asarray(clf.coef_, dtype=np.float64)
    bi = np.asarray(clf.intercept_, dtype=np.float64).ravel()
    deci = np.asarray(clf.decision_function(x_served), dtype=np.float64).reshape(-1)
    u = served[0]
    assert iss.technique == "linear" and iss.issue_id == u
    if Wi.shape[0] == 1:
        assert chosen == BINARY_TEAM and iss.kind == "linear_binary"
        s = 1.0 if u == classes[1] else -1.0
        assert iss.compared_issue == (classes[0] if s > 0 else classes[1])
        _check_identity(iss, iss.method, iss.drivers, names, x, s * Wi[0],
                        float(s * bi[0]), float(s * deci[0]))
    else:
        assert iss.kind == "linear_multiclass"
        v = served[1]
        assert iss.compared_issue == v
        iu, iv = classes.index(u), classes.index(v)
        _check_identity(iss, iss.method, iss.drivers, names, x, Wi[iu] - Wi[iv],
                        float(bi[iu] - bi[iv]), float(deci[iu] - deci[iv]))
