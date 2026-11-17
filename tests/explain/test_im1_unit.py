"""Unit tests for the IM1_Explainer team section and isotonic trace (task 5.1)."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from src.app.pii import mask
from src.explain.explainers import run_explainer
from src.explain.explainers.im1 import (EXACT_TOL, UNCALIBRATED_SOURCE,
                                        explain_im1, linear_diff, runner_up)
from src.explain.record import CALIBRATION_NOTE, IM1Section
from tests.explain.fixtures import TEAMS, make_text, synthetic_im1


@pytest.fixture(scope="module", params=[False, True], ids=["float64", "compact"])
def im1(request):
    return synthetic_im1(compact=request.param)


def _texts(n=6, seed=3):
    rng = random.Random(seed)
    return [make_text(rng, TEAMS[i % len(TEAMS)], pii_rate=0.0) for i in range(n)]


def _explain(m, text, chosen=None, masked=None, n=10):
    cal_classes = list(m.hsvm.calibrated_team_.classes_)
    if chosen is None:
        p = m.hsvm.team_proba(m.vectorizer.transform([text]))[0]
        chosen = str(cal_classes[int(np.argmax(p))])
    return explain_im1(m.vectorizer, m.hsvm, cal_classes, text,
                       text if masked is None else masked, chosen, [], n=n)


def test_team_section_is_exact_and_sums_to_raw_diff(im1):
    names = im1.vectorizer.get_feature_names_out()
    for text in _texts():
        sec = _explain(im1, text)
        d = sec.details
        assert isinstance(sec, IM1Section) and sec.status == "ok"
        assert (sec.method, sec.technique) == ("exact", "linear")
        assert d.residual <= EXACT_TOL and d.uncalibrated_source is None
        assert d.top_team != d.runner_up
        # Recompute raw_diff independently in float64 over all features.
        svc = im1.hsvm.team_clf
        cls = list(svc.classes_)
        a, b = cls.index(d.top_team), cls.index(d.runner_up)
        x = im1.vectorizer.transform([text]).toarray()[0].astype(np.float64)
        W = svc.coef_.astype(np.float64)
        beta = svc.intercept_.astype(np.float64)
        raw = (x @ W[a] + beta[a]) - (x @ W[b] + beta[b])
        assert d.raw_diff == pytest.approx(raw, abs=1e-9)
        assert d.constant == pytest.approx(beta[a] - beta[b], abs=0)
        full = x * (W[a] - W[b])
        assert math.fsum(full) + d.constant == pytest.approx(d.raw_diff, abs=1e-6)
        # served values are information only
        assert d.served_decision_deviation == pytest.approx(
            abs(d.served_diff - d.raw_diff), abs=1e-12)
        # every reported ngram is a real feature with its tf-idf value
        lookup = {str(names[i]): x[i] for i in np.flatnonzero(x)}
        for drv in sec.drivers:
            assert drv.kind == "ngram"
            assert drv.value == pytest.approx(lookup[drv.label])


def test_top_n_per_sign_and_ordering(im1):
    sec = _explain(im1, _texts(1)[0], n=2)
    pos = [d for d in sec.drivers if d.contribution > 0]
    neg = [d for d in sec.drivers if d.contribution < 0]
    assert len(pos) <= 2 and len(neg) <= 2
    keys = [(-abs(d.contribution), d.label) for d in sec.drivers]
    assert keys == sorted(keys)


def test_runner_up_rule():
    probs = np.array([0.2, 0.5, 0.3])
    assert runner_up(probs, ["A", "B", "C"], "B") == (1, 2)
    assert runner_up(probs, ["A", "B", "C"], "A") == (1, 0)
    assert runner_up(np.array([0.4, 0.4, 0.2]), ["A", "B", "C"], "A") == (0, 1)
    with pytest.raises(KeyError):
        runner_up(probs, ["A", "B", "C"], "Z")


def test_chosen_team_differs_from_top(im1):
    text = _texts(1)[0]
    p = im1.hsvm.team_proba(im1.vectorizer.transform([text]))[0]
    classes = [str(c) for c in im1.hsvm.calibrated_team_.classes_]
    top = classes[int(np.argmax(p))]
    other = next(t for t in classes if t != top)
    d = _explain(im1, text, chosen=other).details
    assert (d.top_team, d.runner_up) == (top, other)


def test_isotonic_trace_matches_predict_proba(im1):
    for text in _texts():
        x = im1.vectorizer.transform([text])
        p = im1.hsvm.team_proba(x)[0]
        classes = [str(c) for c in im1.hsvm.calibrated_team_.classes_]
        sec = _explain(im1, text)
        d = sec.details
        assert d.calibration_note == CALIBRATION_NOTE
        assert set(d.isotonic) == {d.top_team, d.runner_up}
        dec = im1.hsvm.team_clf.decision_function(x)[0]
        svc_cls = list(im1.hsvm.team_clf.classes_)
        for t, tr in d.isotonic.items():
            assert tr.calibrated_prob == pytest.approx(p[classes.index(t)], abs=1e-12)
            assert tr.raw_decision == pytest.approx(float(dec[svc_cls.index(t)]))
            assert 0.0 <= tr.isotonic_pre_norm <= 1.0


def test_empty_vector_has_no_drivers(im1):
    assert im1.vectorizer.transform([""]).nnz == 0
    sec = _explain(im1, "")
    d = sec.details
    assert d.n_nonzero == 0 and sec.drivers == []
    assert d.constant == d.raw_diff and d.residual == 0.0
    assert sec.method == "exact"


def test_non_derivable_ngrams_fold_into_redacted_driver(im1):
    text = "My mortgage escrow account was incorrect, call 555-123-4567 please."
    masked = mask(text)
    sec = _explain(im1, text, masked=masked)
    d = sec.details
    redacted = [x for x in sec.drivers if x.kind == "redacted"]
    assert len(redacted) <= 1
    for drv in sec.drivers:
        if drv.kind == "ngram":
            assert "555" not in drv.label and "4567" not in drv.label
    if redacted:
        assert redacted[0].contribution == d.redacted_contribution != 0
    assert d.residual <= EXACT_TOL            # folding never changes exactness
    # Masked text equal to the scored text: nothing is redacted.
    plain = _explain(im1, masked, masked=masked)
    assert all(x.kind == "ngram" for x in plain.drivers)


def test_uncalibrated_source_fallback():
    m = synthetic_im1()
    import copy
    m.hsvm.team_clf = copy.deepcopy(m.hsvm.team_clf)   # break the identity
    sec = _explain(m, _texts(1)[0])
    assert sec.details.uncalibrated_source == UNCALIBRATED_SOURCE
    assert sec.method == "exact"


def test_linear_diff_multiclass_and_binary_sign():
    m = synthetic_im1()
    x = m.vectorizer.transform([_texts(1)[0]]).tocsr().astype(np.float64)
    W = m.hsvm.team_clf.coef_
    beta = m.hsvm.team_clf.intercept_.copy()
    good = linear_diff(x, W, beta, 0, 1)
    assert good.method == "exact"
    # Binary sign rule: s * (x.W[a] + b_a)
    neg = linear_diff(x, W, beta, 0, None, sign=-1.0)
    assert neg.raw_diff == pytest.approx(-(float((x @ W[0]).sum()) + beta[0]))
    assert neg.constant == -beta[0] and neg.method == "exact"


def test_bad_inputs_become_error_sections(im1):
    sec = run_explainer("improved_model_1", explain_im1, im1.vectorizer, im1.hsvm,
                        None, "text", "text", "NOPE", [])
    assert sec.status == "error" and sec.reason.startswith("KeyError")
    with pytest.raises(ValueError):
        explain_im1(im1.vectorizer, im1.hsvm, None, "t", "t", TEAMS[0], [], n=0)


# ---- issue sub-section (task 5.2, Req 3.7-3.9, 3.12) ------------------------

def _issue(m, text, team, served=None):
    x = m.vectorizer.transform([text])
    if served is None:
        served = [i for i, _ in m.hsvm.predict_issue_topk(x, [team], k=3)[0]]
    return _explain_issue_section(m, text, team, served), x


def _explain_issue_section(m, text, team, served):
    cal_classes = list(m.hsvm.calibrated_team_.classes_)
    sec = explain_im1(m.vectorizer, m.hsvm, cal_classes, text, text, team, served)
    return sec.details.issue


def _multiclass_team(m):
    return next(t for t, c in m.hsvm.issue_clfs_.items()
                if hasattr(c, "coef_") and c.coef_.shape[0] > 1)


def test_constant_issue_head(im1):
    from src.improved_model_1.model import _ConstantIssue
    clf = im1.hsvm.issue_clfs_["T01"]
    assert isinstance(clf, _ConstantIssue)
    iss, _ = _issue(im1, _texts(1)[0], "T01")
    assert (iss.kind, iss.method, iss.technique) == ("constant", "exact", "constant")
    assert iss.issue_id == clf.issue_id and iss.drivers == []


def test_multiclass_issue_head_rank1_vs_rank2(im1):
    team = _multiclass_team(im1)
    clf = im1.hsvm.issue_clfs_[team]
    for text in _texts():
        iss, x = _issue(im1, text, team)
        served = [i for i, _ in im1.hsvm.predict_issue_topk(x, [team], k=3)[0]]
        assert iss.kind == "linear_multiclass" and iss.technique == "linear"
        assert (iss.issue_id, iss.compared_issue) == (served[0], served[1])
        assert iss.method == "exact" and iss.residual <= EXACT_TOL
        assert iss.ranking_source == "softmax(decision_function)"
        cls = [str(c) for c in clf.classes_]
        u, v = cls.index(served[0]), cls.index(served[1])
        xd = x.toarray()[0].astype(np.float64)
        W, b = clf.coef_.astype(np.float64), clf.intercept_.astype(np.float64)
        assert iss.raw_diff == pytest.approx((xd @ W[u] + b[u]) - (xd @ W[v] + b[v]),
                                             abs=1e-9)
        assert iss.constant == pytest.approx(b[u] - b[v], abs=0)
        total = math.fsum(d.contribution for d in iss.drivers)
        if len(iss.drivers) < iss.n_nonzero:          # some drivers not shown
            continue
        assert total + iss.constant == pytest.approx(iss.raw_diff, abs=1e-6)
        assert iss.served_decision_deviation == pytest.approx(
            abs(iss.served_diff - iss.raw_diff), abs=1e-12)


@pytest.mark.parametrize("positive", [True, False])
def test_binary_issue_head_sign_rule(im1, positive):
    clf = im1.hsvm.issue_clfs_["T02"]
    assert clf.coef_.shape[0] == 1
    cls = [str(c) for c in clf.classes_]
    u = cls[1] if positive else cls[0]
    text = _texts(1)[0]
    iss = _explain_issue_section(im1, text, "T02", [u])    # one served issue only
    s = 1.0 if positive else -1.0
    xd = im1.vectorizer.transform([text]).toarray()[0].astype(np.float64)
    w, b0 = clf.coef_[0].astype(np.float64), float(np.float64(clf.intercept_[0]))
    assert iss.kind == "linear_binary"
    assert (iss.issue_id, iss.compared_issue) == (u, cls[0] if positive else cls[1])
    assert iss.constant == pytest.approx(s * b0, abs=0)
    assert iss.raw_diff == pytest.approx(s * (xd @ w + b0), abs=1e-9)
    assert iss.method == "exact" and iss.residual <= EXACT_TOL
    for d in iss.drivers:
        if d.kind == "ngram":
            f = list(im1.vectorizer.get_feature_names_out()).index(d.label)
            assert d.contribution == pytest.approx(s * xd[f] * w[f])


def test_issue_ranking_source_calibrated():
    m = synthetic_im1()
    team = _multiclass_team(m)
    m.hsvm.fit_issue_calibrators(m.vectorizer.transform(m.texts), m.teams, m.issues,
                                 min_rows=1)
    if team not in m.hsvm.calibrated_issue_:
        pytest.skip("calibrator not fitted for this team")
    iss, _ = _issue(m, _texts(1)[0], team)
    assert iss.ranking_source == "calibrated"
