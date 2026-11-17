"""Property test for the BM2_Explainer exactness (task 6.2, design Property 10).

The expected values are recomputed here from the fitted MultinomialNB, not
read back from the section, so the test checks the explainer against the
model's own joint log-likelihood.
"""
from __future__ import annotations

import math

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.explain.explainers.bm2 import NO_MATCH_NOTE, explain_bm2, inner_pipeline
from tests.explain.fixtures import TEAMS, st_oov, st_text, synthetic_bm2

TOL = 1e-6
SEEDS = (0, 1)
_MODELS: dict[int, object] = {}


def _model(seed: int):
    """One synthetic calibrated BM2 per seed, built once for the module."""
    if seed not in _MODELS:
        _MODELS[seed] = synthetic_bm2(seed=seed)
    return _MODELS[seed]


def _parts(calibrated, text):
    pipe = inner_pipeline(calibrated)
    tfidf, nb = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    x = tfidf.transform([text]).tocsr().astype(np.float64)
    x.sum_duplicates()
    flp = np.asarray(nb.feature_log_prob_, dtype=np.float64)
    prior = np.asarray(nb.class_log_prior_, dtype=np.float64)
    jll = np.asarray(x @ flp.T).ravel() + prior
    return x, flp, prior, jll, [str(c) for c in nb.classes_]


st_oov_text = st.builds(
    lambda ws, end: " ".join(ws) + end,
    st.lists(st_oov, min_size=1, max_size=10),
    st.sampled_from([".", "!", "?", ""]))


# Feature: model-explainability, Property 10: BM2 exactness
@settings(max_examples=100, deadline=None)
@given(seed=st.sampled_from(SEEDS), text=st_text, chosen=st.sampled_from(TEAMS))
def test_bm2_exactness(seed, text, chosen):
    """**Validates: Requirements 4.1, 4.3, 4.5**"""
    m = _model(seed)
    x, flp, prior, jll, classes = _parts(m.calibrated, text)
    sec = explain_bm2(m.calibrated, text, text, chosen)
    d = sec.details
    a = classes.index(chosen)

    # Runner_Up: the top jll class unless it is the Chosen_Team, then the second.
    order = np.argsort(-jll, kind="stable")
    expected_b = int(order[1] if int(order[0]) == a else order[0])
    assert d.runner_up == classes[expected_b]
    b = expected_b

    # constant is the log-prior difference, zero with fit_prior=False.
    assert d.constant == 0.0
    assert prior[a] - prior[b] == 0.0

    # Sum over all non-zero features plus constant equals jll_a - jll_b.
    contribs = x.data * (flp[a, x.indices] - flp[b, x.indices])
    raw = float(jll[a] - jll[b])
    assert abs(math.fsum(contribs.tolist()) + d.constant - raw) <= TOL
    assert math.isclose(d.raw_diff, raw, abs_tol=1e-9)
    assert d.residual <= TOL
    assert (sec.status, sec.method) == ("ok", "exact")

    # Every reported n-gram Driver is one of those contributions.
    by_label = {}
    names = inner_pipeline(m.calibrated).named_steps["tfidf"].get_feature_names_out()
    for f, c in zip(x.indices.tolist(), contribs.tolist()):
        by_label[str(names[f])] = c
    for dr in sec.drivers:
        if dr.kind == "ngram":
            assert math.isclose(dr.contribution, by_label[dr.label], abs_tol=1e-12)

    if x.nnz == 0:
        assert sec.drivers == [] and d.note == NO_MATCH_NOTE


# Feature: model-explainability, Property 10: BM2 exactness
@settings(max_examples=100, deadline=None)
@given(seed=st.sampled_from(SEEDS), text=st_oov_text, chosen=st.sampled_from(TEAMS))
def test_bm2_no_vocabulary_match(seed, text, chosen):
    """**Validates: Requirements 4.5**"""
    m = _model(seed)
    x, *_ = _parts(m.calibrated, text)
    assume(x.nnz == 0)
    sec = explain_bm2(m.calibrated, text, text, chosen)
    assert sec.status == "ok"
    assert sec.drivers == []
    assert sec.details.note == NO_MATCH_NOTE
    assert sec.details.n_nonzero == 0
    assert sec.details.constant == 0.0
    assert abs(sec.details.raw_diff - sec.details.constant) <= TOL
