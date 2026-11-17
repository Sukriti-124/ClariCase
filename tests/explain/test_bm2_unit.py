"""Brief unit tests for the BM2_Explainer (task 6.1)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.explain.explainers import run_explainer
from src.explain.explainers.bm2 import (EXPLAINS, NO_MATCH_NOTE, explain_bm2,
                                        inner_pipeline)
from src.explain.text import derivable_ngram
from tests.explain.fixtures import TEAMS, synthetic_bm2

TEXT = "my mortgage escrow was incorrect and the foreclosure notice was late"


@pytest.fixture(scope="module")
def m():
    return synthetic_bm2()


def _jll(m, text):
    pipe = inner_pipeline(m.calibrated)
    nb = pipe.named_steps["clf"]
    x = pipe.named_steps["tfidf"].transform([text])
    return np.asarray(x @ nb.feature_log_prob_.T).ravel() + nb.class_log_prior_


def test_exact_section_and_runner_up(m):
    sec = explain_bm2(m.calibrated, TEXT, TEXT, "T01")
    d = sec.details
    assert (sec.status, sec.method, sec.technique) == ("ok", "exact", "nb_llr")
    jll = _jll(m, TEXT)
    order = [TEAMS[i] for i in np.argsort(-jll, kind="stable")]
    assert d.runner_up == (order[1] if order[0] == "T01" else order[0])
    assert d.constant == 0.0
    assert d.residual <= 1e-6
    assert math.isclose(d.raw_diff, jll[TEAMS.index("T01")] - jll[TEAMS.index(d.runner_up)],
                        abs_tol=1e-9)
    assert d.uncalibrated_llr == d.raw_diff
    assert d.explains == EXPLAINS
    p = m.calibrated.predict_proba([TEXT])[0][list(m.calibrated.classes_).index("T01")]
    assert math.isclose(d.calibrated_prob_chosen, p)
    assert sec.drivers and all(dr.kind == "ngram" for dr in sec.drivers)
    assert "mortgage" in {dr.label for dr in sec.drivers}


def test_runner_up_is_top_class_when_chosen_is_not_top(m):
    jll = _jll(m, TEXT)
    top = TEAMS[int(np.argmax(jll))]
    other = next(t for t in TEAMS if t != top)
    sec = explain_bm2(m.calibrated, TEXT, TEXT, other)
    assert sec.details.runner_up == top
    assert sec.details.raw_diff <= 0


def test_driver_order_and_n(m):
    sec = explain_bm2(m.calibrated, TEXT, TEXT, "T05", n=2)
    pos = [d.contribution for d in sec.drivers if d.contribution > 0]
    neg = [d.contribution for d in sec.drivers if d.contribution < 0]
    assert len(pos) <= 2 and len(neg) <= 2
    assert pos == sorted(pos, reverse=True) and neg == sorted(neg)
    with pytest.raises(ValueError):
        explain_bm2(m.calibrated, TEXT, TEXT, "T05", n=51)


def test_no_vocabulary_match(m):
    sec = explain_bm2(m.calibrated, "zzzqx", "zzzqx", "T03")
    assert sec.status == "ok" and sec.drivers == []
    assert sec.details.note == NO_MATCH_NOTE
    assert sec.details.n_nonzero == 0
    assert sec.details.constant == 0.0 and sec.details.residual == 0.0


def test_hidden_ngrams_fold_into_redacted_driver(m):
    masked = "my escrow was incorrect and the notice was late"
    full = explain_bm2(m.calibrated, TEXT, TEXT, "T01", n=50)
    sec = explain_bm2(m.calibrated, TEXT, masked, "T01", n=50)
    red = [d for d in sec.drivers if d.kind == "redacted"]
    assert len(red) == 1
    assert "mortgage" not in {d.label for d in sec.drivers}
    assert sec.details.residual == full.details.residual
    assert sec.method == full.method == "exact"
    # n=50 reports every non-zero n-gram of this short text in `full`.
    assert len(full.drivers) == full.details.n_nonzero
    tfidf = inner_pipeline(m.calibrated).named_steps["tfidf"]
    hidden = math.fsum(d.contribution for d in full.drivers
                       if not derivable_ngram(d.label, masked, tfidf))
    assert math.isclose(red[0].contribution, hidden, abs_tol=1e-9)
    assert sec.details.redacted_contribution == red[0].contribution


def test_unknown_team_becomes_error_section(m):
    with pytest.raises(KeyError):
        explain_bm2(m.calibrated, TEXT, TEXT, "T99")
    sec = run_explainer("baseline_model_2", explain_bm2, m.calibrated, TEXT, TEXT, "T99")
    assert sec.status == "error" and sec.drivers == []
    assert sec.reason.startswith("KeyError")
