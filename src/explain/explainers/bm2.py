"""BM2_Explainer: naive Bayes log-likelihood ratio for baseline_model_2
(design §3, Req 1.6, 1.12, 4.1-4.7).

baseline_model_2 is ``CalibratedClassifierCV(FrozenEstimator(Pipeline[tfidf,
clf]))`` with ``clf = MultinomialNB(alpha=0.1, fit_prior=False)``. The
Drivers explain the *uncalibrated* naive Bayes joint log-likelihood
difference between the Chosen_Team and the Runner_Up:

    jll_a - jll_b = sum_f x_f * (log P(f|a) - log P(f|b)) + (prior_a - prior_b)

``explain_bm2`` raises on bad input (``KeyError`` for an unknown
``chosen_team``); call it through ``run_explainer`` to get an ``error``
section instead (Req 4.6).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.explain.explainers import (DEFAULT_N, NOMINAL, select_drivers,
                                    validate_n)
from src.explain.record import BM2Section, Driver, make_section
from src.explain.text import derivable_ngram, fold_redacted

__all__ = ["NAME", "EXACT_TOL", "NO_MATCH_NOTE", "EXPLAINS", "inner_pipeline",
           "explain_bm2"]

NAME = "baseline_model_2"
EXACT_TOL = 1e-6
NO_MATCH_NOTE = "no vocabulary n-grams matched"
EXPLAINS = "uncalibrated_naive_bayes_score"


def inner_pipeline(calibrated):
    """The fitted ``Pipeline[tfidf, clf]`` inside the calibrated model."""
    return calibrated.calibrated_classifiers_[0].estimator.estimator


def _runner_up(jll: np.ndarray, a: int) -> int:
    """Top class by jll if it is not ``a``, else the second; ties by class order."""
    order = np.argsort(-jll, kind="stable")
    return int(order[1] if int(order[0]) == a else order[0])


def explain_bm2(calibrated, text: str, masked_text: str, chosen_team: str,
                n: int = DEFAULT_N,
                feature_names: Sequence[str] | None = None) -> BM2Section:
    """Build the ``baseline_model_2`` section for one complaint.

    ``text`` is the scored text, ``masked_text`` its Masked_Text (derivability
    of n-gram Drivers is checked against it, F6). ``feature_names`` may be
    passed in to avoid recomputing ``get_feature_names_out()`` per request.
    """
    n = validate_n(NAME, n)
    pipe = inner_pipeline(calibrated)
    tfidf = pipe.named_steps["tfidf"]
    nb = pipe.named_steps["clf"]

    classes = [str(c) for c in nb.classes_]
    if chosen_team not in classes:
        raise KeyError(f"chosen_team {chosen_team!r} is not a baseline_model_2 class")
    a = classes.index(chosen_team)

    x = tfidf.transform([text]).tocsr().astype(np.float64)
    x.sum_duplicates()
    flp = np.asarray(nb.feature_log_prob_, dtype=np.float64)
    prior = np.asarray(nb.class_log_prior_, dtype=np.float64)
    jll = np.asarray(x @ flp.T).ravel() + prior
    b = _runner_up(jll, a)

    constant = float(prior[a] - prior[b])
    raw_diff = float(jll[a] - jll[b])

    cols = x.indices
    weights = x.data
    contribs = weights * (flp[a, cols] - flp[b, cols])
    residual = abs(math.fsum(contribs.tolist()) + constant - raw_diff)
    method = "exact" if residual <= EXACT_TOL else "approximate"

    cal_classes = [str(c) for c in calibrated.classes_]
    p_chosen = float(calibrated.predict_proba([text])[0][cal_classes.index(chosen_team)])
    p_chosen = min(1.0, max(0.0, p_chosen))

    details = dict(chosen_team=chosen_team, runner_up=classes[b],
                   raw_diff=raw_diff, constant=constant, residual=residual,
                   n_nonzero=int(x.nnz), calibrated_prob_chosen=p_chosen,
                   uncalibrated_llr=raw_diff, explains=EXPLAINS)

    drivers: list[Driver] = []
    if x.nnz == 0:
        details["note"] = NO_MATCH_NOTE
    else:
        names = (feature_names if feature_names is not None
                 else tfidf.get_feature_names_out())
        full = [Driver(kind="ngram", label=str(names[f]), value=float(w),
                       contribution=float(c))
                for f, w, c in zip(cols.tolist(), weights.tolist(), contribs.tolist())
                if c != 0.0]
        kept, redacted = fold_redacted(
            full, lambda t: derivable_ngram(t, masked_text, tfidf))
        drivers = select_drivers([d.contribution for d in kept],
                                 [d.label for d in kept],
                                 [d.value for d in kept], n)
        if redacted is not None:
            drivers.append(redacted)
            details["redacted_contribution"] = redacted.contribution

    _, technique = NOMINAL[NAME]
    return make_section(NAME, status="ok", method=method, technique=technique,
                        drivers=drivers, details=details)
