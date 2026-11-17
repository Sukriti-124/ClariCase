"""IM1_Explainer: exact linear attribution for improved_model_1
(design §2, F5, Req 1.6, 1.12, 3.1-3.12).

improved_model_1 is a ``HierarchicalSVM``: a one-vs-rest LinearSVC over a
TF-IDF FeatureUnion for the team, isotonic-calibrated through
``CalibratedClassifierCV(FrozenEstimator(team_clf))``, plus one issue head per
team. The team Drivers explain the *uncalibrated* margin difference between
IM1's top team ``a`` and the Runner_Up ``b``:

    raw_diff = (x·W[a] + β_a) − (x·W[b] + β_b)
             = Σ_f x_f (W[a,f] − W[b,f]) + (β_a − β_b)

evaluated in float64 (F5). The served (possibly float32) difference is kept
for information only. The isotonic trace shows how ``a`` and ``b`` margins
become calibrated probabilities.

``linear_diff`` and ``ngram_drivers`` are shared with the issue sub-section.
``explain_im1`` raises on bad input; call it through ``run_explainer`` to get
an ``error`` section instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.explain.explainers import DEFAULT_N, NOMINAL, select_drivers, validate_n
from src.explain.record import CALIBRATION_NOTE, Driver, IM1Section, make_section
from src.explain.text import derivable_ngram, fold_redacted

__all__ = ["NAME", "EXACT_TOL", "UNCALIBRATED_SOURCE", "LinearDiff",
           "linear_diff", "ngram_drivers", "runner_up", "isotonic_trace",
           "uncalibrated_team_clf", "explain_issue", "explain_im1"]

NAME = "improved_model_1"
EXACT_TOL = 1e-6
# Recorded when calibrated_classifiers_[0].estimator.estimator is not
# hsvm.team_clf, naming where the explained LinearSVC was taken from.
UNCALIBRATED_SOURCE = "calibrated_team_.calibrated_classifiers_[0].estimator.estimator"


def _rows(vectorizer, text: str):
    """``(x_served, x64)``: the transform output as served (IM1's vectorizer
    emits float32) and a float64 CSR copy with duplicates summed (F5)."""
    served = vectorizer.transform([text])
    x = served.tocsr().astype(np.float64)
    x.sum_duplicates()
    return served, x


@dataclass(frozen=True)
class LinearDiff:
    """The decomposition of one linear score difference over a row ``x``."""
    cols: np.ndarray          # non-zero feature indices of x
    values: np.ndarray        # their TF-IDF values (float64)
    contribs: np.ndarray      # per-feature contributions (float64)
    constant: float
    raw_diff: float
    residual: float
    method: str               # "exact" | "approximate"


def linear_diff(x, coef, intercept, a: int, b: int | None = None,
                sign: float = 1.0) -> LinearDiff:
    """Decompose a linear score difference in float64 (Req 3.1, 3.4, 3.5, F5).

    With ``b`` given: ``(x·W[a] + β_a) − (x·W[b] + β_b)`` (multiclass). With
    ``b=None``: ``sign · (x·W[a] + β_a)`` (binary head, ``sign = ±1``).
    ``raw_diff`` is a sparse float64 matrix product, the contributions are
    evaluated per feature, so the residual compares two evaluations.
    """
    rows = [a] if b is None else [a, b]
    W = np.asarray(coef[rows], dtype=np.float64)              # (1|2, F)
    beta = np.asarray(intercept, dtype=np.float64).ravel()[rows]
    scores = np.asarray(x @ W.T).ravel() + beta
    cols = x.indices.astype(np.int64, copy=False)
    values = np.asarray(x.data, dtype=np.float64)
    if b is None:
        s = float(sign)
        contribs = values * (s * W[0, cols])
        constant = float(s * beta[0])
        raw = float(s * scores[0])
    else:
        contribs = values * (W[0, cols] - W[1, cols])
        constant = float(beta[0] - beta[1])
        raw = float(scores[0] - scores[1])
    residual = abs(math.fsum(contribs.tolist()) + constant - raw)
    method = "exact" if residual <= EXACT_TOL else "approximate"
    return LinearDiff(cols, values, contribs, constant, raw, float(residual), method)


def ngram_drivers(diff: LinearDiff, feature_names: Sequence[str],
                  masked_text: str, vectorizer, n: int
                  ) -> tuple[list[Driver], float | None]:
    """Top-N n-gram Drivers per sign plus the Redacted_Driver (Req 1.12, 3.2, 3.3).

    Every non-zero contribution is folded or kept before selection, so the
    Redacted_Driver carries all hidden features. Returns the Drivers and the
    redacted contribution (``None`` when nothing non-zero was folded).
    """
    full = [Driver(kind="ngram", label=str(feature_names[f]), value=float(v),
                   contribution=float(c))
            for f, v, c in zip(diff.cols.tolist(), diff.values.tolist(),
                               diff.contribs.tolist())
            if c != 0.0]
    kept, redacted = fold_redacted(
        full, lambda t: derivable_ngram(t, masked_text, vectorizer))
    drivers = select_drivers([d.contribution for d in kept],
                             [d.label for d in kept],
                             [d.value for d in kept], n)
    if redacted is None:
        return drivers, None
    drivers.append(redacted)
    return drivers, redacted.contribution


def runner_up(probs: np.ndarray, classes: Sequence[str], chosen: str) -> tuple[int, int]:
    """``(a, b)`` column indices: IM1's top team and the Runner_Up.

    ``a`` is the top column by probability (ties by class order). ``b`` is the
    second column when ``classes[a] == chosen``, else the Chosen_Team.
    """
    classes = [str(c) for c in classes]
    if chosen not in classes:
        raise KeyError(f"chosen_team {chosen!r} is not an improved_model_1 team")
    order = np.argsort(-np.asarray(probs, dtype=np.float64), kind="stable")
    a = int(order[0])
    b = int(order[1]) if classes[a] == chosen else classes.index(chosen)
    return a, b


def uncalibrated_team_clf(hsvm) -> tuple[object, str | None]:
    """The LinearSVC the calibrator wraps and, if it is not ``hsvm.team_clf``,
    the ``uncalibrated_source`` to record."""
    inner = hsvm.calibrated_team_.calibrated_classifiers_[0].estimator
    inner = getattr(inner, "estimator", inner)               # FrozenEstimator
    if inner is hsvm.team_clf:
        return inner, None
    return inner, UNCALIBRATED_SOURCE


def isotonic_trace(cal, svc_classes: Sequence[str], decision: np.ndarray
                   ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-class isotonic output and normalised probability (Req 3.10).

    Mirrors ``_CalibratedClassifier.predict_proba`` (scikit-learn 1.9.1) for
    one row of ``decision`` scores in ``svc_classes`` order. Returns the
    calibrated class order and, in that order, the pre-normalisation isotonic
    outputs and the final probabilities (uniform when they sum to 0).
    """
    cal_classes = [str(c) for c in cal.classes]
    k = len(cal_classes)
    if k < 3:
        raise ValueError("isotonic trace needs a multiclass team calibrator")
    pos = LabelEncoder().fit(cal.classes).transform(np.asarray(svc_classes))
    iso = np.zeros(k, dtype=np.float64)
    for p, d, calibrator in zip(pos, decision, cal.calibrators):
        iso[int(p)] = float(np.asarray(calibrator.predict(np.array([d]))).ravel()[0])
    total = iso.sum()
    prob = iso / total if total != 0 else np.full(k, 1.0 / k)
    return cal_classes, iso, prob


def _clip01(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def _issue_ranking(hsvm, x_served, team: str, served_issues: Sequence[str] | None
                   ) -> list[str]:
    """The served issue order for ``team``: ``served_issues`` when it names at
    least two issues, else the order ``hsvm.predict_issue_topk`` serves."""
    served = [str(i) for i in (served_issues or [])]
    if len(served) >= 2:
        return served
    own = [i for i, _ in hsvm.predict_issue_topk(x_served, [team], k=3)[0]]
    return served + [i for i in own if i not in served]


def explain_issue(hsvm, x_served, x, team: str, masked_text: str, vectorizer,
                  served_issues: Sequence[str] | None, n: int,
                  feature_names: Sequence[str] | None = None) -> dict:
    """The IM1 issue sub-section for ``team`` (design §2, Req 3.7-3.9, 3.12).

    Constant head: zero Drivers, ``technique=constant``. Multiclass head: served
    rank-1 ``u`` vs served rank-2 ``v``. Binary head: ``u`` vs the other class
    with ``s = +1`` when ``u == classes_[1]`` else ``-1``. Driver rules,
    exactness check, Redacted_Driver and served deviation match the team section.
    """
    clf = hsvm.issue_clfs_[team]
    if not hasattr(clf, "coef_"):                          # _ConstantIssue
        return dict(kind="constant", method="exact", technique="constant",
                    issue_id=str(clf.issue_id))

    classes = [str(c) for c in clf.classes_]
    ranking = _issue_ranking(hsvm, x_served, team, served_issues)
    u = ranking[0]
    if u not in classes:
        raise KeyError(f"served issue {u!r} is not an issue of team {team!r}")
    dec = np.asarray(clf.decision_function(x_served), dtype=np.float64).reshape(-1)
    coef = np.asarray(clf.coef_)
    if coef.shape[0] == 1:
        kind = "linear_binary"
        s = 1.0 if u == classes[1] else -1.0
        v = classes[0] if s > 0 else classes[1]
        diff = linear_diff(x, coef, clf.intercept_, 0, None, sign=s)
        served_diff = s * float(dec[0])
    else:
        kind = "linear_multiclass"
        v = next((i for i in ranking[1:] if i != u), None)
        if v is None or v not in classes:
            raise KeyError(f"no second-ranked issue of team {team!r} to compare against")
        iu, iv = classes.index(u), classes.index(v)
        diff = linear_diff(x, coef, clf.intercept_, iu, iv)
        served_diff = float(dec[iu]) - float(dec[iv])

    out = dict(
        kind=kind, method=diff.method, technique="linear", issue_id=u,
        compared_issue=v, drivers=[], constant=diff.constant,
        raw_diff=diff.raw_diff, residual=diff.residual, served_diff=served_diff,
        served_decision_deviation=abs(served_diff - diff.raw_diff),
        n_nonzero=int(x.nnz),
        ranking_source=("calibrated" if hsvm.calibrated_issue_.get(team) is not None
                        else "softmax(decision_function)"),
    )
    if x.nnz:
        names = (feature_names if feature_names is not None
                 else vectorizer.get_feature_names_out())
        out["drivers"], out["redacted_contribution"] = ngram_drivers(
            diff, names, masked_text, vectorizer, n)
    return out


def explain_im1(vectorizer, hsvm, team_classes: Sequence[str] | None,
                text: str, masked_text: str, chosen_team: str,
                served_issues: Sequence[str] | None = None, n: int = DEFAULT_N,
                feature_names: Sequence[str] | None = None) -> IM1Section:
    """Build the ``improved_model_1`` section (team decision and isotonic trace).

    ``team_classes`` is the served probability column order
    (``calibrated_team_.classes_``); when given it must match the calibrator.
    ``feature_names`` may be passed in to avoid recomputing
    ``vectorizer.get_feature_names_out()`` per request. ``served_issues`` is
    the served issue order for ``chosen_team``; with fewer than two entries
    the order is recomputed with ``hsvm.predict_issue_topk``.
    """
    n = validate_n(NAME, n)
    svc, source = uncalibrated_team_clf(hsvm)
    cal = hsvm.calibrated_team_.calibrated_classifiers_[0]
    svc_classes = [str(c) for c in svc.classes_]

    x_served, x = _rows(vectorizer, text)
    # The served scores, exactly as predict_proba computes them.
    decision = np.asarray(svc.decision_function(x_served)).reshape(-1)
    if decision.shape[0] != len(svc_classes):
        raise ValueError("binary team heads are not supported by the isotonic trace")

    cal_classes, iso, prob = isotonic_trace(cal, svc_classes, decision)
    if team_classes is not None and [str(t) for t in team_classes] != cal_classes:
        raise ValueError("team_classes do not match the team calibrator's classes")

    ca, cb = runner_up(prob, cal_classes, chosen_team)
    top, ru = cal_classes[ca], cal_classes[cb]
    a, b = svc_classes.index(top), svc_classes.index(ru)

    diff = linear_diff(x, svc.coef_, svc.intercept_, a, b)
    served_diff = float(decision[a]) - float(decision[b])

    details = dict(
        top_team=top, runner_up=ru, raw_diff=diff.raw_diff,
        constant=diff.constant, residual=diff.residual,
        served_diff=served_diff,
        served_decision_deviation=abs(served_diff - diff.raw_diff),
        n_nonzero=int(x.nnz), uncalibrated_source=source,
        isotonic={t: dict(raw_decision=float(decision[svc_classes.index(t)]),
                          isotonic_pre_norm=_clip01(iso[c]),
                          calibrated_prob=_clip01(prob[c]))
                  for t, c in ((top, ca), (ru, cb))},
        calibration_note=CALIBRATION_NOTE,
    )

    drivers: list[Driver] = []
    names = feature_names
    if x.nnz:
        if names is None:
            names = vectorizer.get_feature_names_out()
        drivers, redacted = ngram_drivers(diff, names, masked_text, vectorizer, n)
        details["redacted_contribution"] = redacted

    details["issue"] = explain_issue(hsvm, x_served, x, chosen_team, masked_text,
                                     vectorizer, served_issues, n, names)

    _, technique = NOMINAL[NAME]
    return make_section(NAME, status="ok", method=diff.method, technique=technique,
                        drivers=drivers, details=details)
