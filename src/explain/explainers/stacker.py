"""Stacker_Explainer: per-model contributions to the stacker logit difference
(design §6, Req 7).

The stacker is ``Pipeline[StandardScaler, LogisticRegression]`` over
``stack_features(mats)``: one block of ``len(teams)`` log-probability columns
per input model, in bundle ``models`` order. Its classes are integer indexes
into ``teams``. For the chosen team ``a`` and the comparison team ``b`` the
logit difference splits exactly into one term per model block plus the
intercept difference.

Depends only on numpy and ``src.ensemble``; no platform package is imported.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.ensemble.stack import stack_features
from src.explain.explainers import NOMINAL, skipped_section
from src.explain.record import Driver, Section, make_section

__all__ = ["explain_stacker", "EXACT_TOL", "REASON_NO_STACKER"]

NAME = "stacker"
EXACT_TOL = 1e-6
REASON_NO_STACKER = "bundle does not use a stacker"


def _row(a: Any, what: str, n: int) -> np.ndarray:
    """One probability row of length ``n``: accepts shape (n,) or (1, n)."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (n,):
        raise ValueError(f"{what} must have shape ({n},) or (1, {n}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} holds a non-finite value")
    return arr


def _top2(p: np.ndarray) -> tuple[int, int]:
    """Highest and second-highest index; ties broken by lower index."""
    order = np.argsort(-p, kind="stable")
    return int(order[0]), int(order[1])


def _prob_dict(teams: Sequence[str], p: np.ndarray) -> dict[str, float]:
    # Clip float noise such as 1.0000000000000002 into the Prob range.
    return {t: float(min(1.0, max(0.0, p[j]))) for j, t in enumerate(teams)}


def _full_linear(lr) -> tuple[np.ndarray, np.ndarray]:
    """Per-class weight rows and intercepts whose softmax is predict_proba.

    A binary LogisticRegression stores one row for class 1; class 0 is the
    zero row, which gives the same probabilities under softmax.
    """
    W = np.asarray(lr.coef_, np.float64)
    beta = np.asarray(lr.intercept_, np.float64)
    if len(lr.classes_) == 2 and W.shape[0] == 1:
        W = np.vstack([np.zeros_like(W), W])
        beta = np.concatenate([[0.0], beta])
    return W, beta


def explain_stacker(stacker: Any, models: Sequence[str], teams: Sequence[str],
                    model_probs: Mapping[str, Any] | None, chosen_team: str,
                    p_raw: Any, p_adj: Any | None, prior_active: bool,
                    uses_stacker: bool) -> Section:
    """The ``stacker`` section for one request (Req 7.1-7.8).

    ``model_probs`` maps each input model name to its probability vector over
    ``teams``. ``p_raw`` is the stacker output before label-shift correction
    and ``p_adj`` after it (used only when ``prior_active``).
    """
    if not uses_stacker or stacker is None:
        return skipped_section(NAME, REASON_NO_STACKER)
    teams = list(teams)
    models = list(models)
    T = len(teams)
    if T < 2:
        raise ValueError(f"need at least 2 teams, got {T}")
    if not models:
        raise ValueError("the stacker bundle lists no input models")

    model_probs = model_probs or {}
    missing = [m for m in models if model_probs.get(m) is None]
    if missing:
        return skipped_section(
            NAME, f"probability vector missing for input model(s): {', '.join(missing)}",
            chosen_team=chosen_team)

    lr = stacker[-1]
    classes = [int(c) for c in lr.classes_]
    fitted = {teams[c] for c in classes if 0 <= c < T}
    if chosen_team not in fitted:
        return skipped_section(
            NAME, f"chosen team {chosen_team} is not among the stacker's fitted classes",
            chosen_team=chosen_team)

    p_raw = _row(p_raw, "p_raw", T)
    a = teams.index(chosen_team)
    top, second = _top2(p_raw)
    if top == a:
        b, case = second, "stacker_agrees"
    else:
        b, case = top, "stacker_disagrees"
    comparison = teams[b]
    if comparison not in fitted:
        return skipped_section(
            NAME, f"comparison team {comparison} is not among the stacker's fitted classes",
            chosen_team=chosen_team, comparison_team=comparison, case=case)

    # Standardised stack features for this request.
    mats = [_row(model_probs[m], f"probability vector of {m}", T)[None, :] for m in models]
    x = stack_features(mats)
    s = np.asarray(stacker[:-1].transform(x), np.float64)[0]
    W, beta = _full_linear(lr)
    if W.shape[1] != len(s) or len(s) != T * len(models):
        raise ValueError(f"stacker expects {W.shape[1]} features, built {len(s)} "
                         f"from {len(models)} models x {T} teams")
    ra, rb = classes.index(a), classes.index(b)
    dw = W[ra] - W[rb]
    terms = dw * s
    contribs = {m: float(terms[i * T:(i + 1) * T].sum()) for i, m in enumerate(models)}
    intercept_diff = float(beta[ra] - beta[rb])

    # The logit difference from the estimator itself, for the exactness check.
    dec = np.asarray(lr.decision_function(s[None, :]), np.float64)
    if dec.ndim == 1:                      # binary: margin of class 1 over class 0
        logits = np.array([0.0, dec[0]])
    else:
        logits = dec[0]
    logit_diff = float(logits[ra] - logits[rb])
    residual = abs(sum(contribs.values()) + intercept_diff - logit_diff)
    method = "exact" if residual <= EXACT_TOL else "approximate"

    # Most positive first; ties keep the bundle's model order.
    ranking = sorted(models, key=lambda m: (-contribs[m], models.index(m)))
    carrying = ranking[0] if contribs[ranking[0]] > 0 else None
    drivers = [Driver(kind="model", label=m, contribution=contribs[m]) for m in ranking]

    details: dict[str, Any] = {
        "case": case, "chosen_team": chosen_team, "comparison_team": comparison,
        "intercept_diff": intercept_diff, "logit_diff": logit_diff,
        "residual": float(residual), "ranking": ranking, "carrying_model": carrying,
        "label_shift_active": bool(prior_active),
    }
    if prior_active:
        if p_adj is None:
            raise ValueError("label shift is active but p_adj is missing")
        p_adj = _row(p_adj, "p_adj", T)
        top_after = teams[_top2(p_adj)[0]]
        top_before = teams[top]
        details.update(
            p_before=_prob_dict(teams, p_raw), p_after=_prob_dict(teams, p_adj),
            top_before=top_before, top_after=top_after,
            shift_changed_top=bool(top_before != top_after and top_after == chosen_team))

    return make_section(NAME, status="ok", method=method,
                        technique=NOMINAL[NAME][1], drivers=drivers, details=details)
