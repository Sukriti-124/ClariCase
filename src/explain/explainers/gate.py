"""Gate_Explainer: which evidence raised or lowered the routing gate's score
(design §7, Req 8).

The gate (``src.ensemble.gate``) is a ``Pipeline[StandardScaler,
LogisticRegression]`` (kind ``logreg``) or a ``HistGradientBoostingClassifier``
(kind ``hgb``) over ``prob_features``. Every gate feature belongs to exactly
one feature group, and the section reports one Driver per group:

* ``logreg`` -> technique ``linear``: ``coef_f * scaled_f`` per feature,
  constant ``intercept``, unit log-odds.
* ``hgb`` with a compatible ``shap`` -> technique ``shap``: TreeExplainer
  values on the raw margin, constant ``expected_value``, unit log-odds.
* ``hgb`` otherwise -> technique ``ablation``: per group, the gate score
  minus the score with that group's columns set to the bundle's reference
  values; unit "change in gate score".

``shap`` is optional. It is imported lazily here and is not listed in any
requirements file (Req 8.9).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.ensemble.gate import PROB_FEATURES
from src.explain.explainers import error_reason
from src.explain.record import Absent, Driver, Section, make_section

__all__ = ["explain_gate", "feature_groups", "EXACT_TOL", "REASON_NO_REFERENCE",
           "REASON_NO_THRESHOLD", "UNIT_LOG_ODDS", "UNIT_ABLATION",
           "GROUP_CHOSEN_TEAM", "GROUP_LENGTH"]

NAME = "gate"
EXACT_TOL = 1e-6
REASON_NO_REFERENCE = "gate bundle has no reference values (gate_reference_values)"
REASON_NO_THRESHOLD = "no gate threshold applies to the chosen team"
UNIT_LOG_ODDS = "log-odds"
UNIT_ABLATION = "change in gate score"
GROUP_CHOSEN_TEAM = "chosen_team"
GROUP_LENGTH = "complaint_length"
CHOSEN_PREFIX = "chosen_is_"
LENGTH_FEATURE = "log_n_words"

# (technique, method, unit) per technique.
_NOMINAL = {
    "linear": ("exact", UNIT_LOG_ODDS),
    "shap": ("exact", UNIT_LOG_ODDS),
    "ablation": ("approximate", UNIT_ABLATION),
}


def feature_groups(feature_names: Sequence[str],
                   group_models: Sequence[str]) -> dict[str, list[int]]:
    """Map group name -> column indexes (Req 8.4).

    Groups: ``model:<m>`` for each ``m`` in ``group_models`` (holding
    ``<m>_p_chosen``, ``<m>_top``, ``<m>_margin``, ``<m>_agree`` and
    ``<m>_entropy``), ``chosen_team`` (every ``chosen_is_*`` feature) and
    ``complaint_length`` (``log_n_words``). A feature in no group, or in
    more than one, raises ``ValueError``. Groups without members are omitted.
    """
    model_members = {f"model:{m}": {f"{m}_{f}" for f in PROB_FEATURES}
                     for m in group_models}
    groups: dict[str, list[int]] = {}
    for j, name in enumerate(feature_names):
        hits = [g for g, members in model_members.items() if name in members]
        if name.startswith(CHOSEN_PREFIX):
            hits.append(GROUP_CHOSEN_TEAM)
        if name == LENGTH_FEATURE:
            hits.append(GROUP_LENGTH)
        if not hits:
            raise ValueError(f"gate feature {name!r} belongs to no feature group")
        if len(hits) > 1:
            raise ValueError(f"gate feature {name!r} belongs to several feature "
                             f"groups: {', '.join(hits)}")
        groups.setdefault(hits[0], []).append(j)
    return groups


def _row(X: Any, n: int) -> np.ndarray:
    """The gate feature row as shape (1, n)."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape != (1, n):
        raise ValueError(f"gate feature row must have {n} columns, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("gate feature row holds a non-finite value")
    return arr


def _drivers(group_contribs: dict[str, float]) -> list[Driver]:
    """One Driver per group, by |contribution| descending, ties by name."""
    order = sorted(group_contribs, key=lambda g: (-abs(group_contribs[g]), g))
    return [Driver(kind="feature_group", label=g, contribution=group_contribs[g])
            for g in order]


def _sum_groups(groups: dict[str, list[int]], per_feature: np.ndarray) -> dict[str, float]:
    return {g: float(per_feature[idx].sum()) for g, idx in groups.items()}


def _linear(est: Any, x: np.ndarray, groups: dict[str, list[int]]):
    lr = est[-1]
    s = np.asarray(est[:-1].transform(x), np.float64)[0]
    coef = np.asarray(lr.coef_, np.float64)
    if coef.shape != (1, len(s)):
        raise ValueError(f"logreg gate coef_ has shape {coef.shape}, want (1, {len(s)})")
    per_feature = coef[0] * s
    constant = float(np.asarray(lr.intercept_, np.float64)[0])
    margin = float(np.asarray(est.decision_function(x), np.float64).ravel()[0])
    residual = abs(float(per_feature.sum()) + constant - margin)
    return _sum_groups(groups, per_feature), constant, residual


def _load_shap(shap_module: Any) -> Any:
    if shap_module is not None:
        return shap_module
    try:
        import shap  # noqa: PLC0415  (optional dependency, Req 8.9)
    except Exception:
        return None
    return shap


def _shap(shap_mod: Any, est: Any, x: np.ndarray, groups: dict[str, list[int]]):
    """SHAP values on the raw margin, or ``None`` when the compatibility check
    fails or ``shap`` raises (Req 8.1, 8.2)."""
    try:
        explainer = shap_mod.TreeExplainer(est, model_output="raw")
        vals = explainer.shap_values(x)
        if isinstance(vals, list):           # older shap: one array per class
            vals = vals[-1]
        vals = np.asarray(vals, np.float64)
        if vals.ndim == 3:                   # (n, F, classes)
            vals = vals[..., -1]
        per_feature = vals.reshape(-1, x.shape[1])[0]
        ev = np.asarray(explainer.expected_value, np.float64).ravel()
        constant = float(ev[-1])
        margin = float(np.asarray(est.decision_function(x), np.float64).ravel()[0])
        residual = abs(float(per_feature.sum()) + constant - margin)
        if not (np.all(np.isfinite(per_feature)) and np.isfinite(constant)
                and residual <= EXACT_TOL):
            return None
    except Exception:
        return None
    return _sum_groups(groups, per_feature), constant, residual


def _ablation(est: Any, x: np.ndarray, reference: np.ndarray,
              groups: dict[str, list[int]]) -> dict[str, float]:
    base = float(est.predict_proba(x)[0, 1])
    out = {}
    for g, idx in groups.items():
        x_ref = x.copy()
        x_ref[0, idx] = reference[idx]
        out[g] = base - float(est.predict_proba(x_ref)[0, 1])
    return out


def _gate_fields(gate_kind: Any, score: Any, threshold: Any) -> dict[str, Any]:
    """The fields recorded whatever the status (Req 8.5, 8.8)."""
    s = float(score) if score is not None else None
    if s is not None and not 0.0 <= s <= 1.0:
        raise ValueError(f"gate score must be in [0, 1], got {s!r}")
    if threshold is None:
        thr: Any = Absent(reason=REASON_NO_THRESHOLD)
        ge = False
    else:
        thr = float(threshold)
        ge = s is not None and s >= thr
    return {"gate_kind": None if gate_kind is None else str(gate_kind),
            "score": s, "threshold": thr, "score_ge_threshold": bool(ge)}


def _section(status: str, technique: str, fields: dict, *, reason: str | None = None,
             drivers: list[Driver] | None = None, method: str | None = None,
             **extra: Any) -> Section:
    nominal_method, unit = _NOMINAL[technique]
    details = {**fields, "unit": unit, **extra}
    return make_section(NAME, status=status, reason=reason,
                        method=method or nominal_method, technique=technique,
                        drivers=drivers or [], details=details)


def explain_gate(gate_est: Any, gate_kind: str, feature_names: Sequence[str],
                 X: Any, reference: Sequence[float] | None,
                 group_models: Sequence[str], score: float,
                 threshold: float | None, shap_module: Any = None) -> Section:
    """The ``gate`` section for one request (Req 8.1-8.6, 8.8).

    ``X`` is the single gate feature row (shape ``(F,)`` or ``(1, F)``) in
    ``feature_names`` order. ``reference`` is the bundle's
    ``gate_reference_values`` or ``None``. ``group_models`` is the gate's
    model order (``["stack"] + models`` for stack scenarios). ``score`` and
    ``threshold`` are the served gate score and the applied threshold.
    ``shap_module`` replaces the lazy ``import shap`` (tests inject a fake).
    """
    try:
        fields = _gate_fields(gate_kind, score, threshold)
    except Exception as exc:                 # unusable score/threshold
        return make_section(NAME, status="error", reason=error_reason(exc),
                            method="approximate", technique="ablation",
                            details={"gate_kind": str(gate_kind)})
    technique = "linear" if gate_kind == "logreg" else "ablation"
    try:
        if gate_kind not in ("logreg", "hgb"):
            raise ValueError(f"unknown gate kind {gate_kind!r}")
        names = [str(n) for n in feature_names]
        groups = feature_groups(names, group_models)
        x = _row(X, len(names))

        if gate_kind == "logreg":
            contribs, constant, residual = _linear(gate_est, x, groups)
            method = "exact" if residual <= EXACT_TOL else "approximate"
            return _section("ok", "linear", fields, drivers=_drivers(contribs),
                            method=method, constant=constant, residual=residual)

        shap_mod = _load_shap(shap_module)
        if shap_mod is not None:
            got = _shap(shap_mod, gate_est, x, groups)
            if got is not None:
                contribs, constant, residual = got
                return _section("ok", "shap", fields, drivers=_drivers(contribs),
                                constant=constant, residual=residual)

        if reference is None:
            return _section("skipped", "ablation", fields, reason=REASON_NO_REFERENCE)
        ref = np.asarray(reference, dtype=np.float64).ravel()
        if ref.shape != (len(names),):
            raise ValueError(f"gate_reference_values has {ref.size} entries, "
                             f"want {len(names)} (one per feature)")
        contribs = _ablation(gate_est, x, ref, groups)
        return _section("ok", "ablation", fields, drivers=_drivers(contribs))
    except Exception as exc:
        return _section("error", technique, fields, reason=error_reason(exc))
