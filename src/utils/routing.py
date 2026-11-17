"""Routing / north-star harness.

Ported verbatim from `src/baseline_model_2/train.py` (see that file for the
original implementation; this copy exists so improved models can import the
harness without depending on the served baseline module).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import (
    MIN_ROUTED, N_BOOTSTRAP, NORTH_STAR_KEY, NORTH_STAR_TARGET,
    PRECISION_TARGETS, RANDOM_SEED,
    TEAM_PRECISION_FLOOR,
)


def predict_with_confidence(model, x) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(x)
    return np.asarray(model.classes_)[proba.argmax(axis=1)], proba.max(axis=1)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.sum():
            error += mask.mean() * abs(correct[mask].mean()
                                       - confidence[mask].mean())
    return float(error)


def coverage_curve(confidence: np.ndarray, correct: np.ndarray,
                   min_routed: int = MIN_ROUTED) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0, 1, 501):
        routed = confidence >= threshold
        if routed.sum() < min_routed:
            continue
        rows.append({
            "threshold": round(float(threshold), 3),
            "coverage": float(routed.mean()),
            "precision": float(correct[routed].mean()),
            "n_routed": int(routed.sum()),
        })
    return pd.DataFrame(rows)


def threshold_for(curve: pd.DataFrame, target: float) -> float | None:
    feasible = curve[curve["precision"] >= target]
    return None if feasible.empty else float(feasible.iloc[0]["threshold"])


def routed_stats(confidence: np.ndarray, correct: np.ndarray,
                 threshold: float) -> dict:
    routed = confidence >= threshold
    return {
        "threshold": threshold,
        "coverage": round(float(routed.mean()), 4),
        "precision": (round(float(correct[routed].mean()), 4)
                      if routed.any() else None),
        "n_routed": int(routed.sum()),
    }


def bootstrap_ci(confidence: np.ndarray, correct: np.ndarray,
                 threshold: float, n_boot: int = N_BOOTSTRAP,
                 seed: int = RANDOM_SEED) -> dict:
    rng = np.random.default_rng(seed)
    routed = confidence >= threshold
    n = len(routed)
    cov, prec = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        r, c = routed[idx], correct[idx]
        cov[b] = r.mean()
        prec[b] = c[r].mean() if r.any() else np.nan
    return {
        "coverage_95ci": [round(float(v), 4)
                          for v in np.percentile(cov, [2.5, 97.5])],
        "precision_95ci": [round(float(v), 4)
                           for v in np.nanpercentile(prec, [2.5, 97.5])],
    }


def per_team_routing(y_true: np.ndarray, y_pred: np.ndarray,
                     confidence: np.ndarray, threshold: float,
                     names: dict[str, str]) -> pd.DataFrame:
    routed = confidence >= threshold
    rows = []
    for team in sorted(set(y_true) | set(y_pred)):
        to_team = routed & (y_pred == team)
        own = y_true == team
        n_to = int(to_team.sum())
        precision = (float((y_true[to_team] == team).mean())
                     if n_to else np.nan)
        rows.append({
            "team_id": team,
            "team_name": names.get(team, ""),
            "n_test": int(own.sum()),
            "n_routed_to": n_to,
            "precision_routed": round(precision, 4),
            "auto_routed_share": round(float(routed[own].mean()), 4)
            if own.any() else np.nan,
            "meets_floor": bool(precision >= TEAM_PRECISION_FLOOR)
            if n_to else None,
        })
    return pd.DataFrame(rows)


def north_star_summary(val_conf, val_correct, test_conf, test_correct,
                       test_true, test_pred, names
                       ) -> tuple[dict, pd.DataFrame, pd.DataFrame | None]:
    val_curve = coverage_curve(val_conf, val_correct)
    test_curve = coverage_curve(test_conf, test_correct)
    curves = pd.concat([val_curve.assign(split="validation"),
                        test_curve.assign(split="test")], ignore_index=True)

    operating_points = {}
    for target in PRECISION_TARGETS:
        key = f"at_{int(target * 100)}p"
        threshold = threshold_for(val_curve, target)
        if threshold is None:
            operating_points[key] = None
            continue
        point = routed_stats(test_conf, test_correct, threshold)
        point["validation_coverage"] = routed_stats(
            val_conf, val_correct, threshold)["coverage"]
        point["target_met_on_test"] = (point["precision"] is not None
                                       and point["precision"] >= target)
        operating_points[key] = point

    main = operating_points[NORTH_STAR_KEY]
    teams = None
    if main is not None:
        main.update(bootstrap_ci(test_conf, test_correct, main["threshold"]))
        teams = per_team_routing(test_true, test_pred, test_conf,
                                 main["threshold"], names)

    summary = {
        "metric": (f"autonomous routing rate at >={NORTH_STAR_TARGET:.0%} "
                   "precision"),
        "protocol": ("temporal split, real team mix in validation and test; "
                     "isotonic calibration and thresholds fit on validation, "
                     "frozen for test"),
        "value": None if main is None else main["coverage"],
        "operating_points": operating_points,
        "ece_test": round(expected_calibration_error(test_conf, test_correct), 4),
        "guardrail": None if teams is None else {
            "team_precision_floor": TEAM_PRECISION_FLOOR,
            "teams_below_floor": teams.loc[teams["meets_floor"] == False,
                                           "team_id"].tolist(),
        },
        "n_validation": int(len(val_correct)),
        "n_test": int(len(test_correct)),
    }
    return summary, curves, teams


# ---- NEW: per-team thresholds (spec §1.6) --------------------------------
def per_team_thresholds(val_conf: np.ndarray, val_pred: np.ndarray,
                        val_true: np.ndarray, target: float = 0.95,
                        min_routed: int = MIN_ROUTED,
                        ) -> dict[str, float | None]:
    """For each predicted team, lowest threshold where routed-to-team precision
    on validation reaches `target`. None if unreachable."""
    val_conf = np.asarray(val_conf)
    val_pred = np.asarray(val_pred)
    val_true = np.asarray(val_true)
    thresholds: dict[str, float | None] = {}
    teams = sorted(set(val_pred) | set(val_true))
    grid = np.linspace(0, 1, 501)
    for team in teams:
        pred_mask = val_pred == team
        best: float | None = None
        for t in grid:
            routed = pred_mask & (val_conf >= t)
            if routed.sum() < min_routed:
                continue
            precision = (val_true[routed] == team).mean()
            if precision >= target:
                best = round(float(t), 3)
                break
        thresholds[team] = best
    return thresholds


def apply_per_team_thresholds(y_true: np.ndarray, y_pred: np.ndarray,
                              conf: np.ndarray,
                              thresholds: dict[str, float | None],
                              names: dict[str, str],
                              min_routed: int = MIN_ROUTED) -> dict:
    """Score per-team thresholds on an eval split. Returns {coverage,
    precision, per_team_df, bootstrap_ci}."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    conf = np.asarray(conf)
    routed = np.zeros(len(y_pred), dtype=bool)
    for team, t in thresholds.items():
        if t is None:
            continue
        routed |= (y_pred == team) & (conf >= t)
    correct = (y_pred == y_true) & routed
    coverage = float(routed.mean())
    precision = float((correct[routed]).mean()) if routed.any() else None

    rows = []
    for team in sorted(set(y_true) | set(y_pred)):
        pred_m = y_pred == team
        routed_m = pred_m & routed
        own = y_true == team
        n_to = int(routed_m.sum())
        prec = (float((y_true[routed_m] == team).mean())
                if n_to else np.nan)
        rows.append({
            "team_id": team,
            "team_name": names.get(team, ""),
            "threshold": thresholds.get(team),
            "n_routed_to": n_to,
            "precision_routed": round(prec, 4),
            "auto_routed_share": round(float(routed[own].mean()), 4)
                                 if own.any() else np.nan,
        })
    per_team_df = pd.DataFrame(rows)

    # Bootstrap CI for overall precision at the per-team thresholds.
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y_true)
    cov = np.empty(N_BOOTSTRAP)
    prec = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        r = routed[idx]
        cov[b] = r.mean()
        prec[b] = ((y_pred[idx] == y_true[idx])[r]).mean() if r.any() else np.nan
    return {
        "coverage": round(coverage, 4),
        "precision": round(precision, 4) if precision is not None else None,
        "coverage_95ci": [round(float(v), 4)
                          for v in np.percentile(cov, [2.5, 97.5])],
        "precision_95ci": [round(float(v), 4)
                           for v in np.nanpercentile(prec, [2.5, 97.5])],
        "per_team": per_team_df,
    }
