"""Routing gate: estimates P(the primary model's team is correct).

The gate never changes the predicted team. It takes what each model service
already returns under the platform contract (top team + confidence) and turns
that into one score that is thresholded for auto-routing. Because it only
needs (team, confidence) per model, the same code runs offline on prediction
files and online inside ensemble-api.

This module deliberately imports only numpy and scikit-learn so the service
image stays small and import has no side effects.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GATE_KINDS = ("logreg", "hgb")
MODEL_NAME = "ensemble_gate_v1"


def feature_names(team_classes: Sequence[str],
                  secondary_names: Sequence[str]) -> list[str]:
    names = ["primary_conf"]
    for s in secondary_names:
        names += [f"{s}_conf", f"{s}_agree", f"{s}_agree_x_conf"]
    names += ["agree_share", "log_n_words"]
    names += [f"primary_is_{t}" for t in team_classes]
    return names


def build_features(primary_team, primary_conf,
                   secondary_teams: Sequence, secondary_confs: Sequence,
                   n_words, team_classes: Sequence[str]) -> np.ndarray:
    """Feature matrix for the gate.

    primary_team / primary_conf: arrays of length n.
    secondary_teams / secondary_confs: one array of length n per secondary
    model, in a fixed order (the order is part of the gate bundle).
    """
    primary_team = np.asarray(primary_team).astype(str)
    primary_conf = np.asarray(primary_conf, dtype=np.float64)
    n = len(primary_team)
    if len(secondary_teams) != len(secondary_confs):
        raise ValueError("secondary_teams and secondary_confs differ in length")

    cols = [primary_conf]
    agrees = []
    for teams, confs in zip(secondary_teams, secondary_confs):
        teams = np.asarray(teams).astype(str)
        confs = np.asarray(confs, dtype=np.float64)
        if len(teams) != n or len(confs) != n:
            raise ValueError("secondary arrays must match the primary length")
        agree = (teams == primary_team).astype(np.float64)
        agrees.append(agree)
        cols += [confs, agree, agree * confs]
    agree_share = (np.mean(agrees, axis=0) if agrees
                   else np.zeros(n, dtype=np.float64))
    cols.append(agree_share)
    cols.append(np.log1p(np.asarray(n_words, dtype=np.float64)))
    for t in team_classes:
        cols.append((primary_team == str(t)).astype(np.float64))
    return np.column_stack(cols)


def make_estimator(kind: str = "logreg", seed: int = 42):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000),
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
            min_samples_leaf=200, l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(f"unknown gate kind {kind!r}; want one of {GATE_KINDS}")


def cross_fit_scores(X: np.ndarray, y_correct: np.ndarray,
                     groups: np.ndarray | None = None, kind: str = "logreg",
                     n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Out-of-fold gate scores, so a threshold can be chosen on the same rows
    the gate is fitted on without the scores being optimistic.

    With `groups` (leakage_group_id), near-duplicate templates never sit on
    both sides of a fold.
    """
    y = np.asarray(y_correct).astype(int)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    if groups is not None:
        splits = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                 random_state=seed).split(X, y)
    for fit_idx, hold_idx in splits:
        est = make_estimator(kind, seed).fit(X[fit_idx], y[fit_idx])
        oof[hold_idx] = est.predict_proba(X[hold_idx])[:, 1]
    return oof


def fit_gate(X: np.ndarray, y_correct: np.ndarray, kind: str = "logreg",
             seed: int = 42):
    return make_estimator(kind, seed).fit(X, np.asarray(y_correct).astype(int))


def make_bundle(estimator, kind: str, team_classes: Sequence[str],
                primary_name: str, secondary_names: Sequence[str],
                threshold: float | None,
                reference_values: Sequence[float] | None = None) -> dict:
    """Gate bundle. `reference_values` (per-feature medians of the training
    matrix, aligned with `feature_names`) is stored under
    `gate_reference_values` when given; otherwise the key is absent."""
    names = feature_names(team_classes, secondary_names)
    bundle = {
        "model_name": MODEL_NAME,
        "kind": kind,
        "estimator": estimator,
        "team_classes": [str(t) for t in team_classes],
        "primary_name": primary_name,
        "secondary_names": list(secondary_names),
        "threshold": threshold,
        "feature_names": names,
    }
    if reference_values is not None:
        ref = [float(v) for v in np.asarray(reference_values,
                                            dtype=np.float64).ravel()]
        if len(ref) != len(names):
            raise ValueError(f"reference_values has {len(ref)} entries, "
                             f"want {len(names)} (one per feature)")
        bundle["gate_reference_values"] = ref
    return bundle


def score(bundle: dict, X: np.ndarray) -> np.ndarray:
    return bundle["estimator"].predict_proba(X)[:, 1]


def score_one(bundle: dict, primary_team: str, primary_conf: float,
              secondaries: Sequence[tuple[str, float]], text: str) -> float:
    """Gate score for one live request. `secondaries` must follow the order of
    bundle["secondary_names"]."""
    if len(secondaries) != len(bundle["secondary_names"]):
        raise ValueError(
            f"gate expects {len(bundle['secondary_names'])} secondary models, "
            f"got {len(secondaries)}")
    X = build_features(
        [primary_team], [primary_conf],
        [[t] for t, _ in secondaries], [[c] for _, c in secondaries],
        [count_words(text)], bundle["team_classes"],
    )
    return float(score(bundle, X)[0])


def decide(bundle: dict, gate_score: float) -> bool:
    thr = bundle.get("threshold")
    return thr is not None and gate_score >= thr


def count_words(text: str) -> int:
    return len((text or "").split())


# ---- v2: features from full probability vectors ---------------------------

PROB_FEATURES = ("p_chosen", "top", "margin", "agree", "entropy")


def prob_feature_names(model_names: Sequence[str],
                       team_classes: Sequence[str]) -> list[str]:
    names = [f"{m}_{f}" for m in model_names for f in PROB_FEATURES]
    names += [f"chosen_is_{t}" for t in team_classes]
    names.append("log_n_words")
    return names


def prob_features(chosen_idx, prob_mats: Sequence[np.ndarray], n_words,
                  n_teams: int) -> np.ndarray:
    """Gate features when every model gives a full team probability vector.

    For each model: the probability it gives the chosen team, its own top
    probability, the gap between its first and second choice, whether its top
    team is the chosen one, and its normalised entropy. Plus the chosen team
    (one-hot) and complaint length.
    """
    chosen_idx = np.asarray(chosen_idx, dtype=np.int64)
    n = len(chosen_idx)
    rows = np.arange(n)
    cols = []
    for P in prob_mats:
        P = np.clip(np.asarray(P, dtype=np.float64), 0.0, 1.0)
        if P.shape != (n, n_teams):
            raise ValueError(f"probability matrix shape {P.shape}, "
                             f"want {(n, n_teams)}")
        srt = np.sort(P, axis=1)
        ent = -(P * np.log(P + 1e-12)).sum(axis=1) / np.log(n_teams)
        cols += [P[rows, chosen_idx], srt[:, -1], srt[:, -1] - srt[:, -2],
                 (P.argmax(axis=1) == chosen_idx).astype(np.float64), ent]
    onehot = np.zeros((n, n_teams))
    onehot[rows, chosen_idx] = 1.0
    return np.column_stack(cols + [onehot,
                                   np.log1p(np.asarray(n_words, np.float64))])
