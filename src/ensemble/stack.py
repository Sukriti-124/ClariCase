"""Team stacker: picks the team from every model's probability vector.

A multinomial logistic regression over each model's log-probabilities,
fitted on validation (which has the real team mix). Out-of-fold predictions
on validation stand in for test-time behaviour when the routing gate and
thresholds are fitted on top.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


def stack_features(prob_mats: Sequence[np.ndarray]) -> np.ndarray:
    return np.hstack([np.log(np.clip(np.asarray(P, np.float64), EPS, 1.0))
                      for P in prob_mats])


def make_stacker(C: float = 1.0):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000))


def fold_splits(y, groups=None, n_splits: int = 5, seed: int = 42):
    y = np.asarray(y)
    if groups is not None and len(np.unique(groups)) >= n_splits:
        return list(GroupKFold(n_splits=n_splits).split(y, y, groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed).split(y, y))


def stacker_proba(est, X: np.ndarray, n_classes: int) -> np.ndarray:
    P = np.zeros((len(X), n_classes))
    P[:, est[-1].classes_] = est.predict_proba(X)
    return P


def cross_fit_proba(X: np.ndarray, y_idx: np.ndarray, n_classes: int,
                    splits, C: float = 1.0) -> np.ndarray:
    oof = np.zeros((len(y_idx), n_classes))
    for fit_idx, hold_idx in splits:
        est = make_stacker(C).fit(X[fit_idx], y_idx[fit_idx])
        oof[hold_idx] = stacker_proba(est, X[hold_idx], n_classes)
    return oof


def tune_stacker(X: np.ndarray, y_idx: np.ndarray, n_classes: int, splits,
                 grid: Sequence[float] = (0.1, 1.0)
                 ) -> tuple[float, list[dict], np.ndarray]:
    """Pick C by out-of-fold log loss. Returns (C, table, OOF probabilities
    for that C)."""
    rows, best = [], None
    for C in grid:
        oof = cross_fit_proba(X, y_idx, n_classes, splits, C)
        clipped = np.clip(oof, EPS, 1.0)
        clipped /= clipped.sum(axis=1, keepdims=True)
        ll = float(log_loss(y_idx, clipped, labels=np.arange(n_classes)))
        acc = float((oof.argmax(1) == y_idx).mean())
        rows.append({"C": C, "oof_log_loss": ll, "oof_accuracy": acc})
        if best is None or ll < best[1]:
            best = (C, ll, oof)
    return best[0], rows, best[2]
