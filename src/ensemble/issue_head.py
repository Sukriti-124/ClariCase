"""Issue classifier on sentence embeddings: one logistic regression per team,
trained on that team's training complaints (same hierarchy as
improved_model_1's SVM issue heads)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.ensemble.probs import ISSUE_TEAM, ISSUES


def fit_issue_heads(X: np.ndarray, teams, issues, C: float = 1.0,
                    max_iter: int = 1000) -> dict:
    teams = np.asarray(teams).astype(str)
    issues = np.asarray(issues).astype(str)
    heads: dict = {}
    for team in sorted(set(teams)):
        m = teams == team
        uniq = np.unique(issues[m])
        if len(uniq) == 1:
            heads[team] = str(uniq[0])  # only one issue seen: constant
            continue
        heads[team] = LogisticRegression(
            C=C, class_weight="balanced", max_iter=max_iter,
        ).fit(X[m], issues[m])
    return heads


def conditional_issue_probs(heads: dict, X: np.ndarray,
                            issues: Sequence[str] = ISSUES,
                            issue_team: np.ndarray = ISSUE_TEAM) -> np.ndarray:
    """P(issue | its team) for every row and every team's block."""
    col = {iss: j for j, iss in enumerate(issues)}
    Q = np.zeros((len(X), len(issues)), dtype=np.float64)
    for team, head in heads.items():
        if isinstance(head, str):
            if head in col:
                Q[:, col[head]] = 1.0
            continue
        P = head.predict_proba(X)
        for j, c in enumerate(head.classes_):
            if str(c) in col:
                Q[:, col[str(c)]] = P[:, j]
    return Q
