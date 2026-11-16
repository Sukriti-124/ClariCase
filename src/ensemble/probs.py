"""Probability vectors inside prediction files.

Team probabilities sit in the per-model prediction file
(data/cache/predictions/<model>/<split>.parquet) as one column per team,
`p_<team_id>`, in TEAMS order.

Issue probabilities sit in a sibling file `<split>_issues.parquet` as one
column per canonical issue, `q_<issue_id>`, in ISSUES order. They are stored
*conditional on the issue's own team*: inside each team's block of columns
they sum to 1. That is what a hierarchical issue head produces, and it lets
any team choice be combined with any issue model afterwards.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.ensemble.data import predictions_path
from src.utils.config import ID_COL, ISSUE_COL, TEAM_COL
from src.utils.labels import issue_to_team, team_names

TEAMS: list[str] = sorted(team_names)
ISSUES: list[str] = sorted(issue_to_team)
ISSUE_TEAM: np.ndarray = np.array([issue_to_team[i] for i in ISSUES])


def team_cols(teams: Sequence[str] = TEAMS) -> list[str]:
    return [f"p_{t}" for t in teams]


def issue_cols(issues: Sequence[str] = ISSUES) -> list[str]:
    return [f"q_{i}" for i in issues]


def reorder(P: np.ndarray, classes: Sequence, order: Sequence[str]) -> np.ndarray:
    """Columns of P (labelled by `classes`) rearranged into `order`. Classes
    the model never saw get probability 0."""
    P = np.asarray(P, dtype=np.float64)
    out = np.zeros((P.shape[0], len(order)), dtype=np.float64)
    pos = {str(c): j for j, c in enumerate(order)}
    for j, c in enumerate(classes):
        if str(c) in pos:
            out[:, pos[str(c)]] = P[:, j]
    return out


def team_matrix(df: pd.DataFrame, teams: Sequence[str] = TEAMS) -> np.ndarray:
    cols = team_cols(teams)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"prediction file has no team probabilities ({missing[:3]}...). "
            "Re-export it with `python -m src.ensemble.export_predictions`.")
    return df[cols].to_numpy(dtype=np.float64)


def issue_matrix(df: pd.DataFrame, issues: Sequence[str] = ISSUES) -> np.ndarray:
    cols = issue_cols(issues)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"issue file is missing columns {missing[:3]}...")
    return df[cols].to_numpy(dtype=np.float64)


def block_normalize(Q: np.ndarray, issue_team: np.ndarray = ISSUE_TEAM,
                    eps: float = 1e-3) -> np.ndarray:
    """Make each team's block of issue scores a proper distribution. `eps`
    smooths zeros (an issue a model never predicts must not get log(0))."""
    Q = np.asarray(Q, dtype=np.float64)
    out = np.empty_like(Q)
    for t in np.unique(issue_team):
        cols = issue_team == t
        B = np.clip(Q[:, cols], 0.0, None) + eps
        out[:, cols] = B / B.sum(axis=1, keepdims=True)
    return out


def with_team_probs(frame: pd.DataFrame, P: np.ndarray,
                    teams: Sequence[str] = TEAMS) -> pd.DataFrame:
    out = frame.copy()
    for j, c in enumerate(team_cols(teams)):
        out[c] = P[:, j].astype(np.float32)
    return out


def issues_path(model_name: str, split: str):
    return predictions_path(model_name, f"{split}_issues")


def write_issue_probs(model_name: str, split: str, df: pd.DataFrame,
                      Q: np.ndarray, issues: Sequence[str] = ISSUES):
    frame = pd.DataFrame({
        ID_COL: df[ID_COL].astype(str).to_numpy(),
        "actual_team_id": df[TEAM_COL].astype(str).to_numpy(),
        "actual_issue_id": df[ISSUE_COL].astype(str).to_numpy(),
    })
    Q = np.asarray(Q, dtype=np.float32)
    frame = pd.concat([frame, pd.DataFrame(Q, columns=issue_cols(issues))],
                      axis=1)
    path = issues_path(model_name, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def read_issue_probs(model_name: str, split: str) -> pd.DataFrame:
    path = issues_path(model_name, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Export it with `python -m "
            f"src.ensemble.export_predictions --models {model_name}`")
    df = pd.read_parquet(path)
    df[ID_COL] = df[ID_COL].astype(str)
    return df
