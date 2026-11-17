"""Which complaints get asked, and with which candidates.

Inputs are the prediction files from round 2:
  ensemble_stack/<split>.parquet     stacker team, gate score, auto_route, p_<team>
  improved_model_1/<split>_issues.parquet   q_<issue> = P(issue | its team)

Team task: every complaint the gate did not auto-route. Candidates are the
stacker's top-N teams, its own choice first.

Issue task: every complaint whose ML issue head is unsure inside the final
team (top-1 minus top-2 below ISSUE_GAP_THRESHOLD). Candidates are the
head's top-N issues in that team, its own choice first. The final team is
the LLM's pick where the team task ran, else the stacker's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.agentic import config as C
from src.ensemble.data import read_predictions
from src.ensemble.probs import (ISSUE_TEAM, ISSUES, TEAMS, block_normalize,
                                issue_matrix, read_issue_probs, team_matrix)
from src.utils.config import ID_COL

TEAM_ARR = np.asarray(TEAMS)
ISSUE_ARR = np.asarray(ISSUES)


@dataclass
class SplitData:
    ids: np.ndarray
    y_team: np.ndarray
    y_issue: np.ndarray
    stack_team: np.ndarray
    gate_score: np.ndarray
    auto_route: np.ndarray
    P: np.ndarray          # stacker team probabilities
    Q: np.ndarray          # issue probabilities, conditional per team block

    def __len__(self) -> int:
        return len(self.ids)


def load_split_data(split: str, team_model: str = "ensemble_stack",
                    issue_model: str = "improved_model_1") -> SplitData:
    s = read_predictions(team_model, split)
    q = read_issue_probs(issue_model, split)
    d = s.merge(q.drop(columns=["actual_team_id"]), on=ID_COL, how="inner",
                validate="one_to_one")
    if "auto_route" not in d.columns:
        raise KeyError(f"{team_model}/{split} has no auto_route column; "
                       "run src.ensemble.train_stack first")
    return SplitData(
        ids=d[ID_COL].astype(str).to_numpy(),
        y_team=d["actual_team_id"].astype(str).to_numpy(),
        y_issue=d["actual_issue_id"].astype(str).to_numpy(),
        stack_team=d["predicted_team_id"].astype(str).to_numpy(),
        gate_score=d["confidence"].to_numpy(np.float64),
        auto_route=d["auto_route"].to_numpy(bool),
        P=team_matrix(d),
        Q=block_normalize(issue_matrix(d), ISSUE_TEAM),
    )


def team_candidates(P: np.ndarray, n: int = C.N_TEAM_CANDIDATES) -> np.ndarray:
    """(rows, n) team ids, best first. Row's argmax is always first."""
    order = np.argsort(-P, axis=1, kind="stable")[:, :n]
    return TEAM_ARR[order]


def issue_candidates(Q: np.ndarray, team: np.ndarray,
                     n: int = C.N_ISSUE_CANDIDATES
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Top-n issues inside each row's team ('' where the team has fewer),
    and the top-1 minus top-2 probability gap (1.0 for a single issue)."""
    team = np.asarray(team).astype(str)
    L = np.where(ISSUE_TEAM[None, :] == team[:, None], Q, -np.inf)
    order = np.argsort(-L, axis=1, kind="stable")[:, :n]
    vals = np.take_along_axis(L, order, axis=1)
    names = np.where(np.isfinite(vals), ISSUE_ARR[order], "")
    second = np.where(np.isfinite(vals[:, 1]), vals[:, 1], 0.0) if n > 1 \
        else np.zeros(len(team))
    gap = np.where(np.isfinite(vals[:, 0]), vals[:, 0] - second, 1.0)
    return names, gap


def team_task_rows(d: SplitData) -> np.ndarray:
    return np.flatnonzero(~d.auto_route)


def issue_task_rows(gap: np.ndarray, names: np.ndarray,
                    threshold: float = C.ISSUE_GAP_THRESHOLD) -> np.ndarray:
    has_two = names[:, 1] != ""
    return np.flatnonzero(has_two & (gap < threshold))
