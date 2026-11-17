"""KNN_Explainer: the knn_embed neighbour vote (design §5, Req 6).

``explain_knn`` takes the neighbours of one query, as returned by
``KnnIndex.neighbours`` (``sims`` and ``idx``, k_eff entries each, most
similar first), recomputes the team distribution with the same ``vote`` the
serving index uses, and records it with the nearest neighbours' ids and
labels. No neighbour text is ever read or recorded (Req 6.6).

Depends only on numpy and ``src.ensemble``; no platform package is imported.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.ensemble.knn import vote
from src.ensemble.probs import TEAMS
from src.explain.explainers import NOMINAL, skipped_section
from src.explain.record import KNNSection, Section, make_section

__all__ = ["explain_knn", "skipped_streamlit", "skipped_no_index",
           "MAX_NEIGHBOURS", "REASON_NO_IDS", "REASON_STREAMLIT", "REASON_NO_INDEX"]

NAME = "knn_embed"
MAX_NEIGHBOURS = 5
REASON_NO_IDS = "index has no complaint ids"
REASON_STREAMLIT = "not available on this serving path"
REASON_NO_INDEX = "knn index not loaded"


def _row(a: Any, what: str, dtype) -> np.ndarray:
    """One query row: accepts shape (k,) or (1, k)."""
    arr = np.asarray(a, dtype=dtype)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"{what} must have shape (k,) or (1, k), got {arr.shape}")
    return arr


def explain_knn(sims: np.ndarray, idx: np.ndarray, team_idx: Any,
                issue_idx: Any | None, issues: Sequence[str] | None,
                complaint_ids: Any | None, k: int, n_index: int,
                served_probs: Mapping[str, float] | None) -> Section:
    """The ``knn_embed`` section for one query (Req 6.1, 6.2, 6.4, 6.5, 6.6).

    ``served_probs`` maps team id to the knn_embed probability the
    Ensemble_API used; ``max_abs_diff_vs_served`` is ``None`` without it.
    """
    s = _row(sims, "sims", np.float64)
    i = _row(idx, "idx", np.int64)
    if len(s) != len(i):
        raise ValueError(f"sims and idx lengths differ ({len(s)} != {len(i)})")
    effective_k = min(int(k), int(n_index))
    if len(i) != effective_k:
        raise ValueError(f"got {len(i)} neighbours, expected effective k {effective_k}")
    team_idx = np.asarray(team_idx)

    # Same arithmetic as KnnIndex.team_vote, so the distribution is exact.
    p = vote(np.asarray(idx, np.int64).reshape(1, -1),
             np.asarray(sims).reshape(1, -1), team_idx, len(TEAMS))[0]
    dist = {t: float(p[j]) for j, t in enumerate(TEAMS)}

    max_diff = None
    if served_probs is not None:
        missing = [t for t in TEAMS if t not in served_probs]
        if missing:
            raise KeyError(f"served knn_embed probabilities lack teams {missing}")
        max_diff = float(max(abs(dist[t] - float(served_probs[t])) for t in TEAMS))

    has_ids = complaint_ids is not None
    has_issues = issue_idx is not None and issues is not None
    order = np.argsort(-s, kind="stable")[:min(MAX_NEIGHBOURS, effective_k)]
    neighbours = []
    for rank, j in enumerate(order, start=1):
        row = int(i[j])
        neighbours.append({
            "rank": rank,
            "similarity": float(s[j]),
            "team_id": TEAMS[int(team_idx[row])],
            "issue_id": str(issues[int(issue_idx[row])]) if has_issues else None,
            "complaint_id": str(complaint_ids[row]) if has_ids else "",
        })

    method, technique = NOMINAL[NAME]
    return make_section(NAME, status="ok", method=method, technique=technique,
                        details={"k": int(k), "n_index": int(n_index),
                                 "effective_k": effective_k,
                                 "team_distribution": dist,
                                 "max_abs_diff_vs_served": max_diff,
                                 "neighbours": neighbours,
                                 "reason_ids": None if has_ids else REASON_NO_IDS})


def skipped_streamlit() -> KNNSection:
    """The vote does not run on the Streamlit Serving_Path (Req 6.8)."""
    return skipped_section(NAME, REASON_STREAMLIT)


def skipped_no_index() -> KNNSection:
    """The kNN index could not be loaded (Req 6.8)."""
    return skipped_section(NAME, REASON_NO_INDEX)
