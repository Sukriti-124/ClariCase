"""Unit tests for the KNN_Explainer (Req 6.1, 6.2, 6.4, 6.5, 6.6, 6.8)."""
from __future__ import annotations

import numpy as np
import pytest

from src.ensemble.knn import topk_neighbours, vote
from src.ensemble.probs import TEAMS
from src.explain.explainers.knn import (REASON_NO_IDS, REASON_NO_INDEX,
                                        REASON_STREAMLIT, explain_knn,
                                        skipped_no_index, skipped_streamlit)
from src.explain.record import KNNSection


def _index(n=40, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    team_idx = rng.integers(0, len(TEAMS), n).astype(np.int16)
    issue_idx = rng.integers(0, 3, n).astype(np.int16)
    issues = ["I_A", "I_B", "I_C"]
    ids = np.asarray([f"C{r:04d}" for r in range(n)], dtype=str)
    q = rng.normal(size=d).astype(np.float32)
    q /= np.linalg.norm(q)
    return X, team_idx, issue_idx, issues, ids, q


def _served(X, team_idx, q, k):
    sims, idx = topk_neighbours(X, q[None, :], k, backend="numpy")
    p = vote(idx, sims, team_idx, len(TEAMS))[0]
    return sims[0], idx[0], dict(zip(TEAMS, p.tolist()))


def test_ok_section_matches_served_vote():
    X, team_idx, issue_idx, issues, ids, q = _index()
    sims, idx, served = _served(X, team_idx, q, k=10)
    sec = explain_knn(sims, idx, team_idx, issue_idx, issues, ids, 10, len(X), served)

    assert isinstance(sec, KNNSection)
    assert (sec.status, sec.method, sec.technique) == ("ok", "exact", "neighbours")
    d = sec.details
    assert d.effective_k == 10 and d.reason_ids is None
    assert list(d.team_distribution) == list(TEAMS)
    assert abs(sum(d.team_distribution.values()) - 1.0) < 1e-6
    assert d.max_abs_diff_vs_served < 1e-6
    assert [nb.rank for nb in d.neighbours] == [1, 2, 3, 4, 5]
    sim_list = [nb.similarity for nb in d.neighbours]
    assert sim_list == sorted(sim_list, reverse=True)
    for nb, row in zip(d.neighbours, idx[:5]):
        assert nb.complaint_id == ids[row]
        assert nb.team_id == TEAMS[team_idx[row]]
        assert nb.issue_id == issues[issue_idx[row]]


def test_small_index_records_effective_k_neighbours():
    X, team_idx, issue_idx, issues, ids, q = _index(n=3)
    sims, idx, served = _served(X, team_idx, q, k=25)
    sec = explain_knn(sims, idx, team_idx, issue_idx, issues, ids, 25, 3, served)
    assert sec.details.effective_k == 3
    assert len(sec.details.neighbours) == 3


def test_no_complaint_ids_gives_empty_ids_and_reason():
    X, team_idx, issue_idx, issues, _, q = _index()
    sims, idx, served = _served(X, team_idx, q, k=5)
    sec = explain_knn(sims, idx, team_idx, None, None, None, 5, len(X), served)
    assert sec.status == "ok"
    assert sec.details.reason_ids == REASON_NO_IDS
    assert all(nb.complaint_id == "" and nb.issue_id is None
               for nb in sec.details.neighbours)


def test_non_positive_similarities_give_uniform_distribution():
    team_idx = np.arange(6) % len(TEAMS)
    sims = np.array([0.0, -0.1, -0.2, -0.3, -0.4, -0.5])
    idx = np.arange(6)
    served = {t: 1.0 / len(TEAMS) for t in TEAMS}
    sec = explain_knn(sims, idx, team_idx, None, None, None, 6, 6, served)
    assert all(abs(v - 1.0 / len(TEAMS)) < 1e-12
               for v in sec.details.team_distribution.values())
    assert sec.details.max_abs_diff_vs_served < 1e-12


def test_no_text_fields_recorded():
    X, team_idx, issue_idx, issues, ids, q = _index()
    sims, idx, served = _served(X, team_idx, q, k=5)
    sec = explain_knn(sims, idx, team_idx, issue_idx, issues, ids, 5, len(X), served)
    for nb in sec.details.neighbours:
        assert set(nb.model_dump()) == {"rank", "similarity", "team_id",
                                        "issue_id", "complaint_id"}
    assert sec.drivers == []


def test_missing_served_team_raises():
    X, team_idx, issue_idx, issues, ids, q = _index()
    sims, idx, served = _served(X, team_idx, q, k=5)
    served.pop(TEAMS[0])
    with pytest.raises(KeyError):
        explain_knn(sims, idx, team_idx, issue_idx, issues, ids, 5, len(X), served)


@pytest.mark.parametrize("fn, reason", [(skipped_streamlit, REASON_STREAMLIT),
                                        (skipped_no_index, REASON_NO_INDEX)])
def test_skipped_helpers(fn, reason):
    sec = fn()
    assert isinstance(sec, KNNSection)
    assert (sec.status, sec.reason, sec.method, sec.technique) == (
        "skipped", reason, "exact", "neighbours")
    assert sec.drivers == [] and sec.details.neighbours == []
