"""Property test for the KNN_Explainer (Property 13, Req 6.1, 6.2)."""
from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.ensemble.knn import topk_neighbours, vote
from src.ensemble.probs import TEAMS
from src.explain.explainers.knn import explain_knn

ISSUES = ["I_A", "I_B", "I_C", "I_D"]


@st.composite
def knn_cases(draw):
    n = draw(st.integers(1, 40))
    d = draw(st.integers(2, 8))
    k = draw(st.integers(1, 30))
    seed = draw(st.integers(0, 2**32 - 1))
    non_positive = draw(st.booleans())
    with_ids = draw(st.booleans())
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    q = rng.normal(size=d).astype(np.float32)
    if non_positive:
        # Non-negative rows against a non-positive query: every similarity <= 0.
        X, q = np.abs(X), -np.abs(q)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    team_idx = rng.integers(0, len(TEAMS), n).astype(np.int16)
    issue_idx = rng.integers(0, len(ISSUES), n).astype(np.int16)
    ids = np.asarray([f"C{r:05d}" for r in range(n)], dtype=str) if with_ids else None
    return X, q, k, team_idx, issue_idx, ids


# Feature: model-explainability, Property 13: kNN record equals the served vote
@settings(max_examples=100, deadline=None)
@given(knn_cases())
def test_knn_record_equals_served_vote(case):
    """**Validates: Requirements 6.1, 6.2**"""
    X, q, k, team_idx, issue_idx, ids = case
    n = len(X)

    # The served vote, computed the way embed-api's KnnIndex does.
    sims, idx = topk_neighbours(X, q[None, :], k, backend="numpy")
    sims, idx = sims[0], idx[0]
    served = vote(idx[None, :], sims[None, :], team_idx, len(TEAMS))[0]
    served_probs = dict(zip(TEAMS, served.tolist()))

    issues = ISSUES if ids is not None else None
    issue_arg = issue_idx if ids is not None else None
    sec = explain_knn(sims, idx, team_idx, issue_arg, issues, ids, k, n, served_probs)
    d = sec.details

    assert sec.status == "ok"
    assert list(d.team_distribution) == list(TEAMS)
    for j, t in enumerate(TEAMS):
        assert abs(d.team_distribution[t] - float(served[j])) <= 1e-6
    assert abs(sum(d.team_distribution.values()) - 1.0) <= 1e-6
    assert d.max_abs_diff_vs_served <= 1e-6

    eff = min(k, n)
    assert d.effective_k == eff
    assert len(d.neighbours) == min(5, eff)
    assert [nb.rank for nb in d.neighbours] == list(range(1, min(5, eff) + 1))
    sim_list = [nb.similarity for nb in d.neighbours]
    assert all(a >= b for a, b in zip(sim_list, sim_list[1:]))

    # topk_neighbours returns rows most similar first, so rank r is idx[r - 1];
    # each neighbour's label is teams[team_idx[row]].
    for nb, row, s in zip(d.neighbours, idx, sims):
        row = int(row)
        assert abs(nb.similarity - float(s)) <= 1e-6
        assert nb.team_id == TEAMS[int(team_idx[row])]
        if ids is None:
            assert nb.complaint_id == "" and nb.issue_id is None
            assert d.reason_ids is not None
        else:
            assert nb.complaint_id == ids[row]
            assert nb.issue_id == ISSUES[int(issue_idx[row])]
