"""KnnIndex: optional issue labels / complaint ids and the neighbours() split
leave the team vote unchanged."""
import numpy as np

from embed_api.knn_index import KnnIndex
from src.ensemble.knn import topk_neighbours, vote
from src.ensemble.probs import ISSUES, TEAMS


def _write(path, n=12, k=5, with_ids=True, with_issues=True, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 16)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    arrays = dict(embeddings=X.astype(np.float16),
                  team_idx=(np.arange(n) % len(TEAMS)).astype(np.int16),
                  teams=np.array(TEAMS), encoder_name=np.array("fake"),
                  max_seq_length=np.array(256), k=np.array(k))
    if with_issues:
        arrays["issue_idx"] = (np.arange(n) % len(ISSUES)).astype(np.int16)
        arrays["issues"] = np.array(ISSUES)
    if with_ids:
        arrays["complaint_ids"] = np.asarray([f"c{i}" for i in range(n)],
                                             dtype=str)
    np.savez(path, **arrays)
    return X


def _old_team_vote(index, q):
    # The pre-change implementation, verbatim.
    sims, idx = topk_neighbours(index.X, np.asarray(q, np.float32)[None, :],
                                index.k, backend="numpy")
    return vote(idx, sims, index.team_idx, len(TEAMS))[0]


def test_old_index_loads_without_optional_arrays(tmp_path):
    p = tmp_path / "old.npz"
    X = _write(p, with_ids=False, with_issues=False)
    index = KnnIndex(p)
    assert index.issue_idx is None and index.issues is None
    assert index.complaint_ids is None
    for q in X[:4]:
        np.testing.assert_array_equal(index.team_vote(q),
                                      _old_team_vote(index, q))


def test_new_index_exposes_ids_and_neighbours(tmp_path):
    p = tmp_path / "new.npz"
    X = _write(p)
    index = KnnIndex(p)
    assert list(index.complaint_ids) == [f"c{i}" for i in range(12)]
    assert index.issues == list(ISSUES)
    sims, idx = index.neighbours(X[3])
    assert sims.shape == idx.shape == (5,)
    assert idx[0] == 3 and np.all(np.diff(sims) <= 0)
    np.testing.assert_array_equal(index.team_vote(X[3]),
                                  _old_team_vote(index, X[3]))


def test_neighbours_capped_at_index_size(tmp_path):
    p = tmp_path / "small.npz"
    X = _write(p, n=3, k=25)
    sims, idx = KnnIndex(p).neighbours(X[0])
    assert len(sims) == len(idx) == 3
