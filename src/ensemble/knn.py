"""Nearest training complaints by sentence-embedding similarity, and the
team / issue vote among them.

Embeddings are L2-normalised, so a dot product is cosine similarity. Uses
torch (GPU if available) when it is installed, otherwise numpy.
"""
from __future__ import annotations

import numpy as np


def _resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    try:
        import torch  # noqa: F401
        return "torch"
    except ImportError:
        return "numpy"


def topk_neighbours(train_X: np.ndarray, query_X: np.ndarray, k: int = 25,
                    chunk: int = 512, backend: str = "auto",
                    verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(similarities, indices), each (n_query, k), most similar first."""
    train_X = np.ascontiguousarray(train_X, dtype=np.float32)
    query_X = np.ascontiguousarray(query_X, dtype=np.float32)
    k = min(k, len(train_X))
    nq = len(query_X)
    sims = np.empty((nq, k), dtype=np.float32)
    idx = np.empty((nq, k), dtype=np.int64)
    backend = _resolve_backend(backend)

    if backend == "torch":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        T = torch.from_numpy(train_X).to(dev)
        with torch.inference_mode():
            for s in range(0, nq, chunk):
                q = torch.from_numpy(query_X[s:s + chunk]).to(dev)
                v, i = torch.topk(q @ T.T, k, dim=1)
                sims[s:s + chunk] = v.cpu().numpy()
                idx[s:s + chunk] = i.cpu().numpy()
                if verbose and (s // chunk) % 50 == 0:
                    print(f"  [knn] {min(s + chunk, nq):,}/{nq:,}", flush=True)
        return sims, idx

    for s in range(0, nq, chunk):
        S = query_X[s:s + chunk] @ train_X.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-vals, axis=1, kind="stable")
        sims[s:s + chunk] = np.take_along_axis(vals, order, axis=1)
        idx[s:s + chunk] = np.take_along_axis(part, order, axis=1)
        if verbose and (s // chunk) % 50 == 0:
            print(f"  [knn] {min(s + chunk, nq):,}/{nq:,}", flush=True)
    return sims, idx


def vote(idx: np.ndarray, sims: np.ndarray, labels: np.ndarray,
         n_classes: int, power: float = 1.0) -> np.ndarray:
    """Similarity-weighted share of neighbours carrying each label.
    `labels` holds an integer class index per training row."""
    n, k = idx.shape
    w = np.clip(sims.astype(np.float64), 0.0, None) ** power
    lab = np.asarray(labels)[idx]
    flat = (np.repeat(np.arange(n), k) * n_classes + lab.ravel())
    out = np.bincount(flat, weights=w.ravel(),
                      minlength=n * n_classes).reshape(n, n_classes)
    s = out.sum(axis=1, keepdims=True)
    return np.where(s > 0, out / np.where(s > 0, s, 1.0), 1.0 / n_classes)
