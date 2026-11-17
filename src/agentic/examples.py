"""Labelled example pool and per-candidate retrieval.

The pool is a capped sample of train_cap_issue (at most POOL_CAP_PER_ISSUE
complaints per issue, ~20k rows) with its sentence embeddings taken from
improved_model_2's cache. Retrieval is cosine similarity restricted to one
candidate at a time (a team or an issue), so each candidate in the prompt is
illustrated by the complaints most like the query *that carry that label*.
That is what lets the model tell neighbouring labels apart; the k most
similar examples overall would mostly carry the majority label.

Optional cross-encoder reranking re-scores a wider pool per candidate with
a query-example relevance model. It needs sentence-transformers and a
~90 MB download, so it is off unless asked for.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.agentic import config as C
from src.utils.config import ID_COL, ISSUE_COL, RANDOM_SEED, TEAM_COL, TEXT_COL

_REDACT = re.compile(r"\b[xX]{2,}\b")
_SPACE = re.compile(r"\s+")


def clip(text: str, limit: int) -> str:
    text = _SPACE.sub(" ", _REDACT.sub("XXXX", str(text or ""))).strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


@dataclass
class Example:
    complaint_id: str
    text: str
    team: str
    issue: str
    score: float


class ExamplePool:
    def __init__(self, frame: pd.DataFrame, X: np.ndarray):
        if len(frame) != len(X):
            raise ValueError("frame and embeddings differ in length")
        self.frame = frame.reset_index(drop=True)
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.team = self.frame[TEAM_COL].astype(str).to_numpy()
        self.issue = self.frame[ISSUE_COL].astype(str).to_numpy()
        self.text = self.frame[TEXT_COL].astype(str).to_numpy()
        self.ids = self.frame[ID_COL].astype(str).to_numpy()
        self._by_team = {t: np.flatnonzero(self.team == t)
                         for t in np.unique(self.team)}
        self._by_issue = {i: np.flatnonzero(self.issue == i)
                          for i in np.unique(self.issue)}
        self.reranker = None

    def __len__(self) -> int:
        return len(self.frame)

    # ---- build / persist ----
    @classmethod
    def build(cls, train: pd.DataFrame, X_train: np.ndarray,
              cap_per_issue: int = C.POOL_CAP_PER_ISSUE,
              seed: int = RANDOM_SEED) -> "ExamplePool":
        rng = np.random.default_rng(seed)
        keep = []
        for _, idx in train.groupby(ISSUE_COL).indices.items():
            idx = np.asarray(idx)
            if len(idx) > cap_per_issue:
                idx = rng.choice(idx, cap_per_issue, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
        frame = train.iloc[keep][[ID_COL, TEAM_COL, ISSUE_COL, TEXT_COL]].copy()
        frame[TEXT_COL] = frame[TEXT_COL].map(
            lambda t: clip(t, C.MAX_EXAMPLE_CHARS))
        return cls(frame, X_train[keep])

    def save(self, stem: Path = C.EXAMPLE_POOL) -> None:
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(stem.with_suffix(".parquet"), index=False)
        np.save(stem.with_suffix(".npy"), self.X.astype(np.float16))

    @classmethod
    def load(cls, stem: Path = C.EXAMPLE_POOL) -> "ExamplePool":
        frame = pd.read_parquet(stem.with_suffix(".parquet"))
        X = np.load(stem.with_suffix(".npy")).astype(np.float32)
        return cls(frame, X)

    # ---- retrieval ----
    def enable_reranker(self, name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(name)

    def _top(self, q: np.ndarray, idx: np.ndarray, k: int,
             query_text: str | None, widen: int = 10) -> list[Example]:
        if len(idx) == 0 or k <= 0:
            return []
        sims = self.X[idx] @ q
        n = min(len(idx), k * widen if self.reranker is not None else k)
        part = np.argpartition(-sims, n - 1)[:n] if n < len(idx) else np.arange(len(idx))
        part = part[np.argsort(-sims[part])]
        chosen = idx[part]
        scores = sims[part]
        if self.reranker is not None and query_text is not None:
            rs = self.reranker.predict([(query_text, self.text[i]) for i in chosen])
            order = np.argsort(-np.asarray(rs))[:k]
            chosen, scores = chosen[order], np.asarray(rs)[order]
        else:
            chosen, scores = chosen[:k], scores[:k]
        return [Example(self.ids[i], self.text[i], self.team[i], self.issue[i],
                        float(s)) for i, s in zip(chosen, scores)]

    def for_teams(self, q: np.ndarray, teams, k: int = C.EXAMPLES_PER_CANDIDATE,
                  query_text: str | None = None) -> dict[str, list[Example]]:
        return {t: self._top(q, self._by_team.get(t, np.array([], int)), k,
                             query_text) for t in teams}

    def for_issues(self, q: np.ndarray, issues,
                   k: int = C.EXAMPLES_PER_CANDIDATE,
                   query_text: str | None = None) -> dict[str, list[Example]]:
        return {i: self._top(q, self._by_issue.get(i, np.array([], int)), k,
                             query_text) for i in issues}


def build_pool_from_cache(embed_model: str = "improved_model_2",
                          device: str | None = None) -> ExamplePool:
    """Build the pool from train_cap_issue and improved_model_2's cached
    embeddings, and save it under data/cache/agentic/."""
    from src.ensemble.embeddings import load_embed_bundle, split_embeddings
    from src.utils.data import load_split
    train = load_split("train_cap_issue")[[ID_COL, TEAM_COL, ISSUE_COL, TEXT_COL]]
    train = train.dropna(subset=[TEXT_COL]).reset_index(drop=True)
    X = split_embeddings("train_cap_issue", train, load_embed_bundle(embed_model),
                         device)
    pool = ExamplePool.build(train, X)
    pool.save()
    print(f"[pool] {len(pool):,} examples saved to {C.EXAMPLE_POOL}.*")
    return pool
