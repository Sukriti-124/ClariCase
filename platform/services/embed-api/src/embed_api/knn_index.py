"""Nearest-neighbour team vote for serving.

Loads the index written by `src.ensemble.export_predictions` (float16
training embeddings + team labels) and votes exactly as the offline
`knn_embed` model did, so the ensemble sees the same signal it was fitted on.
The index is ~200 MB, so it is mounted into the container rather than baked
into the image; when it is absent the service runs without the vote.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ensemble.knn import topk_neighbours, vote
from src.ensemble.probs import TEAMS


class KnnIndex:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.path = Path(path)
        self.X = np.ascontiguousarray(z["embeddings"], dtype=np.float32)
        self.team_idx = z["team_idx"].astype(np.int64)
        self.k = int(z["k"])
        self.encoder_name = str(z["encoder_name"])
        self.max_seq_length = int(z["max_seq_length"])
        teams = [str(t) for t in z["teams"]]
        if teams != list(TEAMS):
            raise RuntimeError(f"index team order {teams} != {list(TEAMS)}")
        self.n = len(self.X)
        # Optional arrays: older indexes may lack issue labels or complaint
        # ids; the team vote never needs them.
        files = set(z.files)
        self.issue_idx = (z["issue_idx"].astype(np.int64)
                          if "issue_idx" in files else None)
        self.issues = ([str(i) for i in z["issues"]]
                       if "issues" in files else None)
        self.complaint_ids = (z["complaint_ids"]
                              if "complaint_ids" in files else None)
        for name in ("team_idx", "issue_idx", "complaint_ids"):
            arr = getattr(self, name)
            if arr is not None and len(arr) != self.n:
                raise RuntimeError(f"index {name} has {len(arr)} rows, "
                                   f"embeddings have {self.n}")

    def check_encoder(self, encoder_name: str, max_seq_length: int) -> None:
        if (encoder_name, max_seq_length) != (self.encoder_name,
                                              self.max_seq_length):
            raise RuntimeError(
                f"knn index was built with {self.encoder_name} "
                f"(len {self.max_seq_length}) but the model uses "
                f"{encoder_name} (len {max_seq_length}); rebuild the index")

    def neighbours(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(sims, idx) for one query, each with min(k, n) entries, most
        similar first."""
        sims, idx = topk_neighbours(self.X, np.asarray(q, np.float32)[None, :],
                                    self.k, backend="numpy")
        return sims[0], idx[0]

    def team_vote(self, q: np.ndarray) -> np.ndarray:
        sims, idx = self.neighbours(q)
        return vote(idx[None, :], sims[None, :], self.team_idx, len(TEAMS))[0]


def load_index(path: Path | None) -> KnnIndex | None:
    if path is None or not Path(path).exists():
        return None
    return KnnIndex(Path(path))
