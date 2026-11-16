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

    def check_encoder(self, encoder_name: str, max_seq_length: int) -> None:
        if (encoder_name, max_seq_length) != (self.encoder_name,
                                              self.max_seq_length):
            raise RuntimeError(
                f"knn index was built with {self.encoder_name} "
                f"(len {self.max_seq_length}) but the model uses "
                f"{encoder_name} (len {max_seq_length}); rebuild the index")

    def team_vote(self, q: np.ndarray) -> np.ndarray:
        sims, idx = topk_neighbours(self.X, np.asarray(q, np.float32)[None, :],
                                    self.k, backend="numpy")
        return vote(idx, sims, self.team_idx, len(TEAMS))[0]


def load_index(path: Path | None) -> KnnIndex | None:
    if path is None or not Path(path).exists():
        return None
    return KnnIndex(Path(path))
