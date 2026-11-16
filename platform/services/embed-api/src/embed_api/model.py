"""Model artifact loader and prediction wrapper for embed-api.

- loads improved_model_2's `model.joblib` and the sentence encoder it names
- optionally loads the nearest-neighbour index (knn_embed) and serves its
  team vote as a co-hosted second opinion; both reuse the one embedding
- computes a sha256 fingerprint for `model_version`
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.ensemble.probs import TEAMS, reorder
from src.improved_model_2 import predict as upstream


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def encoder_for(raw: dict, device: str | None = None):
    """Separate function so tests can swap in a fake encoder."""
    return upstream.encoder_for(raw, device)


def _probs_dict(P: np.ndarray) -> dict[str, float]:
    return {t: float(p) for t, p in zip(TEAMS, P)}


@dataclass(frozen=True)
class ModelBundle:
    raw: Any
    encoder: Any
    version: str
    model_name: str
    artifact_path: Path
    knn: Any = None

    @property
    def extra_models(self) -> tuple[str, ...]:
        return ("knn_embed",) if self.knn is not None else ()

    def predict(self, text: str, top_k: int = 3) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("text is empty")
        q = self.encoder.encode([text])[0]
        P = reorder(self.raw["classifier"].predict_proba(q[None, :]),
                    self.raw["team_classes"], TEAMS)[0]
        j = int(P.argmax())
        team, confidence = TEAMS[j], float(P[j])
        thr = self.raw.get("global_threshold")
        out = {
            "model_name": self.model_name,
            "model_version": self.version,
            "team": team,
            "team_confidence": confidence,
            "top_issues": [],  # improved_model_2 predicts the team only
            "auto_route": thr is not None and confidence >= thr,
            "team_probs": _probs_dict(P),
        }
        if self.knn is not None:
            out["extra_team_probs"] = {"knn_embed": _probs_dict(
                self.knn.team_vote(q))}
        return out


def load_bundle(path: Path, device: str | None = None,
                knn_index_path: Path | None = None) -> ModelBundle:
    from embed_api.knn_index import load_index
    raw = joblib.load(path)
    knn = load_index(knn_index_path)
    if knn is not None:
        knn.check_encoder(raw["encoder_name"], raw["max_seq_length"])
    return ModelBundle(
        raw=raw,
        encoder=encoder_for(raw, device),
        version=_fingerprint(path),
        model_name=raw.get("model_name", "improved_model_2"),
        artifact_path=path,
        knn=knn,
    )
