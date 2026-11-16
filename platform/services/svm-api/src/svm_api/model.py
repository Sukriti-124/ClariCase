"""Model artifact loader and prediction wrapper for svm-api.

Wraps improved_model_1's bundle (vectorizer + hierarchical SVM) and, when
BASELINE2_MODEL_PATH points at an artifact, baseline_model_2 (TF-IDF +
Naive Bayes) as a co-hosted second opinion. Both are TF-IDF models with the
same scikit-learn pin, so one container serves both.

Prediction follows src.improved_model_1.predict.predict step for step, but
keeps the full team probability vector (the ensemble needs it) instead of
the top-k only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.ensemble.probs import TEAMS, reorder


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _probs_dict(P: np.ndarray) -> dict[str, float]:
    return {t: float(p) for t, p in zip(TEAMS, P)}


@dataclass(frozen=True)
class ModelBundle:
    raw: Any
    version: str
    model_name: str
    artifact_path: Path
    baseline2: Any = None
    baseline2_version: str | None = None
    extra_models: tuple[str, ...] = field(default=())

    # ---- improved_model_1 ----
    def _team_probs(self, X) -> np.ndarray:
        proba = self.raw["model"].team_proba(X)
        return reorder(proba, self.raw["team_classes"], TEAMS)[0]

    def issues_for_team(self, X, team: str, top_k: int) -> list[dict]:
        hsvm = self.raw["model"]
        if team not in hsvm.issue_clfs_:
            return []
        rows = hsvm.predict_issue_topk(X, teams=np.array([team]), k=top_k)[0]
        return [{"issue": iid, "prob": float(conf)} for iid, conf in rows]

    def predict(self, text: str, top_k: int = 3) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("text is empty")
        X = self.raw["vectorizer"].transform([text])
        P = self._team_probs(X)
        j = int(P.argmax())
        team, confidence = TEAMS[j], float(P[j])
        thr = self.raw.get("global_threshold")
        out = {
            "model_name": self.model_name,
            "model_version": self.version,
            "team": team,
            "team_confidence": confidence,
            "top_issues": self.issues_for_team(X, team, top_k),
            "auto_route": thr is not None and confidence >= thr,
            "team_probs": _probs_dict(P),
        }
        if self.baseline2 is not None:
            B = reorder(self.baseline2.predict_proba([text]),
                        self.baseline2.classes_, TEAMS)[0]
            out["extra_team_probs"] = {"baseline_model_2": _probs_dict(B)}
        return out

    def issues(self, text: str, team: str, top_k: int = 3) -> list[dict]:
        if not text or not text.strip():
            raise ValueError("text is empty")
        if team not in TEAMS:
            raise KeyError(f"unknown team {team!r}")
        X = self.raw["vectorizer"].transform([text])
        return self.issues_for_team(X, team, top_k)


def load_bundle(path: Path, baseline2_path: Path | None = None) -> ModelBundle:
    raw = joblib.load(path)
    b2 = b2_version = None
    extra: tuple[str, ...] = ()
    if baseline2_path is not None and Path(baseline2_path).exists():
        b2 = joblib.load(baseline2_path)
        b2_version = _fingerprint(Path(baseline2_path))
        extra = ("baseline_model_2",)
    return ModelBundle(
        raw=raw,
        version=_fingerprint(path),
        model_name=raw.get("model_name", "improved_model_1"),
        artifact_path=path,
        baseline2=b2,
        baseline2_version=b2_version,
        extra_models=extra,
    )
