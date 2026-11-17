"""Importable + CLI prediction for improved_model_2.

The output dict has the same keys as src.improved_model_1.predict.predict so
callers (and the platform service) can treat both models alike. This model
predicts the team only, so `issue_top_k` is empty.

Usage:
    python -m src.improved_model_2.predict "complaint text here"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from src.improved_model_2.config import ARTIFACT_DIR
from src.improved_model_2.encoder import Encoder, load_encoder


def load_model(path: Path = ARTIFACT_DIR / "model.joblib") -> dict:
    return joblib.load(path)


def encoder_for(bundle: dict, device: str | None = None) -> Encoder:
    return load_encoder(bundle["encoder_name"], bundle["max_seq_length"],
                        device)


def predict(text: str, top_k: int = 3, model: dict | None = None,
            encoder: Encoder | None = None,
            team_names: dict[str, str] | None = None) -> dict:
    bundle = model if model is not None else load_model()
    enc = encoder if encoder is not None else encoder_for(bundle)
    team_names = team_names or {}
    classes = np.asarray(bundle["team_classes"])

    X = enc.encode([text])
    proba = bundle["classifier"].predict_proba(X)[0]
    order = np.argsort(proba)[::-1][:top_k]
    team = str(classes[order[0]])
    confidence = float(proba[order[0]])
    thr = bundle.get("global_threshold")

    return {
        "predicted_team_id": team,
        "predicted_team_name": team_names.get(team, ""),
        "confidence": confidence,
        "top_k": [{"team_id": str(classes[j]),
                   "team_name": team_names.get(str(classes[j]), ""),
                   "confidence": float(proba[j])} for j in order],
        "issue_top_k": [],
        "route": "auto" if thr is not None and confidence >= thr else "review",
        "model_name": bundle.get("model_name", "improved_model_2"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    text = args.text or sys.stdin.read()
    if not text.strip():
        raise SystemExit("no complaint text provided")
    from src.utils.labels import team_names as _names
    print(json.dumps(predict(text, top_k=args.top_k, team_names=_names),
                     indent=2))
