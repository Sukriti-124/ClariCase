"""Load the fitted TF-IDF model and route a complaint to a team.

Run:
    python -m src.baseline_model_1.predict "My credit card was charged twice ..."
"""

from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from src.baseline_model_1.config import ARTIFACTS_DIR, TEAM_MAPPING


def load_model(path=ARTIFACTS_DIR / "model.joblib"):
    return joblib.load(path)


def _team_lookup() -> dict[str, str]:
    df = pd.read_csv(TEAM_MAPPING)
    return dict(zip(df["team_id"], df["team_name"]))


def predict(text: str, top_k: int = 3) -> dict:
    model = load_model()
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    order = proba.argsort()[::-1][:top_k]
    lookup = _team_lookup()
    return {
        "predicted_team_id": str(classes[order[0]]),
        "predicted_team_name": lookup.get(str(classes[order[0]]), ""),
        "confidence": float(proba[order[0]]),
        "top_k": [
            {"team_id": str(classes[i]),
             "team_name": lookup.get(str(classes[i]), ""),
             "confidence": float(proba[i])}
            for i in order
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Complaint narrative")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    text = args.text or sys.stdin.read()
    if not text.strip():
        raise SystemExit("no complaint text provided")

    import json
    print(json.dumps(predict(text, top_k=args.top_k), indent=2))
