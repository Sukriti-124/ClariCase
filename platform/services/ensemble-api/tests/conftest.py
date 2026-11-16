"""A small stack bundle fitted on synthetic data with the real training
code, so the service exercises the same stacker / gate / threshold objects
it will load in production."""
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.ensemble import gate as G
from src.ensemble.label_shift import prior_from_labels
from src.ensemble.probs import TEAMS
from src.ensemble.stack import make_stacker, stack_features
from src.ensemble.thresholds import choose_threshold, team_floor_thresholds

MODELS = ["improved_model_1", "improved_model_2", "baseline_model_2",
          "knn_embed"]
T = len(TEAMS)


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _synthetic(n, seed):
    rng = np.random.default_rng(seed)
    yi = rng.choice(T, n, p=[0.55, 0.15, 0.07, 0.07, 0.04, 0.03, 0.02, 0.03,
                             0.02, 0.01, 0.01])
    mats = []
    for acc, sharp in ((0.85, 3.0), (0.80, 2.5), (0.75, 2.0), (0.75, 2.0)):
        hit = rng.random(n) < acc
        noisy = np.where(hit, yi, rng.integers(0, T, n))
        mats.append(_softmax(sharp * np.eye(T)[noisy] + rng.normal(0, 0.8, (n, T))))
    return yi, mats, rng.integers(20, 300, n)


def build_bundle():
    yi, mats, n_words = _synthetic(6000, 0)
    stacker = make_stacker(0.1).fit(stack_features(mats), yi)
    P = np.zeros((len(yi), T))
    P[:, stacker[-1].classes_] = stacker.predict_proba(stack_features(mats))
    chosen = P.argmax(1)
    X = G.prob_features(chosen, [P] + mats, n_words, T)
    gate = G.make_estimator("hgb", 0).fit(X, (chosen == yi).astype(int))
    score = gate.predict_proba(X)[:, 1]
    y = np.array(TEAMS)[yi]
    ch = np.array(TEAMS)[chosen]
    base = choose_threshold(score, ch == y, 0.9, 0.0025, wilson=True) or 0.8
    teams = team_floor_thresholds(score, ch, y, base, 0.85, margin=0.0025,
                                  min_routed=30)
    p1 = mats[0]
    c1 = np.array(TEAMS)[p1.argmax(1)]
    fb_base = choose_threshold(p1.max(1), c1 == y, 0.9, 0.0025,
                               wilson=True) or 0.8
    return {
        "model_name": "ensemble_stack_v2", "models": MODELS,
        "teams": list(TEAMS), "scenario": "stack_gate",
        "stacker": stacker, "gate": gate, "gate_kind": "hgb",
        "base_threshold": base, "team_thresholds": teams,
        "target": 0.9, "floor": 0.85,
        "source_prior": prior_from_labels(y, TEAMS).tolist(),
        "primary_name": "improved_model_1",
        "primary_fallback_policy": {"base": fb_base, "teams": None},
    }


@pytest.fixture(scope="session")
def stack_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("stack") / "stack.joblib"
    joblib.dump(build_bundle(), path)
    return path


@pytest.fixture
def client(stack_path, monkeypatch):
    monkeypatch.setenv("STACK_PATH", str(stack_path))
    monkeypatch.setenv("SVM_API_URL", "http://svm-api.test")
    monkeypatch.setenv("EMBED_API_URL", "http://embed-api.test")
    monkeypatch.setenv("PRIOR_MIN_ROWS", "40")
    monkeypatch.setenv("PRIOR_REFRESH", "10")
    from ensemble_api.main import create_app
    with TestClient(create_app()) as c:
        yield c


def probs_for(team: str, conf: float) -> dict:
    rest = (1 - conf) / (T - 1)
    return {t: (conf if t == team else rest) for t in TEAMS}
