"""A tiny example pool, a fake encoder and the mock LLM backend."""
import hashlib

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.agentic.examples import ExamplePool
from src.ensemble.probs import ISSUE_TEAM, ISSUES
from src.utils.config import ID_COL, ISSUE_COL, TEAM_COL, TEXT_COL

DIM = 64


class FakeEncoder:
    name = "fake-encoder"

    def encode(self, texts, batch_size=64, show_progress=False):
        X = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                X[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(n == 0, 1, n)


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("agentic")
    enc = FakeEncoder()
    rows = [{ID_COL: f"{j}-{k}", TEAM_COL: ISSUE_TEAM[j], ISSUE_COL: iss,
             TEXT_COL: f"sample complaint about {iss} variant {k}"}
            for j, iss in enumerate(ISSUES) for k in range(3)]
    frame = pd.DataFrame(rows)
    ExamplePool(frame, enc.encode(frame[TEXT_COL].tolist())).save(d / "pool")
    joblib.dump({"model_name": "improved_model_2", "encoder_name": enc.name,
                 "max_seq_length": 256, "team_classes": [], "classifier": None,
                 "global_threshold": None}, d / "model.joblib")
    return d


@pytest.fixture
def client(artifacts, monkeypatch):
    monkeypatch.setenv("AGENTIC_POOL_PATH", str(artifacts / "pool"))
    monkeypatch.setenv("EMBED_MODEL_PATH", str(artifacts / "model.joblib"))
    monkeypatch.setenv("AGENTIC_LLM_CACHE", str(artifacts / "cache.sqlite"))
    monkeypatch.setenv("AGENTIC_BACKEND", "mock")
    monkeypatch.setattr("agentic_api.engine.encoder_for",
                        lambda raw, device=None: FakeEncoder())
    monkeypatch.setattr("src.agentic.config.BACKEND", "mock")
    from agentic_api.main import create_app
    with TestClient(create_app()) as c:
        yield c
