"""A tiny improved_model_2 bundle and a fake encoder, so these tests run on
the host without torch, a model download, or the real artifact."""
import hashlib

import joblib
import numpy as np
import pytest

from src.improved_model_2.encoder import prepare_text
from src.improved_model_2.model import calibrate, fit_classifier, make_bundle

VOCAB = {
    "T01": ["credit", "report", "bureau", "score", "inquiry"],
    "T02": ["collector", "debt", "calls", "collection", "agency"],
    "T06": ["mortgage", "escrow", "foreclosure", "servicer", "home"],
}


class FakeEncoder:
    name = "fake-encoder"

    def encode(self, texts, batch_size=64, show_progress=False):
        X = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in prepare_text(text).lower().split():
                X[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 64] += 1.0
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms == 0, 1.0, norms)


def _texts(n: int, seed: int):
    rng = np.random.default_rng(seed)
    teams = [list(VOCAB)[i % 3] for i in range(n)]
    return [" ".join(rng.choice(VOCAB[t], size=8)) for t in teams], teams


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    enc = FakeEncoder()
    train_x, train_y = _texts(150, 0)
    val_x, val_y = _texts(150, 1)
    clf = fit_classifier(enc.encode(train_x), train_y)
    cal = calibrate(clf, enc.encode(val_x), val_y)
    path = tmp_path_factory.mktemp("im2") / "model.joblib"
    joblib.dump(make_bundle(cal, enc.name, 256, global_threshold=0.5), path)
    return path


@pytest.fixture(scope="session")
def knn_index_path(tmp_path_factory):
    from src.ensemble.probs import ISSUES, TEAMS
    enc = FakeEncoder()
    texts, teams = _texts(90, 5)
    path = tmp_path_factory.mktemp("knn") / "index.npz"
    np.savez(path, embeddings=enc.encode(texts).astype(np.float16),
             team_idx=np.array([TEAMS.index(t) for t in teams], np.int16),
             issue_idx=np.zeros(90, np.int16), teams=np.array(TEAMS),
             issues=np.array(ISSUES), encoder_name=np.array(enc.name),
             max_seq_length=np.array(256), k=np.array(5))
    return path


@pytest.fixture
def app_env(model_path, knn_index_path, monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_PATH", str(model_path))
    monkeypatch.setenv("KNN_INDEX_PATH", str(knn_index_path))
    monkeypatch.setattr("embed_api.model.encoder_for",
                        lambda raw, device=None: FakeEncoder())
    return model_path
