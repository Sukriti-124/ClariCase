"""improved_model_2 with a fake encoder, so no torch or model download."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from src.improved_model_2 import train as T
from src.improved_model_2.encoder import prepare_text
from src.improved_model_2.predict import predict
from src.utils import config as C


class FakeEncoder:
    """Deterministic hashed bag-of-words, L2-normalised."""
    name = "fake-encoder"

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.calls = 0

    def encode(self, texts, batch_size=64, show_progress=False):
        self.calls += 1
        X = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in prepare_text(text).lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                X[i, h % self.dim] += 1.0
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms == 0, 1.0, norms)


VOCAB = {
    "T01": ["credit", "report", "bureau", "score", "inquiry"],
    "T02": ["collector", "debt", "calls", "collection", "agency"],
    "T06": ["mortgage", "escrow", "foreclosure", "servicer", "home"],
}


def _frame(n: int, seed: int, start_id: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        team = list(VOCAB)[i % 3]
        words = list(rng.choice(VOCAB[team], size=8))
        # A little cross-team noise so confidence is not always 1.0.
        other = list(VOCAB)[(i + 1) % 3]
        words += list(rng.choice(VOCAB[other], size=int(rng.integers(0, 4))))
        rows.append({C.ID_COL: str(start_id + i), C.TEAM_COL: team,
                     C.TEXT_COL: " ".join(words) + " XXXX"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frames():
    return _frame(300, 0, 0), _frame(240, 1, 10_000), _frame(240, 2, 20_000)


def test_prepare_text_drops_redactions_and_extra_space():
    assert prepare_text("  I paid  XXXX on XX/XX/2025 \n ok ") == \
        "I paid on / /2025 ok"


def test_run_trains_calibrates_and_routes(frames, tmp_path):
    train, val, test = frames
    out = T.run(train, val, test, FakeEncoder(), tmp_path, 32, 256, {},
                verbose=False)
    assert out["metrics"]["test"]["accuracy"] > 0.9
    assert out["bundle"]["model_name"] == "improved_model_2"
    assert out["bundle"]["team_classes"] == ["T01", "T02", "T06"]
    assert out["bundle"]["encoder_name"] == "fake-encoder"
    for split in ("val", "test"):
        df = out[f"{split}_predictions"]
        assert list(df.columns)[:5] == [C.ID_COL, "actual_team_id",
                                        "predicted_team_id", "confidence",
                                        "n_words"]
        p = df[["p_T01", "p_T02", "p_T06"]].to_numpy()
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
        assert (df["predicted_team_id"].to_numpy()
                == np.array(["T01", "T02", "T06"])[p.argmax(1)]).all()
        assert df["confidence"].between(0, 1).all()
    assert out["ns"]["n_test"] == len(test)


def test_embedding_cache_is_reused_only_for_same_ids(frames, tmp_path):
    train, _, _ = frames
    enc = FakeEncoder()
    a = T.embed_split("train", train, enc, tmp_path, 32, verbose=False)
    b = T.embed_split("train", train, enc, tmp_path, 32, verbose=False)
    assert enc.calls == 1
    np.testing.assert_array_equal(a, b)
    T.embed_split("train", train.iloc[:50], enc, tmp_path, 32, verbose=False)
    assert enc.calls == 2  # different rows -> cache not trusted


def test_predict_matches_improved_model_1_output_keys(frames, tmp_path):
    train, val, test = frames
    enc = FakeEncoder()
    bundle = T.run(train, val, test, enc, None, 32, 256, {},
                   verbose=False)["bundle"]
    out = predict("the collector calls about a debt every day", top_k=2,
                  model=bundle, encoder=enc, team_names={"T02": "Debt"})
    assert set(out) == {"predicted_team_id", "predicted_team_name",
                        "confidence", "top_k", "issue_top_k", "route",
                        "model_name"}
    assert out["predicted_team_id"] == "T02"
    assert out["predicted_team_name"] == "Debt"
    assert len(out["top_k"]) == 2
    assert out["issue_top_k"] == []
    assert out["route"] in ("auto", "review")
