"""Gate reference medians (model-explainability Req 8.7), on tiny synthetic data."""
import numpy as np
import pandas as pd
import pytest

from src.ensemble import gate as G
from src.ensemble.train_stack import Aligned, build_scenarios

TEAMS3 = ["T01", "T02", "T03"]
MODELS = ["m1", "m2"]


def _aligned(n, seed, start):
    rng = np.random.default_rng(seed)
    y = rng.choice(TEAMS3, n, p=[0.5, 0.3, 0.2])
    yi = np.array([TEAMS3.index(t) for t in y])
    probs = {}
    for name, acc in (("m1", 0.8), ("m2", 0.7)):
        hit = rng.random(n) < acc
        noisy = np.where(hit, yi, rng.integers(0, 3, n))
        z = 2.0 * np.eye(3)[noisy] + rng.normal(0, 0.8, (n, 3))
        e = np.exp(z - z.max(1, keepdims=True))
        probs[name] = e / e.sum(1, keepdims=True)
    return Aligned(ids=np.arange(start, start + n).astype(str), y=y,
                   n_words=rng.integers(20, 400, n),
                   groups=(np.arange(n) // 3).astype(str),
                   dates=pd.date_range("2025-07-01", periods=n,
                                       freq="10min").to_numpy(),
                   probs=probs)


def test_build_scenarios_stores_training_matrix_medians():
    val, test = _aligned(600, 0, 0), _aligned(200, 1, 10_000)
    sc, _ = build_scenarios(val, {"test": test}, MODELS, teams=TEAMS3,
                            kind="logreg", C_grid=(1.0,), n_splits=3)
    Pv = [val.probs[m] for m in MODELS]
    expected = {
        "gate_on_primary": (Pv, MODELS),
        "stack_gate": ([sc["stack_alone"]["val_probs"]] + Pv,
                       ["stack"] + MODELS),
    }
    for name, (mats, model_names) in expected.items():
        s = sc[name]
        Xv = G.prob_features(s["val_chosen"], mats, val.n_words, 3)
        ref = s["gate_reference"]
        np.testing.assert_array_equal(ref, np.median(Xv, axis=0))
        assert len(ref) == len(G.prob_feature_names(model_names, TEAMS3))
    assert "gate_reference" not in sc["primary_alone"]
    assert "gate_reference" not in sc["stack_alone"]


def test_make_bundle_reference_values_optional_and_aligned():
    names = G.feature_names(TEAMS3, ["s1"])
    plain = G.make_bundle(None, "logreg", TEAMS3, "p", ["s1"], 0.5)
    assert "gate_reference_values" not in plain
    ref = np.arange(len(names), dtype=np.float64) / 2
    b = G.make_bundle(None, "logreg", TEAMS3, "p", ["s1"], 0.5,
                      reference_values=ref)
    assert b["gate_reference_values"] == ref.tolist()
    assert all(type(v) is float for v in b["gate_reference_values"])
    assert {k: v for k, v in b.items() if k != "gate_reference_values"} \
        == plain
    with pytest.raises(ValueError):
        G.make_bundle(None, "logreg", TEAMS3, "p", ["s1"], 0.5,
                      reference_values=ref[:-1])
