"""Decision's explainer fields (Req 7.4, 8.5, 9.4) are filled by decide()
and leave the routing outcome unchanged."""
import numpy as np
import pytest

from src.ensemble import gate as G
from src.ensemble.stack import stack_features, stacker_proba
from src.ensemble.thresholds import route_mask

from ensemble_api.stack_model import load_stack

from tests.conftest import MODELS, T, _synthetic


def _reference(bundle, model_probs, n_words):
    """The pre-change decide() logic, recomputed independently."""
    raw = bundle.raw
    mats = [bundle._vec(model_probs[m])[None, :] for m in bundle.models]
    P_raw = stacker_proba(raw["stacker"], stack_features(mats), T)[0]
    bundle.prior.observe(P_raw)
    P = bundle.prior.adjust(P_raw)
    j = int(P.argmax())
    X = G.prob_features([j], [P[None, :]] + mats, [n_words], T)
    score = float(raw["gate"].predict_proba(X)[0, 1])
    auto = bool(route_mask(np.array([score]), np.array([bundle.teams[j]]),
                           bundle.base_threshold, bundle.team_thresholds)[0])
    return bundle.teams[j], score, auto, P_raw, X[0]


def test_decide_unchanged_and_fields_filled(stack_path):
    served = load_stack(stack_path, min_rows=40, refresh=10)
    twin = load_stack(stack_path, min_rows=40, refresh=10)
    _, mats, n_words = _synthetic(80, 7)
    for i in range(80):   # crosses min_rows, so label shift activates
        mp = {m: dict(zip(served.teams, mats[k][i]))
              for k, m in enumerate(MODELS)}
        d = served.decide(mp, int(n_words[i]))
        team, score, auto, P_raw, x = _reference(twin, mp, int(n_words[i]))
        assert (d.team, d.auto_route) == (team, auto)
        assert d.score == pytest.approx(score, abs=0)
        np.testing.assert_array_equal(d.stack_raw_probs, P_raw)
        np.testing.assert_array_equal(d.gate_features, x)
        assert d.prior_n_seen == i + 1 and d.prior_min_rows == 40
    assert served.prior.active


def test_fallback_leaves_fields_empty(stack_path):
    d = load_stack(stack_path).fallback({"T01": 0.9})
    assert d.stack_raw_probs is None and d.gate_features is None
    assert d.prior_n_seen is None and d.prior_min_rows is None
