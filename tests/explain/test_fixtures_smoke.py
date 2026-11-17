"""Smoke tests: the synthetic fixtures build quickly and have the promised shape."""
from __future__ import annotations

import random
import time

import numpy as np
import pytest
from hypothesis import given, settings

from tests.explain import fixtures as F


def test_fixtures_build_quickly():
    F._build_im1.cache_clear()
    F._build_bm2.cache_clear()
    F._build_stack.cache_clear()
    t0 = time.perf_counter()
    F.synthetic_im1()
    F.synthetic_im1(compact=True)
    F.synthetic_bm2()
    F.synthetic_stack_bundle("logreg")
    F.synthetic_stack_bundle("hgb", with_reference=False)
    assert time.perf_counter() - t0 < 30.0


def test_im1_heads_and_calibrator():
    from src.improved_model_1.model import _ConstantIssue

    m = F.synthetic_im1()
    assert list(m.hsvm.team_classes_) == F.TEAMS
    heads = m.hsvm.issue_clfs_
    assert isinstance(heads[F.CONSTANT_TEAM], _ConstantIssue)
    assert heads[F.BINARY_TEAM].coef_.shape[0] == 1
    assert any(not isinstance(h, _ConstantIssue) and h.coef_.shape[0] > 2
               for h in heads.values())
    cal = m.hsvm.calibrated_team_
    assert cal.calibrated_classifiers_[0].estimator.estimator is m.hsvm.team_clf
    names = m.vectorizer.get_feature_names_out()
    assert any(n.startswith("word__") for n in names)
    assert any(n.startswith("char__") for n in names)
    X = m.vectorizer.transform(m.texts[:22])
    acc = (m.hsvm.predict_team(X) == np.asarray(m.teams[:22])).mean()
    assert acc > 0.8
    assert F.synthetic_im1(compact=True).hsvm.team_clf.coef_.dtype == np.float32


def test_fixture_copies_are_independent():
    a = F.synthetic_im1()
    a.hsvm.team_clf.coef_[0, 0] += 1.0
    b = F.synthetic_im1()
    assert b.hsvm.team_clf.coef_[0, 0] != a.hsvm.team_clf.coef_[0, 0]


def test_bm2_wrapping():
    m = F.synthetic_bm2()
    assert m.calibrated.calibrated_classifiers_[0].estimator.estimator is m.pipeline
    assert list(m.calibrated.classes_) == F.TEAMS
    p = m.calibrated.predict_proba(m.texts[:3])
    assert p.shape == (3, 11)


def test_fake_encoder():
    enc = F.FakeEncoder()
    E = enc.encode(["a b c", "a b c", ""])
    assert E.shape == (3, 16)
    np.testing.assert_allclose(np.linalg.norm(E[:2], axis=1), 1.0)
    np.testing.assert_array_equal(E[0], E[1])
    assert not E[2].any()
    assert enc.n_encoded == 3 and enc.n_calls == 1
    bad = F.FakeEncoder(raise_on_call=True)
    with pytest.raises(RuntimeError):
        bad.encode(["x"])
    assert bad.n_encoded == 0


@pytest.mark.parametrize("kind", ["logreg", "hgb"])
@pytest.mark.parametrize("with_reference", [True, False])
def test_stack_bundle(kind, with_reference):
    s = F.synthetic_stack_bundle(kind, with_reference)
    b = s.bundle
    assert b["gate_kind"] == kind and b["teams"] == F.TEAMS
    assert len(b["gate_feature_names"]) == s.gate_X.shape[1]
    assert ("gate_reference_values" in b) == with_reference
    if with_reference:
        assert len(b["gate_reference_values"]) == len(b["gate_feature_names"])
    assert s.stack_probs.shape == (len(s.y), 11)
    assert b["stacker"] is not None
    assert 0.0 <= b["gate"].predict_proba(s.gate_X[:1])[0, 1] <= 1.0


def test_fake_shap_exact_and_wrong():
    s = F.synthetic_stack_bundle("hgb")
    est, X = s.bundle["gate"], s.gate_X[:3]
    good = F.fake_shap.TreeExplainer(est, model_output="raw")
    vals = good.shap_values(X)
    assert vals.shape == X.shape
    np.testing.assert_allclose(vals.sum(1) + good.expected_value,
                               est.decision_function(X), atol=1e-9)
    wrong = F.fake_shap_wrong.TreeExplainer(est, model_output="raw")
    gap = np.abs(wrong.shap_values(X).sum(1) + wrong.expected_value
                 - est.decision_function(X))
    assert (gap > 1e-6).all()


def test_sentence_generators_have_pii_and_oov():
    rng = random.Random(3)
    texts = [F.make_text(rng, "T03", pii_rate=1.0, oov_rate=1.0)
             for _ in range(20)]
    joined = " ".join(texts)
    assert "zq" in joined
    assert any(ch.isdigit() for ch in joined)
    toks = F.pii_tokens(rng)
    assert set(toks) == {"phone", "card", "ssn", "email", "digit_run"}
    assert "@" in toks["email"] and len(toks["digit_run"]) >= 10


@settings(max_examples=20, deadline=None)
@given(F.st_text)
def test_text_strategy_produces_sentences(text):
    assert text.strip()
    assert text.rstrip()[-1] in ".!?"
