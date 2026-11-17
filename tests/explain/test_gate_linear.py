"""Property test: gate linear exactness and feature grouping (design §7, Req 8)."""
from __future__ import annotations

import copy
from functools import lru_cache

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.ensemble.gate import prob_feature_names
from src.explain.explainers import gate as GE
from tests.explain.fixtures import MODELS, synthetic_stack_bundle

GROUP_MODELS = ["stack"] + MODELS

# Model / team identifiers. A model name starting with "chosen_is" would make
# "<m>_top" also a chosen-team flag, which feature_groups rejects by design.
_ident = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
                 min_size=1, max_size=12)
_model_names = st.lists(_ident.filter(lambda s: not s.startswith("chosen_is")),
                        min_size=1, max_size=6, unique=True)
_team_names = st.lists(_ident, min_size=0, max_size=12, unique=True)


@lru_cache(maxsize=1)
def _bundle():
    return synthetic_stack_bundle("logreg")


# Feature: model-explainability, Property 16: Gate linear exactness and grouping
@settings(max_examples=100, deadline=None)
@given(models=_model_names, teams=_team_names)
def test_feature_groups_partition_prob_feature_names(models, teams):
    """**Validates: Requirements 8.4**"""
    names = prob_feature_names(models, teams)
    groups = GE.feature_groups(names, models)
    flat = sorted(i for idx in groups.values() for i in idx)
    assert flat == list(range(len(names)))          # every name exactly once
    for m in models:
        assert sorted(names[i] for i in groups[f"model:{m}"]) == sorted(
            f"{m}_{f}" for f in ("p_chosen", "top", "margin", "agree", "entropy"))
    assert [names[i] for i in groups.get("chosen_team", [])] == [
        f"chosen_is_{t}" for t in teams]
    assert [names[i] for i in groups["complaint_length"]] == ["log_n_words"]


# Feature: model-explainability, Property 16: Gate linear exactness and grouping
@settings(max_examples=100, deadline=None)
@given(row=st.integers(min_value=0, max_value=599),
       noise=st.floats(0.0, 2.0, allow_nan=False),
       seed=st.integers(0, 2**32 - 1),
       coef_scale=st.floats(0.1, 5.0, allow_nan=False),
       intercept=st.floats(-3.0, 3.0, allow_nan=False))
def test_logreg_gate_linear_exactness(row, noise, seed, coef_scale, intercept):
    """**Validates: Requirements 8.3, 8.4, 8.6**"""
    sb = _bundle()
    b = sb.bundle
    gate = copy.deepcopy(b["gate"])
    names = b["gate_feature_names"]
    rng = np.random.default_rng(seed)

    # A different logreg gate per example: rescaled, sign-jittered coefficients
    # and a new intercept.
    lr = gate[-1]
    lr.coef_ = lr.coef_ * coef_scale * rng.choice([-1.0, 1.0], size=lr.coef_.shape)
    lr.intercept_ = np.array([intercept])

    x = sb.gate_X[row] + noise * rng.standard_normal(len(names))
    score = float(gate.predict_proba(x[None, :])[0, 1])
    thr = b["base_threshold"]
    sec = GE.explain_gate(gate, "logreg", names, x, None, GROUP_MODELS, score, thr)

    assert sec.status == "ok"
    assert (sec.technique, sec.method, sec.details.unit) == ("linear", "exact",
                                                             "log-odds")

    scaler = gate[0]
    per_feature = lr.coef_[0] * (x - scaler.mean_) / scaler.scale_
    groups = GE.feature_groups(names, GROUP_MODELS)
    by_label = {d.label: d.contribution for d in sec.drivers}
    assert set(by_label) == set(groups)
    for g, idx in groups.items():
        assert by_label[g] == pytest.approx(per_feature[idx].sum(), abs=1e-9)

    margin = float(gate.decision_function(x[None, :])[0])
    assert sec.details.constant == pytest.approx(intercept, abs=1e-12)
    total = sum(by_label.values()) + sec.details.constant
    assert abs(total - margin) <= 1e-6
