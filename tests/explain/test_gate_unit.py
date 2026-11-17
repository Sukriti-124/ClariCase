"""Unit tests for the Gate_Explainer (design §7, Req 8)."""
from __future__ import annotations

import numpy as np
import pytest

from src.ensemble.gate import prob_feature_names
from src.explain.explainers import gate as GE
from tests.explain.fixtures import (MODELS, TEAMS, fake_shap, fake_shap_wrong,
                                    synthetic_stack_bundle)

GROUP_MODELS = ["stack"] + MODELS


def _call(sb, row=0, shap_module=None, reference="bundle", threshold="bundle",
          kind=None):
    b = sb.bundle
    x = sb.gate_X[row]
    score = float(b["gate"].predict_proba(x[None, :])[0, 1])
    ref = b.get("gate_reference_values") if reference == "bundle" else reference
    thr = b["base_threshold"] if threshold == "bundle" else threshold
    return GE.explain_gate(b["gate"], kind or b["gate_kind"], b["gate_feature_names"],
                           x, ref, GROUP_MODELS, score, thr,
                           shap_module=shap_module), score, thr


def _check_common(sec, score, thr):
    d = sec.details
    assert d.score == pytest.approx(score)
    assert d.threshold == thr
    assert d.score_ge_threshold == (score >= thr)
    assert sec.technique and sec.method and d.unit and d.gate_kind


def test_feature_groups_cover_every_feature_once():
    names = prob_feature_names(GROUP_MODELS, TEAMS)
    groups = GE.feature_groups(names, GROUP_MODELS)
    flat = sorted(i for idx in groups.values() for i in idx)
    assert flat == list(range(len(names)))
    assert set(groups) == {f"model:{m}" for m in GROUP_MODELS} | {"chosen_team",
                                                                 "complaint_length"}
    assert len(groups["chosen_team"]) == len(TEAMS)


def test_feature_groups_reject_unknown_feature():
    with pytest.raises(ValueError, match="no feature group"):
        GE.feature_groups(["stack_top", "mystery"], ["stack"])


def test_feature_groups_reject_double_membership():
    # With a model named "chosen_is", "chosen_is_top" is both a model
    # feature and a chosen-team flag.
    with pytest.raises(ValueError, match="several feature groups"):
        GE.feature_groups(["chosen_is_top"], ["chosen_is"])


def test_logreg_linear_exact():
    sb = synthetic_stack_bundle("logreg")
    sec, score, thr = _call(sb, row=3)
    assert sec.status == "ok"
    assert (sec.technique, sec.method, sec.details.unit) == ("linear", "exact", "log-odds")
    x = sb.gate_X[3][None, :]
    margin = float(sb.bundle["gate"].decision_function(x)[0])
    total = sum(d.contribution for d in sec.drivers) + sec.details.constant
    assert total == pytest.approx(margin, abs=1e-6)
    assert len(sec.drivers) == len(GROUP_MODELS) + 2
    keys = [(-abs(d.contribution), d.label) for d in sec.drivers]
    assert keys == sorted(keys)
    _check_common(sec, score, thr)


def test_logreg_ignores_shap_and_reference():
    sb = synthetic_stack_bundle("logreg", with_reference=False)
    sec, _, _ = _call(sb, shap_module=fake_shap)
    assert sec.status == "ok" and sec.technique == "linear"


def test_logreg_perturbed_coef_is_approximate():
    sb = synthetic_stack_bundle("logreg")
    est = sb.bundle["gate"]

    # A wrapper whose margin differs from the linear sum by 0.1.
    class Shifted:
        def __init__(self, inner):
            self.inner = inner

        def __getitem__(self, k):
            return self.inner[k]

        def decision_function(self, X):
            return self.inner.decision_function(X) + 0.1

    x = sb.gate_X[0]
    sec = GE.explain_gate(Shifted(est), "logreg", sb.bundle["gate_feature_names"], x,
                          None, GROUP_MODELS, 0.5, 0.4)
    assert sec.status == "ok" and sec.method == "approximate"
    assert sec.details.residual == pytest.approx(0.1, abs=1e-9)
    assert sec.drivers


def test_hgb_with_fake_shap_is_exact():
    sb = synthetic_stack_bundle("hgb")
    sec, score, thr = _call(sb, row=1, shap_module=fake_shap)
    assert sec.status == "ok"
    assert (sec.technique, sec.method, sec.details.unit) == ("shap", "exact", "log-odds")
    x = sb.gate_X[1][None, :]
    margin = float(sb.bundle["gate"].decision_function(x)[0])
    total = sum(d.contribution for d in sec.drivers) + sec.details.constant
    assert total == pytest.approx(margin, abs=1e-6)
    _check_common(sec, score, thr)


def test_hgb_incompatible_shap_falls_back_to_ablation():
    sb = synthetic_stack_bundle("hgb")
    sec, score, thr = _call(sb, row=2, shap_module=fake_shap_wrong)
    assert sec.status == "ok"
    assert (sec.technique, sec.method, sec.details.unit) == (
        "ablation", "approximate", "change in gate score")
    est, x = sb.bundle["gate"], sb.gate_X[2][None, :]
    ref = np.asarray(sb.bundle["gate_reference_values"])
    groups = GE.feature_groups(sb.bundle["gate_feature_names"], GROUP_MODELS)
    by_label = {d.label: d.contribution for d in sec.drivers}
    for g, idx in groups.items():
        xr = x.copy()
        xr[0, idx] = ref[idx]
        want = est.predict_proba(x)[0, 1] - est.predict_proba(xr)[0, 1]
        assert by_label[g] == want
    _check_common(sec, score, thr)


def test_hgb_without_reference_is_skipped():
    sb = synthetic_stack_bundle("hgb", with_reference=False)
    sec, score, thr = _call(sb, shap_module=fake_shap_wrong)
    assert sec.status == "skipped"
    assert "gate_reference_values" in sec.reason
    assert sec.drivers == []
    _check_common(sec, score, thr)


def test_error_keeps_gate_fields():
    sb = synthetic_stack_bundle("logreg")
    b = sb.bundle
    sec = GE.explain_gate(b["gate"], "logreg", b["gate_feature_names"],
                          sb.gate_X[0][:5], None, GROUP_MODELS, 0.7, 0.6)
    assert sec.status == "error" and sec.reason.startswith("ValueError")
    assert sec.drivers == []
    d = sec.details
    assert (d.gate_kind, d.score, d.threshold, d.score_ge_threshold) == (
        "logreg", 0.7, 0.6, True)
    assert sec.technique == "linear" and d.unit == "log-odds"


def test_missing_threshold_is_absent():
    sb = synthetic_stack_bundle("logreg")
    sec, _, _ = _call(sb, threshold=None)
    assert sec.status == "ok"
    assert sec.details.threshold.absent is True
    assert sec.details.score_ge_threshold is False
