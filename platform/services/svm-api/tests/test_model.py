import pytest

from svm_api.model import ModelBundle, load_bundle


SAMPLE_TEXT = (
    "I applied for a credit card and the bank reported late payments to the "
    "credit bureaus even though I paid on time every month for the last year."
)


def test_load_bundle_reads_artifact(model_path):
    bundle = load_bundle(model_path)
    assert isinstance(bundle, ModelBundle)
    assert bundle.model_name == "improved_model_1"
    assert bundle.version.startswith("sha256:")
    assert len(bundle.version) == len("sha256:") + 64


def test_predict_returns_contract_shaped_dict(model_path):
    bundle = load_bundle(model_path)
    out = bundle.predict(SAMPLE_TEXT, top_k=3)
    assert isinstance(out["team"], str)
    assert 0.0 <= out["team_confidence"] <= 1.0
    assert isinstance(out["top_issues"], list)
    assert len(out["top_issues"]) <= 3
    for row in out["top_issues"]:
        assert set(row.keys()) == {"issue", "prob"}
        assert 0.0 <= row["prob"] <= 1.0
    assert isinstance(out["auto_route"], bool)


def test_predict_rejects_empty_text(model_path):
    bundle = load_bundle(model_path)
    with pytest.raises(ValueError):
        bundle.predict("", top_k=3)


def test_predict_matches_upstream_and_has_full_probs(model_path):
    from src.improved_model_1 import predict as upstream
    bundle = load_bundle(model_path)
    out = bundle.predict(SAMPLE_TEXT, top_k=3)
    ref = upstream.predict(SAMPLE_TEXT, top_k=3, model=bundle.raw)
    assert out["team"] == ref["predicted_team_id"]
    assert out["team_confidence"] == pytest.approx(ref["confidence"])
    assert [r["issue"] for r in out["top_issues"]] == \
        [r["issue_id"] for r in ref["issue_top_k"]]
    assert out["auto_route"] == (ref["route"] == "auto")
    probs = out["team_probs"]
    assert len(probs) == 11 and abs(sum(probs.values()) - 1.0) < 1e-6
    assert max(probs, key=probs.get) == out["team"]


def test_issues_for_another_team_stay_in_that_team(model_path):
    bundle = load_bundle(model_path)
    rows = bundle.issues(SAMPLE_TEXT, "T02", top_k=3)
    assert rows and all(r["issue"].startswith("T02_") for r in rows)
    with pytest.raises(KeyError):
        bundle.issues(SAMPLE_TEXT, "T99")


def test_baseline2_is_optional(model_path, tmp_path):
    bundle = load_bundle(model_path, baseline2_path=tmp_path / "missing.joblib")
    assert bundle.extra_models == ()
    assert "extra_team_probs" not in bundle.predict(SAMPLE_TEXT)
