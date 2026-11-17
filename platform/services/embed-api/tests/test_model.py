import pytest

from embed_api.model import ModelBundle, load_bundle


def test_load_bundle_reads_artifact(app_env):
    bundle = load_bundle(app_env)
    assert isinstance(bundle, ModelBundle)
    assert bundle.model_name == "improved_model_2"
    assert bundle.version.startswith("sha256:")
    assert len(bundle.version) == len("sha256:") + 64


def test_predict_returns_contract_shaped_dict(app_env):
    out = load_bundle(app_env).predict(
        "the collector calls about a debt from a collection agency")
    assert out["team"] == "T02"
    assert 0.0 <= out["team_confidence"] <= 1.0
    assert out["top_issues"] == []
    assert isinstance(out["auto_route"], bool)


def test_predict_rejects_empty_text(app_env):
    with pytest.raises(ValueError):
        load_bundle(app_env).predict("   ")


def test_full_probs_and_knn_vote(app_env, knn_index_path):
    out = load_bundle(app_env, knn_index_path=knn_index_path).predict(
        "the collector calls about a debt from a collection agency")
    probs = out["team_probs"]
    assert len(probs) == 11 and abs(sum(probs.values()) - 1.0) < 1e-6
    assert max(probs, key=probs.get) == out["team"] == "T02"
    vote = out["extra_team_probs"]["knn_embed"]
    assert abs(sum(vote.values()) - 1.0) < 1e-6
    assert max(vote, key=vote.get) == "T02"


def test_knn_index_is_optional_and_checks_encoder(app_env, knn_index_path,
                                                  tmp_path):
    bundle = load_bundle(app_env, knn_index_path=tmp_path / "none.npz")
    assert bundle.extra_models == ()
    assert "extra_team_probs" not in bundle.predict("mortgage escrow home " * 3)

    import joblib
    raw = joblib.load(app_env)
    raw["encoder_name"] = "another-encoder"
    other = tmp_path / "other.joblib"
    joblib.dump(raw, other)
    with pytest.raises(RuntimeError, match="rebuild the index"):
        load_bundle(other, knn_index_path=knn_index_path)
