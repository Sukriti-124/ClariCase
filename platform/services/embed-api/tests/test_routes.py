from fastapi.testclient import TestClient

from embed_api.main import create_app

TEXT = "My mortgage servicer mishandled the escrow on my home loan."


def test_healthz_is_200_after_model_loaded(app_env):
    with TestClient(create_app()) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_predict_returns_contract_shape(app_env):
    with TestClient(create_app()) as client:
        resp = client.post("/predict", json={"text": TEXT})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model_name"] == "improved_model_2"
        assert body["model_version"].startswith("sha256:")
        assert body["team"] == "T06"
        assert 0.0 <= body["team_confidence"] <= 1.0
        assert body["top_issues"] == []
        assert isinstance(body["auto_route"], bool)
        assert body["request_id"]
        assert body["latency_ms"] >= 0


def test_predict_rejects_short_text(app_env):
    with TestClient(create_app()) as client:
        assert client.post("/predict", json={"text": "short"}).status_code == 422


def test_predict_preserves_request_id(app_env):
    with TestClient(create_app()) as client:
        resp = client.post("/predict",
                           json={"text": TEXT, "request_id": "rid-xyz-1"})
        assert resp.json()["request_id"] == "rid-xyz-1"


def test_metrics_include_model_counters(app_env):
    with TestClient(create_app()) as client:
        client.post("/predict", json={"text": TEXT})
        text = client.get("/metrics").text
        assert "model_predictions_total" in text
        assert "model_confidence" in text


def test_response_carries_probs_and_vote(app_env):
    with TestClient(create_app()) as client:
        body = client.post("/predict", json={"text": TEXT}).json()
        assert set(body["team_probs"]) == {f"T{i:02d}" for i in range(1, 12)}
        assert "knn_embed" in body["extra_team_probs"]
        assert client.get("/healthz").json()["extra_models"] == ["knn_embed"]


def test_explain_smoke_returns_both_sections(app_env):
    from src.explain.record import SECTION_MODELS
    with TestClient(create_app()) as client:
        resp = client.post("/explain", json={
            "text": TEXT, "masked_text": TEXT, "chosen_team": "T06",
            "request_id": "rid-explain-1"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert set(body["sections"]) == {"improved_model_2", "knn_embed"}
        for name, sec in body["sections"].items():
            SECTION_MODELS[name].model_validate(sec)
            assert sec["status"] == "ok", sec
        assert body["model_versions"]["improved_model_2"].startswith("sha256:")
