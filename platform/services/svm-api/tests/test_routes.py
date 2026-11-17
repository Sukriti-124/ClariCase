from fastapi.testclient import TestClient

from svm_api.main import create_app


def test_healthz_is_200_after_model_loaded(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_predict_returns_contract_shape(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            json={"text": "I disputed a charge on my credit card and never got a response from the bank."},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model_name"] == "improved_model_1"
        assert body["model_version"].startswith("sha256:")
        assert isinstance(body["team"], str)
        assert 0.0 <= body["team_confidence"] <= 1.0
        assert isinstance(body["top_issues"], list)
        assert isinstance(body["auto_route"], bool)
        assert body["request_id"]
        assert body["latency_ms"] >= 0


def test_predict_rejects_short_text(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    app = create_app()
    with TestClient(app) as client:
        resp = client.post("/predict", json={"text": "short"})
        assert resp.status_code == 422


def test_predict_preserves_request_id(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            json={
                "text": "The credit bureau refuses to remove an incorrect collections account from my report.",
                "request_id": "rid-xyz-1",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["request_id"] == "rid-xyz-1"


def test_metrics_endpoint_present(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    app = create_app()
    with TestClient(app) as client:
        client.get("/healthz")
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "http_requests_total" in resp.text or "http_request_duration" in resp.text


def test_issues_endpoint(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    app = create_app()
    with TestClient(app) as client:
        resp = client.post("/issues", json={
            "text": "The debt collector keeps calling my workplace about a debt I do not owe.",
            "team": "T02", "top_k": 2})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["team"] == "T02" and len(body["top_issues"]) <= 2
        assert all(r["issue"].startswith("T02_") for r in body["top_issues"])
        assert client.post("/issues", json={"text": "x" * 20, "team": "T99"}
                           ).status_code == 422


def test_explain_smoke(model_path, monkeypatch):
    """Smoke check for POST /explain; full route tests live in test_explain.py."""
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    b2 = model_path.parents[1] / "baseline_model_2" / "model.joblib"
    monkeypatch.setenv("BASELINE2_MODEL_PATH", str(b2))
    text = "I disputed a charge on my credit card and never got a response from the bank."
    app = create_app()
    with TestClient(app) as client:
        team = client.post("/predict", json={"text": text}).json()["team"]
        resp = client.post("/explain", json={
            "text": text, "masked_text": text, "chosen_team": team,
            "request_id": "rid-explain-1"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert set(body["sections"]) == {"improved_model_1", "baseline_model_2"}
        im1 = body["sections"]["improved_model_1"]
        assert im1["status"] == "ok", im1.get("reason")
        assert body["model_versions"]["improved_model_1"].startswith("sha256:")
        bm2 = body["sections"]["baseline_model_2"]
        if b2.exists():
            assert bm2["status"] == "ok", bm2.get("reason")
            assert body["model_versions"]["baseline_model_2"].startswith("sha256:")
        else:
            assert bm2["status"] == "skipped"
