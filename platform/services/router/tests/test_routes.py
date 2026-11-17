import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from router.main import create_app


PRIMARY_RESPONSE = {
    "model_name": "improved_model_1",
    "model_version": "sha256:abc",
    "team": "T01",
    "team_confidence": 0.82,
    "top_issues": [{"issue": "I001", "prob": 0.55}],
    "auto_route": True,
    "request_id": "server-generated",
    "latency_ms": 11,
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PRIMARY_URL", "http://svm-api.test")
    app = create_app()
    with TestClient(app) as c:
        yield c


@respx.mock
def test_predict_proxies_to_primary_and_returns_response(client):
    route = respx.post("http://svm-api.test/predict").mock(
        return_value=httpx.Response(200, json=PRIMARY_RESPONSE)
    )
    resp = client.post(
        "/predict",
        json={"text": "I have a long enough complaint text to pass validation."},
    )
    assert route.called
    assert resp.status_code == 200
    body = resp.json()
    assert body["team"] == "T01"


@respx.mock
def test_predict_generates_request_id_when_absent(client):
    captured: dict = {}

    def _record(request):
        captured["body"] = request.read().decode()
        return httpx.Response(200, json=PRIMARY_RESPONSE)

    respx.post("http://svm-api.test/predict").mock(side_effect=_record)
    resp = client.post(
        "/predict",
        json={"text": "I have a long enough complaint text to pass validation."},
    )
    assert resp.status_code == 200
    assert "request_id" in captured["body"]


@respx.mock
def test_predict_preserves_client_request_id(client):
    captured: dict = {}

    def _record(request):
        captured["body"] = request.read().decode()
        return httpx.Response(200, json=PRIMARY_RESPONSE)

    respx.post("http://svm-api.test/predict").mock(side_effect=_record)
    resp = client.post(
        "/predict",
        json={
            "text": "I have a long enough complaint text to pass validation.",
            "request_id": "rid-42",
        },
    )
    assert resp.status_code == 200
    assert "rid-42" in captured["body"]


@respx.mock
def test_primary_5xx_maps_to_503(client):
    respx.post("http://svm-api.test/predict").mock(
        return_value=httpx.Response(500, text="boom")
    )
    resp = client.post(
        "/predict",
        json={"text": "I have a long enough complaint text to pass validation."},
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "primary_unhealthy"


@respx.mock
def test_primary_timeout_maps_to_503(client):
    respx.post("http://svm-api.test/predict").mock(
        side_effect=httpx.TimeoutException("slow")
    )
    resp = client.post(
        "/predict",
        json={"text": "I have a long enough complaint text to pass validation."},
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "primary_unhealthy"


def test_metrics_endpoint_present(client):
    client.get("/healthz")  # ignore outcome
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "# HELP" in resp.text
