"""Smoke tests for GET /explain/{request_id} (Req 15.8, 17.9).

Full coverage (byte-for-byte token forwarding, caplog checks) is task 26.2.
"""
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from router.main import create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PRIMARY_URL", "http://svm-api.test")
    monkeypatch.setenv("ENSEMBLE_API_URL", "http://ensemble-api.test")
    with TestClient(create_app()) as c:
        yield c


@respx.mock
def test_explain_forwards_token_and_passes_status_through(client):
    captured: dict = {}

    def _record(request):
        captured["token"] = request.headers.get("x-claricase-agent-token")
        return httpx.Response(202, json={"request_id": "rid-1", "explanation_status": "pending"})

    respx.get("http://ensemble-api.test/explain/rid-1").mock(side_effect=_record)
    resp = client.get("/explain/rid-1", headers={"X-ClariCase-Agent-Token": "s3cret"})
    assert captured["token"] == "s3cret"
    assert resp.status_code == 202
    assert resp.json() == {"request_id": "rid-1", "explanation_status": "pending"}


@respx.mock
def test_explain_unreachable_returns_503(client):
    respx.get("http://ensemble-api.test/explain/rid-2").mock(
        side_effect=httpx.ConnectError("down")
    )
    resp = client.get("/explain/rid-2")
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "explanation_service_unavailable"
