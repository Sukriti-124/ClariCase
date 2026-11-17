"""Forwarding tests for the router's GET /explain/{request_id} (Req 15.8, 17.9).

ensemble-api is mocked with respx, as in test_explain_smoke.py.
"""
import logging

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from router.main import create_app

ENSEMBLE = "http://ensemble-api.test"
HEADER = "X-ClariCase-Agent-Token"

# Visible ASCII (0x21-0x7E): every value a client can send as an HTTP header
# without the server trimming or re-encoding it.
_tokens = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E), min_size=1, max_size=64
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PRIMARY_URL", "http://svm-api.test")
    monkeypatch.setenv("ENSEMBLE_API_URL", ENSEMBLE)
    with TestClient(create_app()) as c:
        yield c


def _raw_header(request: httpx.Request, name: str) -> list[bytes]:
    return [v for k, v in request.headers.raw if k.lower() == name.lower().encode()]


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(token=_tokens)
def test_token_header_forwarded_byte_for_byte(client, token):
    captured: list[httpx.Request] = []

    def _record(request):
        captured.append(request)
        return httpx.Response(200, json={"request_id": "rid", "explanation_status": "ready"})

    with respx.mock:
        respx.get(f"{ENSEMBLE}/explain/rid").mock(side_effect=_record)
        resp = client.get("/explain/rid", headers={HEADER: token})

    assert resp.status_code == 200
    assert len(captured) == 1
    assert _raw_header(captured[0], HEADER) == [token.encode("ascii")]


@respx.mock
def test_no_token_header_when_client_sends_none(client):
    captured: list[httpx.Request] = []

    def _record(request):
        captured.append(request)
        return httpx.Response(200, json={})

    respx.get(f"{ENSEMBLE}/explain/rid").mock(side_effect=_record)
    client.get("/explain/rid")
    assert _raw_header(captured[0], HEADER) == []


@pytest.mark.parametrize(
    "status, body",
    [
        (
            200,
            b'{"request_id":"rid-x","explanation_status":"ready","explanation":{"consumer_summary":"ok"}}',
        ),
        (202, b'{"request_id":"rid-x","explanation_status":"pending"}'),
        (404, b'{"error_code":"explanation_not_found","request_id":"rid-x"}'),
    ],
)
@respx.mock
def test_status_and_body_pass_through_unchanged(client, status, body):
    respx.get(f"{ENSEMBLE}/explain/rid-x").mock(
        return_value=httpx.Response(
            status, content=body, headers={"content-type": "application/json"}
        )
    )
    resp = client.get("/explain/rid-x", headers={HEADER: "tok"})
    assert resp.status_code == status
    assert resp.content == body
    assert resp.headers["content-type"].startswith("application/json")


@respx.mock
def test_connect_error_gives_503(client):
    respx.get(f"{ENSEMBLE}/explain/rid-down").mock(side_effect=httpx.ConnectError("down"))
    resp = client.get("/explain/rid-down", headers={HEADER: "tok"})
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "explanation_service_unavailable"


@respx.mock
def test_token_never_appears_in_logs(client, caplog):
    token = "agent-token-7f3c9e1b-SHOULD-NOT-LEAK"
    respx.get(f"{ENSEMBLE}/explain/rid-ok").mock(
        return_value=httpx.Response(200, json={"request_id": "rid-ok", "explanation_status": "ready"})
    )
    respx.get(f"{ENSEMBLE}/explain/rid-404").mock(
        return_value=httpx.Response(404, json={"error_code": "explanation_not_found"})
    )
    respx.get(f"{ENSEMBLE}/explain/rid-down").mock(side_effect=httpx.ConnectError("down"))

    with caplog.at_level(logging.DEBUG):
        responses = [
            client.get(f"/explain/{rid}", headers={HEADER: token})
            for rid in ("rid-ok", "rid-404", "rid-down")
        ]

    assert [r.status_code for r in responses] == [200, 404, 503]
    assert token not in caplog.text
    for record in caplog.records:
        assert token not in record.getMessage()
        assert token not in repr(record.__dict__)
    # The token is not echoed back to the caller either.
    for r in responses:
        assert token.encode() not in r.content
