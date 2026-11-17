"""Smoke tests for GET /explain/{request_id} and the /healthz narrator block."""
from dataclasses import replace

from src.explain.narration import Explanation, ValidationSummary
from src.explain.store import StoreEntry, _Slot

TOKEN = "s3cret-token"


def _cfg(client, token=TOKEN):
    client.app.state.explain_cfg = replace(client.app.state.explain_cfg,
                                           agent_token=token)


def _put(client, rid, status="pending", explanation=None):
    store = client.app.state.explain_store
    e = StoreEntry(status=status, explanation=explanation,
                   inserted_at=store._clock())
    store._data[rid] = _Slot(e)
    return e


def test_unknown_id_404(client):
    assert client.get("/explain/nope").status_code == 404


def test_pending_202_and_watchdog_failed(client):
    e = _put(client, "r1")
    r = client.get("/explain/r1")
    assert r.status_code == 202 and r.json()["explanation_status"] == "pending"
    e.inserted_at -= 121
    r = client.get("/explain/r1")
    assert r.status_code == 200
    assert r.json()["explanation_status"] == "failed"
    assert r.json()["reason"] == "explanation timed out after 120 s"


def test_ready_roles(client):
    _cfg(client)
    ex = Explanation(consumer_summary="c", agent_summary="a", source="template",
                     validation=ValidationSummary(accepted=True),
                     record_hash="h", created_at="t")
    _put(client, "r2", "ready", ex)
    r = client.get("/explain/r2")
    assert set(r.json()) == {"request_id", "explanation_status",
                             "consumer_summary"}
    bad = client.get("/explain/r2", headers={"X-ClariCase-Agent-Token": "x"})
    assert "agent_summary" not in bad.json()
    ok = client.get("/explain/r2", headers={"X-ClariCase-Agent-Token": TOKEN})
    assert ok.json()["agent_summary"] == "a" and "audit" in ok.json()
    assert TOKEN not in ok.text


def test_empty_token_never_agent(client):
    _cfg(client, "")
    _put(client, "r3")
    r = client.get("/explain/r3", headers={"X-ClariCase-Agent-Token": ""})
    assert "audit" not in r.json()
