"""Secrets (EXPLAIN_API_KEY / EXPLAIN_AGENT_TOKEN) never leak into logs,
responses, stored records or audit entries."""
import json
import logging
import time

import httpx
import respx
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings, strategies as st

from tests.test_routes import EMB, SVM, TEXT, _mock

SECRET = st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                 min_size=16, max_size=40)
HEADER = "X-ClariCase-Agent-Token"


# Feature: model-explainability, Property 35: Secrets never appear in logs or responses
# **Validates: Requirements 15.8, 19.4**
@settings(max_examples=100, deadline=None,
          suppress_health_check=list(HealthCheck))
@given(api_key=SECRET, token=SECRET)
def test_secrets_never_appear_in_logs_or_responses(stack_path, monkeypatch,
                                                   caplog, capsys,
                                                   api_key, token):
    caplog.clear()
    capsys.readouterr()
    monkeypatch.setenv("STACK_PATH", str(stack_path))
    monkeypatch.setenv("SVM_API_URL", SVM)
    monkeypatch.setenv("EMBED_API_URL", EMB)
    monkeypatch.setenv("EXPLAIN_BACKEND", "mock")
    monkeypatch.setenv("EXPLAIN_API_KEY", api_key)
    monkeypatch.setenv("EXPLAIN_AGENT_TOKEN", token)
    from ensemble_api.main import create_app

    texts = []
    with caplog.at_level(logging.DEBUG), respx.mock:
        _mock()
        respx.get(f"{SVM}/healthz").mock(return_value=httpx.Response(
            200, json={"status": "ok", "extra_models": []}))
        respx.get(f"{EMB}/healthz").mock(return_value=httpx.Response(
            200, json={"status": "ok"}))
        respx.post(f"{SVM}/explain").mock(return_value=httpx.Response(500))
        respx.post(f"{EMB}/explain").mock(
            side_effect=httpx.ConnectError("down"))
        with TestClient(create_app()) as c:
            assert c.app.state.explain_cfg.api_key == api_key
            r = c.post("/predict", json={"text": TEXT, "request_id": "rid-s"})
            texts.append(r.text)
            store = c.app.state.explain_store
            for _ in range(50):
                e = store.get("rid-s")
                if e is not None and e.status != "pending":
                    break
                time.sleep(0.1)
            for hdrs in ({}, {HEADER: token}):
                g = c.get("/explain/rid-s", headers=hdrs)
                texts.append(g.text)
            texts.append(c.get("/healthz").text)
            e = store.get("rid-s")
            if e is not None:
                texts.append(repr(e))
                texts.append(json.dumps([a.model_dump(mode="json")
                                         for a in e.audit]))

    out = capsys.readouterr()
    texts += [caplog.text, out.out, out.err]
    for t in texts:
        assert api_key not in t
        assert token not in t
