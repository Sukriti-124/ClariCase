import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from issue_api.config import Settings
from issue_api.main import create_app

ROOT = Path(__file__).resolve().parents[4]
TEXT = "I paid my credit card bill on time, but the bank charged me a late fee."
REQUEST = {"text": TEXT, "team": "T03", "request_id": "example-1"}
GOOD = {"status": "classified", "issue_id": "T03_I004", "candidate_issue_ids": [],
        "evidence": ["the bank charged me a late fee"], "reason": "The dispute concerns a fee."}


def settings(**kw):
    return Settings(_env_file=None, catalog_dir=ROOT / "data/processed", backend="openai_compat",
                    model="test-model", api_key="test-only-secret", base_url="https://llm.test/v1", **kw)


def envelope(data, finish="stop"):
    return {"choices": [{"message": {"content": json.dumps(data)}, "finish_reason": finish}]}


def client_for(data=GOOD, **kw):
    def handle(request):
        return httpx.Response(200, json=envelope(data))
    return TestClient(create_app(settings(**kw), transport=httpx.MockTransport(handle)))


def test_full_original_catalog_and_product_filter():
    with client_for() as c:
        health = c.get("/healthz").json()
        assert (health["teams"], health["issues"]) == (11, 73)
        all_products = c.get("/catalog").json()["products"]
        assert len(all_products) == 11
        rows = c.get("/catalog?team=T01").json()["products"][0]["issues"]
        assert {"T01_I003", "T01_I001", "T01_I007", "T01_I008"} <= {r["issue_id"] for r in rows}
        assert c.get("/catalog?team=T99").status_code == 422


def test_success_and_no_fabricated_confidence():
    with client_for() as c:
        r = c.post("/classify-issue", json=REQUEST)
        assert r.status_code == 200, r.text
        result = r.json()
        assert result["issue_id"] == "T03_I004"
        assert result["issue_label"] == "Credit card :: Fees or interest"
        assert result["team"] == "T03" and result["request_id"] == "example-1"
        assert result["review_required"] is True
        assert "confidence" not in result
        assert "issue_classifications_total" in c.get("/metrics").text


@pytest.mark.parametrize("decision", [
    {**GOOD, "status": "ambiguous", "issue_id": None,
     "candidate_issue_ids": ["T03_I004", "T03_I009"]},
    {**GOOD, "status": "product_mismatch", "issue_id": None},
    {**GOOD, "status": "insufficient_context", "issue_id": None, "evidence": []},
])
def test_unresolved_is_not_forced_to_first_issue(decision):
    with client_for(decision) as c:
        r = c.post("/classify-issue", json=REQUEST)
        assert r.status_code == 200, r.text
        assert r.json()["issue_id"] is None
        assert r.json()["team"] == "T03" and r.json()["review_required"]


@pytest.mark.parametrize("change", [
    {"issue_id": "T02_I001"}, {"issue_id": "T03_I999"}, {"issue_id": None},
    {"evidence": ["a quote absent from the complaint"]}, {"evidence": []},
    {"confidence": .99}, {"reason": ""}, {"evidence": [" "]},
    {"candidate_issue_ids": ["T03_I009"]}, {"status": "guessed"},
    {"status": "ambiguous", "issue_id": None, "candidate_issue_ids": ["T03_I004"]},
    {"status": "ambiguous", "issue_id": None, "candidate_issue_ids": ["T03_I004", "T03_I004"]},
    {"status": "ambiguous", "issue_id": None, "candidate_issue_ids": ["T03_I004", "T02_I001"]},
    {"status": "product_mismatch"},
])
def test_invalid_decisions_fail_closed(change):
    with client_for({**GOOD, **change}) as c:
        r = c.post("/classify-issue", json=REQUEST)
        assert r.status_code == 502, r.text
        assert r.json()["detail"] == "invalid_issue_response"
        assert TEXT not in r.text and "test-only-secret" not in r.text


@pytest.mark.parametrize("body", [
    {**REQUEST, "team": "T99"}, {**REQUEST, "text": " " * 20},
    {**REQUEST, "text": "a" * 20001}, {**REQUEST, "request_id": "bad\nlog"},
    {**REQUEST, "expected_issue": "T03_I004"},
])
def test_bad_input_makes_no_provider_call(body):
    def fail(_):
        pytest.fail("provider should not be called")
    with TestClient(create_app(settings(), transport=httpx.MockTransport(fail))) as c:
        assert c.post("/classify-issue", json=body).status_code == 422


@pytest.mark.parametrize("response_mode", ["json_object", "json_schema", "none"])
def test_prompt_contains_whole_catalog_readable_labels_and_separate_complaint(response_mode):
    def handle(request):
        payload = json.loads(request.content)
        assert str(request.url) == "https://llm.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-only-secret"
        assert payload["messages"][0]["role"] == "system"
        user = json.loads(payload["messages"][1]["content"])
        assert user["complaint"] == TEXT
        assert len(user["catalog"]["issues"]) == 12
        assert all(row["team"] == "T03" and row["issue"] for row in user["catalog"]["issues"])
        if response_mode == "json_schema":
            enum = payload["response_format"]["json_schema"]["schema"]["properties"]["issue_id"]["anyOf"][0]["enum"]
            assert "T03_I004" in enum and "T02_I001" not in enum
        if response_mode == "none":
            assert "response_format" not in payload
        return httpx.Response(200, json=envelope(GOOD))
    with TestClient(create_app(settings(response_format=response_mode), transport=httpx.MockTransport(handle))) as c:
        assert c.post("/classify-issue", json=REQUEST).status_code == 200


@pytest.mark.parametrize("reply", [
    {}, {"choices": []}, envelope(GOOD, "length"), envelope(GOOD, "content_filter"),
    {"choices": [{"message": {"content": "```json\n{}\n```"}, "finish_reason": "stop"}]},
])
def test_provider_envelope_and_truncation(reply):
    with TestClient(create_app(settings(), transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=reply)))) as c:
        assert c.post("/classify-issue", json=REQUEST).status_code == 502


def test_timeout_is_bounded_and_releases_slot():
    calls = 0
    async def slow(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(.1)
        return httpx.Response(200, json=envelope(GOOD))
    with TestClient(create_app(settings(timeout_s=.02, concurrency=1), transport=httpx.MockTransport(slow))) as c:
        assert c.post("/classify-issue", json=REQUEST).status_code == 504
        assert c.post("/classify-issue", json=REQUEST).status_code == 200


@pytest.mark.parametrize("code", [401, 429, 500])
def test_provider_errors_are_sanitized(code):
    with TestClient(create_app(settings(), transport=httpx.MockTransport(
            lambda _: httpx.Response(code, text="private upstream details")))) as c:
        r = c.post("/classify-issue", json=REQUEST)
        assert r.status_code == 502 and "private" not in r.text


def product_response(rid="example-1", team="T03"):
    return {"model_name": "ensemble", "model_version": "fixture", "team": team,
            "team_confidence": .9, "top_issues": [], "auto_route": True,
            "request_id": rid, "latency_ms": 10}


def test_combined_product_then_issue_without_overwriting_model_probabilities():
    calls = []
    def handle(request):
        calls.append(request.url.host)
        if request.url.host == "product.test":
            assert "authorization" not in request.headers
            assert json.loads(request.content) == {"text": TEXT, "request_id": "example-1"}
            return httpx.Response(200, json=product_response())
        return httpx.Response(200, json=envelope(GOOD))
    with TestClient(create_app(settings(product_url="http://product.test"),
                              transport=httpx.MockTransport(handle))) as c:
        r = c.post("/predict", json={"text": TEXT, "request_id": "example-1"})
        assert r.status_code == 200, r.text
        result = r.json()
        assert calls == ["product.test", "llm.test"]
        assert result["team_prediction"]["team_confidence"] == .9
        assert result["issue_result"]["issue_id"] == "T03_I004"
        assert result["review_required"]


@pytest.mark.parametrize("upstream", [{}, product_response(team="T99"), product_response(rid="wrong")])
def test_bad_product_result_does_not_call_llm(upstream):
    def handle(request):
        assert request.url.host == "product.test"
        return httpx.Response(200, json=upstream)
    with TestClient(create_app(settings(product_url="http://product.test"),
                              transport=httpx.MockTransport(handle))) as c:
        assert c.post("/predict", json={"text": TEXT, "request_id": "example-1"}).status_code == 502


def test_mock_startup_needs_no_key_model_or_encoder():
    cfg = Settings(backend="mock", catalog_dir=ROOT / "data/processed")
    with TestClient(create_app(cfg)) as c:
        r = c.post("/classify-issue", json=REQUEST)
        assert r.status_code == 200
        assert r.json()["backend"] == "mock" and r.json()["issue_id"] is None
        assert c.post("/predict", json={"text": TEXT}).status_code == 503


def test_real_backend_requires_explicit_credentials():
    with pytest.raises(RuntimeError, match="ISSUE_MODEL"):
        with TestClient(create_app(Settings(catalog_dir=ROOT / "data/processed"))):
            pass


def test_duplicate_json_keys_are_rejected():
    raw = json.dumps(GOOD)[:-1] + ', "issue_id": "T03_I009"}'
    with TestClient(create_app(settings(), transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"choices": [{"message": {"content": raw},
                                                             "finish_reason": "stop"}]})))) as c:
        assert c.post("/classify-issue", json=REQUEST).status_code == 502


def test_concurrent_requests_are_bounded():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def blocked(request):
            entered.set()
            await release.wait()
            return httpx.Response(200, json=envelope(GOOD))
        app = create_app(settings(concurrency=1), transport=httpx.MockTransport(blocked))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as c:
                first = asyncio.create_task(c.post("/classify-issue", json=REQUEST))
                await asyncio.wait_for(entered.wait(), timeout=1)
                try:
                    second = await c.post("/classify-issue", json=REQUEST)
                    assert second.status_code == 503
                    assert second.headers["Retry-After"] == "1"
                finally:
                    release.set()
                assert (await first).status_code == 200
    asyncio.run(run())


def test_missing_request_id_is_generated_and_propagated():
    ids = []
    def handle(request):
        if request.url.host == "product.test":
            rid = json.loads(request.content)["request_id"]
            ids.append(rid)
            return httpx.Response(200, json=product_response(rid=rid))
        return httpx.Response(200, json=envelope(GOOD))
    with TestClient(create_app(settings(product_url="http://product.test"),
                              transport=httpx.MockTransport(handle))) as c:
        r = c.post("/predict", json={"text": TEXT})
        assert r.status_code == 200
        assert r.json()["issue_result"]["request_id"] == ids[0]
