TEXT = "The collection agency keeps calling me about a debt I never owed and will not validate it."


def test_healthz_reports_model_and_pool(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok" and body["model"] == "agentic:mock"
    assert body["pool_size"] > 0


def test_choose_team_returns_a_candidate(client):
    resp = client.post("/choose", json={"text": TEXT, "task": "team",
                                        "candidates": ["T01", "T02", "T11"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choice"] in ("T01", "T02", "T11") and body["parsed"]
    assert body["model_name"] == "agentic:mock" and body["task"] == "team"
    assert body["request_id"] and body["latency_ms"] >= 0


def test_choose_issue_needs_team_and_in_team_candidates(client):
    ok = client.post("/choose", json={"text": TEXT, "task": "issue", "team": "T02",
                                      "candidates": ["T02_I001", "T02_I004"]})
    assert ok.status_code == 200 and ok.json()["choice"].startswith("T02_")
    assert client.post("/choose", json={"text": TEXT, "task": "issue",
                                        "candidates": ["T02_I001"]}).status_code == 422
    assert client.post("/choose", json={"text": TEXT, "task": "issue", "team": "T02",
                                        "candidates": ["T01_I005"]}).status_code == 422
    assert client.post("/choose", json={"text": TEXT, "task": "team",
                                        "candidates": ["T99"]}).status_code == 422


def test_single_candidate_short_circuits_without_llm(client):
    body = client.post("/choose", json={"text": TEXT, "task": "team",
                                        "candidates": ["T02"]}).json()
    assert body["choice"] == "T02" and body["cached"]


def test_second_identical_request_is_cached(client):
    req = {"text": TEXT, "task": "team", "candidates": ["T01", "T02"]}
    client.post("/choose", json=req)
    assert client.post("/choose", json=req).json()["cached"] is True
    text = client.get("/metrics").text
    assert "agentic_choices_total" in text and "agentic_cache_total" in text
