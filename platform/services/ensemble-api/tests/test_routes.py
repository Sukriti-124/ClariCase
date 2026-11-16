import httpx
import pytest
import respx

from tests.conftest import probs_for

TEXT = "The bank reported wrong information to the credit bureau after many disputes."
SVM = "http://svm-api.test"
EMB = "http://embed-api.test"
ISSUES_T01 = [{"issue": "T01_I005", "prob": 0.6}, {"issue": "T01_I004", "prob": 0.3}]


def svm_resp(team="T01", conf=0.9, b2_team=None, b2_conf=0.8):
    return {
        "model_name": "improved_model_1", "model_version": "sha256:a",
        "team": team, "team_confidence": conf, "top_issues": ISSUES_T01,
        "auto_route": True, "request_id": "up", "latency_ms": 3,
        "team_probs": probs_for(team, conf),
        "extra_team_probs": {"baseline_model_2":
                             probs_for(b2_team or team, b2_conf)},
    }


def emb_resp(team="T01", conf=0.9, knn_team=None, knn_conf=0.8):
    return {
        "model_name": "improved_model_2", "model_version": "sha256:b",
        "team": team, "team_confidence": conf, "top_issues": [],
        "auto_route": True, "request_id": "up", "latency_ms": 3,
        "team_probs": probs_for(team, conf),
        "extra_team_probs": {"knn_embed": probs_for(knn_team or team, knn_conf)},
    }


def _mock(svm=None, emb=None):
    respx.post(f"{SVM}/predict").mock(return_value=httpx.Response(
        200, json=svm or svm_resp()))
    respx.post(f"{EMB}/predict").mock(return_value=httpx.Response(
        200, json=emb or emb_resp()))


@respx.mock
def test_agreement_auto_routes_with_issues_from_svm(client):
    _mock()
    resp = client.post("/predict", json={"text": TEXT})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model_name"] == "ensemble_stack_v2"
    assert body["model_version"].startswith("sha256:")
    assert body["team"] == "T01"
    assert body["scenario"] == "auto" and body["auto_route"] is True
    assert body["top_issues"] == ISSUES_T01
    assert abs(sum(body["stack_team_probs"].values()) - 1.0) < 1e-6
    assert body["team_probs"] == probs_for("T01", 0.9)
    assert resp.headers["x-routing-scenario"] == "auto"


@respx.mock
def test_disagreement_is_contested_and_not_auto_routed(client):
    _mock(svm=svm_resp("T01", 0.45, b2_team="T01", b2_conf=0.4),
          emb=emb_resp("T02", 0.45, knn_team="T02", knn_conf=0.4))
    respx.post(f"{SVM}/issues").mock(return_value=httpx.Response(
        200, json={"model_name": "improved_model_1", "model_version": "sha256:a",
                   "team": "T02", "request_id": "r", "latency_ms": 1,
                   "top_issues": []}))
    body = client.post("/predict", json={"text": TEXT}).json()
    assert body["scenario"] == "contested"
    assert body["auto_route"] is False


@respx.mock
def test_stacker_override_refetches_issues_for_its_team(client):
    _mock(svm=svm_resp("T01", 0.45, b2_team="T02", b2_conf=0.9),
          emb=emb_resp("T02", 0.95, knn_team="T02", knn_conf=0.95))
    issues_route = respx.post(f"{SVM}/issues").mock(return_value=httpx.Response(
        200, json={"model_name": "improved_model_1", "model_version": "sha256:a",
                   "team": "T02", "request_id": "r", "latency_ms": 1,
                   "top_issues": [{"issue": "T02_I001", "prob": 0.5}]}))
    body = client.post("/predict", json={"text": TEXT}).json()
    assert body["team"] == "T02"
    assert issues_route.called
    sent = issues_route.calls[0].request.read().decode()
    assert '"team": "T02"' in sent or '"team":"T02"' in sent
    assert body["top_issues"] == [{"issue": "T02_I001", "prob": 0.5}]


@respx.mock
def test_issue_refetch_failure_returns_no_issues_not_wrong_ones(client):
    _mock(svm=svm_resp("T01", 0.45, b2_team="T02", b2_conf=0.9),
          emb=emb_resp("T02", 0.95, knn_team="T02", knn_conf=0.95))
    respx.post(f"{SVM}/issues").mock(return_value=httpx.Response(500))
    body = client.post("/predict", json={"text": TEXT}).json()
    assert body["team"] == "T02" and body["top_issues"] == []


@respx.mock
def test_embed_api_down_degrades_to_primary_policy(client):
    respx.post(f"{SVM}/predict").mock(return_value=httpx.Response(
        200, json=svm_resp("T01", 0.97)))
    respx.post(f"{EMB}/predict").mock(side_effect=httpx.ConnectError("down"))
    resp = client.post("/predict", json={"text": TEXT})
    body = resp.json()
    assert resp.status_code == 200
    assert body["scenario"] == "degraded"
    assert body["team"] == "T01" and body["auto_route"] is True
    assert body["team_confidence"] == 0.97
    assert body["stack_team_probs"] is None
    assert resp.headers["x-missing-models"] == "improved_model_2,knn_embed"


@respx.mock
def test_embed_api_down_low_confidence_goes_to_review(client):
    respx.post(f"{SVM}/predict").mock(return_value=httpx.Response(
        200, json=svm_resp("T01", 0.4)))
    respx.post(f"{EMB}/predict").mock(side_effect=httpx.ConnectError("down"))
    body = client.post("/predict", json={"text": TEXT}).json()
    assert body["scenario"] == "degraded" and body["auto_route"] is False


@respx.mock
@pytest.mark.parametrize("failure", [httpx.Response(500, text="boom"),
                                     httpx.TimeoutException("slow")])
def test_svm_api_failure_is_503(client, failure):
    kw = ({"return_value": failure} if isinstance(failure, httpx.Response)
          else {"side_effect": failure})
    respx.post(f"{SVM}/predict").mock(**kw)
    respx.post(f"{EMB}/predict").mock(return_value=httpx.Response(
        200, json=emb_resp()))
    resp = client.post("/predict", json={"text": TEXT})
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "primary_unhealthy"


@respx.mock
def test_phase1_svm_response_without_probs_is_handled(client):
    old = {k: v for k, v in svm_resp().items()
           if k not in ("team_probs", "extra_team_probs")}
    respx.post(f"{SVM}/predict").mock(return_value=httpx.Response(200, json=old))
    respx.post(f"{EMB}/predict").mock(return_value=httpx.Response(
        200, json=emb_resp()))
    resp = client.post("/predict", json={"text": TEXT})
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "primary_has_no_probabilities"


@respx.mock
def test_label_shift_activates_after_enough_requests(client):
    _mock(svm=svm_resp("T02", 0.9, b2_team="T02"),
          emb=emb_resp("T02", 0.9, knn_team="T02"))
    assert client.get("/prior").json()["active"] is False
    for _ in range(60):
        client.post("/predict", json={"text": TEXT})
    p = client.get("/prior").json()
    assert p["active"] is True and p["n_seen"] == 60
    assert p["estimated_prior"]["T02"] > p["source_prior"]["T02"]
    body = client.post("/predict", json={"text": TEXT}).json()
    assert body["team"] == "T02"
    assert client.post("/prior/reset").json() == {"status": "reset"}
    assert client.get("/prior").json()["active"] is False


@respx.mock
def test_healthz_needs_svm_api_only(client):
    respx.get(f"{SVM}/healthz").mock(return_value=httpx.Response(
        200, json={"status": "ok", "extra_models": ["baseline_model_2"]}))
    respx.get(f"{EMB}/healthz").mock(side_effect=httpx.ConnectError("down"))
    body = client.get("/healthz").json()
    assert body["status"] == "ok" and body["embed_api"] == "unreachable"
    assert body["models"] == ["improved_model_1", "improved_model_2",
                              "baseline_model_2", "knn_embed"]
    respx.get(f"{SVM}/healthz").mock(return_value=httpx.Response(503))
    assert client.get("/healthz").status_code == 503


@respx.mock
def test_metrics_expose_routing_counters(client):
    _mock()
    client.post("/predict", json={"text": TEXT})
    text = client.get("/metrics").text
    assert "routing_decisions_total" in text
    assert "routing_team_total" in text
    assert "label_shift_active" in text


def test_short_text_is_rejected(client):
    assert client.post("/predict", json={"text": "short"}).status_code == 422
