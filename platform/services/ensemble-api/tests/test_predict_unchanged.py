"""The PredictResponse is unchanged by explanations (Req 1.7, 1.8, 17.2).

Only `explanation_status` and per-request fields may differ between runs with
explanations enabled, disabled, and with a failing builder/explainer.
"""
import httpx
import respx

from tests.test_routes import SVM, EMB, TEXT, svm_resp, emb_resp

VOLATILE = {"explanation_status", "request_id", "latency_ms"}


def _post(client):
    respx.post(f"{SVM}/predict").mock(return_value=httpx.Response(
        200, json=svm_resp()))
    respx.post(f"{EMB}/predict").mock(return_value=httpx.Response(
        200, json=emb_resp()))
    respx.post(f"{SVM}/explain").mock(return_value=httpx.Response(500))
    respx.post(f"{EMB}/explain").mock(return_value=httpx.Response(500))
    resp = client.post("/predict", json={"text": TEXT})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _stable(body):
    return {k: v for k, v in body.items() if k not in VOLATILE}


@respx.mock
def test_predict_identical_enabled_disabled_and_failing(client, monkeypatch):
    enabled = _post(client)
    assert enabled["explanation_status"] == "pending"

    # Failing explainer (background pipeline raises).
    import src.explain.pipeline as pipeline
    def boom(*a, **k):
        raise RuntimeError("explainer down")
    monkeypatch.setattr(pipeline, "explain_platform", boom)
    failing_explainer = _post(client)

    # Failing builder (context construction raises, setup fails).
    import src.explain.explainers.routing as routing
    monkeypatch.setattr(routing, "PlatformDecisionContext", boom)
    failing_builder = _post(client)
    assert failing_builder["explanation_status"] == "unavailable"

    # Disabled: no store on app state.
    client.app.state.explain_store = None
    disabled = _post(client)
    assert disabled["explanation_status"] == "unavailable"

    base = _stable(enabled)
    assert _stable(failing_explainer) == base
    assert _stable(failing_builder) == base
    assert _stable(disabled) == base
