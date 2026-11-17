from fastapi import FastAPI
from fastapi.testclient import TestClient

from claricase_platform_shared.telemetry import setup_observability


def test_metrics_endpoint_exposes_prometheus_format():
    app = FastAPI()
    setup_observability(app, service_name="svm-api")

    @app.get("/ping")
    def ping():
        return {"ok": True}

    client = TestClient(app)
    client.get("/ping")  # drive at least one request

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "# HELP" in body
    assert "http_request_duration" in body or "http_requests_total" in body


def test_otel_no_op_when_endpoint_unset(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    app = FastAPI()
    # Must not raise even though no collector exists:
    setup_observability(app, service_name="router")
