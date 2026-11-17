"""Observability setup: Prometheus metrics + OpenTelemetry tracing.

Call `setup_observability(app, service_name)` once in each FastAPI
service's app factory. Metrics are always on. OTel tracing exports
only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set; otherwise a no-op
tracer provider is installed so `trace.get_tracer(...)` still works.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_fastapi_instrumentator import Instrumentator


def _configure_tracing(service_name: str) -> None:
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if endpoint:
        # Imported lazily so services that don't configure OTel avoid
        # paying for gRPC at startup.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )

    trace.set_tracer_provider(provider)


def setup_observability(app: FastAPI, service_name: str) -> None:
    _configure_tracing(service_name)
    FastAPIInstrumentor.instrument_app(app)

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
