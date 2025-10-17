"""Distributed tracing helpers for EasyLife AI services."""

from __future__ import annotations

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_CONFIGURED = False
_FASTAPI_INSTRUMENTED = False


def _tracing_disabled() -> bool:
    return os.getenv("OTEL_TRACING_ENABLED", "0") in {"0", "false", "False"}


def setup_tracing(
    service_name: str,
    otlp_endpoint: Optional[str] = None,
) -> Optional[trace.Tracer]:
    """
    Configure OpenTelemetry tracing for the given service.

    The function is idempotent and safe to call multiple times; tracing can be
    disabled entirely by setting ``OTEL_TRACING_ENABLED=0``.
    """
    global _CONFIGURED
    if _tracing_disabled():
        return None

    if not _CONFIGURED:
        endpoint = otlp_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://127.0.0.1:4317",
        )

        resource = Resource.create({SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        RequestsInstrumentor().instrument()
        URLLib3Instrumentor().instrument()

        _CONFIGURED = True

    return trace.get_tracer(service_name)


def instrument_fastapi_app(app, service_name: str) -> None:
    """Attach FastAPI instrumentation for tracing."""
    global _FASTAPI_INSTRUMENTED
    if _tracing_disabled():
        return
    if not _FASTAPI_INSTRUMENTED:
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=trace.get_tracer_provider(),
            excluded_urls="health",
        )
        _FASTAPI_INSTRUMENTED = True


def create_span(operation_name: str, **attributes):
    """
    Convenience helper to create a span context manager for ad-hoc tracing.
    """
    tracer = trace.get_tracer(__name__)
    return tracer.start_as_current_span(operation_name, attributes=attributes)
