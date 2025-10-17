from __future__ import annotations

from fastapi import FastAPI

from shared import tracing


def test_setup_tracing_respects_disable_flag(monkeypatch):
    monkeypatch.setenv("OTEL_TRACING_ENABLED", "0")
    tracer = tracing.setup_tracing("test-service")
    assert tracer is None


def test_setup_tracing_idempotent(monkeypatch):
    monkeypatch.setenv("OTEL_TRACING_ENABLED", "1")
    monkeypatch.setattr(tracing, "_CONFIGURED", False, raising=False)
    tracer_one = tracing.setup_tracing("test-service")
    tracer_two = tracing.setup_tracing("test-service")
    assert tracer_one is not None
    assert tracer_two is not None
    assert (
        tracer_one._instrumentation_scope == tracer_two._instrumentation_scope
    )  # noqa: SLF001


def test_instrument_fastapi_app_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("OTEL_TRACING_ENABLED", "0")
    monkeypatch.setattr(tracing, "_FASTAPI_INSTRUMENTED", False, raising=False)
    app = FastAPI()
    tracing.instrument_fastapi_app(app, "test-service")


def test_instrument_fastapi_app_configures_once(monkeypatch):
    monkeypatch.setenv("OTEL_TRACING_ENABLED", "1")
    monkeypatch.setattr(tracing, "_FASTAPI_INSTRUMENTED", False, raising=False)
    monkeypatch.setattr(tracing, "_CONFIGURED", False, raising=False)
    tracing.setup_tracing("test-service")
    app = FastAPI()
    tracing.instrument_fastapi_app(app, "test-service")
    tracing.instrument_fastapi_app(app, "test-service")
    assert tracing._FASTAPI_INSTRUMENTED is True
