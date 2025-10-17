# Phase 7 – Resiliency, Alerting & Production Ops

Phase 7 hardens the platform for production by introducing centralised logging,
distributed tracing, and actionable alerting. These upgrades build on the
existing observability foundation introduced in Phase 5 and ensure that every
service can be monitored and debugged end-to-end.

## Deliverables

1. **Distributed tracing across services** – instrument the FastAPI apps using
   OpenTelemetry helpers (`shared/tracing.py`) and route spans to the Jaeger
   collector via the OTLP pipeline.
2. **Logging stack** – provide a ready-to-run ELK stack (`docker/logging`),
   Filebeat shipping, and provisioning scripts so container logs land in Kibana.
3. **Alerting & dashboards** – wire Prometheus alert rules, update Grafana
   dashboards, and bundle a convenience script (`scripts/setup-observability.sh`)
   that brings the full observability stack online.

The remaining checklist for this phase: connect production-grade TLS, add alert
receivers (PagerDuty/Slack), and integrate log correlation IDs across the
services once a structured logging library is adopted.
