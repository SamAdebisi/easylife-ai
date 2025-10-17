# Phase 5 – MLOps Automation & Explainability

This phase focuses on adding observability and explainability layers across the
existing services so that models can be inspected and monitored in production.

Key deliverables:

1. **Explainability for NLP predictions** – expose an `/explain` endpoint that
   returns token-level contributions for the logistic regression pipeline and
   logs the explanation payload to MLflow for offline inspection.
2. **Cross-service monitoring utilities** – add a lightweight reporting script
   under `mlops/` that aggregates Prometheus metrics and dumps them as CSV so
   that batch anomaly detection jobs can be scheduled (or run via cron/DVC).
3. **Documentation & Tests** – extend docs to describe explainability/monitoring
   flows and ensure API contracts are covered by unit tests.

The implementation targets Phase 1 (NLP) for explainability while laying the
groundwork for other services to plug into the same monitoring harness.
