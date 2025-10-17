# EasyLife AI Observability Stack

This document describes the complete observability stack for EasyLife AI, including monitoring, logging, tracing, and alerting.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Services      │    │   Observability │    │   Dashboards    │
│                 │    │                 │    │                 │
│ • NLP Service   │───▶│ • Prometheus    │───▶│ • Grafana       │
│ • CV Service    │    │ • Jaeger        │    │ • MLflow UI     │
│ • TS Service    │    │ • ELK Stack     │    │ • Jaeger UI     │
│ • Recsys Service│    │ • AlertManager  │    │ • Kibana        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Monitoring Stack

### Prometheus
- **Purpose**: Metrics collection and storage
- **URL**: http://localhost:9090
- **Configuration**: `docker/prometheus.yml`
- **Alert Rules**: `docker/alert_rules.yml`

### Grafana
- **Purpose**: Metrics visualization and dashboards
- **URL**: http://localhost:3000
- **Credentials**: admin/admin123
- **Dashboards**: Pre-configured for all services

## 🔍 Distributed Tracing

### Jaeger & OpenTelemetry
- **Purpose**: Distributed tracing across services
- **URL**: http://localhost:16686 (Jaeger UI)
- **Configuration**: `docker/tracing/`, `shared/tracing.py`
- **Enable**: set `OTEL_TRACING_ENABLED=1` and optionally
  `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317` before starting a service.

## 📝 Centralized Logging

### ELK Stack
- **Elasticsearch**: Log storage and indexing
- **Logstash**: Log processing and transformation
- **Kibana**: Log visualization and analysis
- **Filebeat**: Log collection from containers

## 🚨 Alerting

### Alert Rules
Located in `docker/alert_rules.yml`:

- **Service Health**: Monitor service availability
- **Error Rates**: Track HTTP 5xx errors
- **Response Times**: Monitor 95th percentile latency
- **Service-Specific**: NLP, CV, TS, Recsys specific alerts

### Automation Script

- Run `./scripts/setup-observability.sh` to start Prometheus, Grafana, MLflow,
  logging (ELK), and tracing stacks in one command. The script also performs
  basic health checks and prints access URLs.

### Alert Severity Levels
- **Critical**: Service down, MLflow unavailable
- **Warning**: High error rates, slow response times

## 🚀 Quick Start

### 1. Start Observability Stack
```bash
./scripts/setup-observability.sh
```

### 2. Start Services with Tracing
```bash
# Set environment variables for tracing
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=nlp-service

# Start services
python -m uvicorn nlp_service.app.main:app --host 0.0.0.0 --port 8001
```

### 3. Access Dashboards
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686
- **Kibana**: http://localhost:5601
- **MLflow**: http://localhost:5001

## 📈 Metrics Collected

### Service Metrics
- **HTTP Requests**: Rate, duration, status codes
- **Service Health**: Up/down status
- **Custom Metrics**: Service-specific business metrics

### Business Metrics
- **NLP**: Prediction counts, confidence scores
- **CV**: Image processing counts, accuracy
- **TS**: Forecast requests, horizon distribution
- **Recsys**: Recommendation counts, quality scores

## 🔧 Configuration

### Prometheus Scraping
Services are configured to be scraped by Prometheus:
- **NLP Service**: localhost:8001/metrics
- **CV Service**: localhost:8002/metrics
- **TS Service**: localhost:8003/metrics
- **Recsys Service**: localhost:8004/metrics

### Grafana Dashboards
Pre-configured dashboards:
- **Service Overview**: All services health and performance
- **NLP Service**: NLP-specific metrics
- **Time Series**: Forecasting metrics
- **Recommendation System**: Recsys metrics

### Tracing Configuration
Each service should be instrumented with:
```python
from shared.tracing import setup_tracing, instrument_fastapi_app

# Setup tracing
tracer = setup_tracing("service-name")

# Instrument FastAPI app
instrument_fastapi_app(app, "service-name")
```

## 🛠️ Troubleshooting

### Common Issues

1. **Services not appearing in Prometheus**
   - Check if services are exposing `/metrics` endpoint
   - Verify Prometheus configuration includes service targets

2. **No traces in Jaeger**
   - Ensure `OTEL_EXPORTER_OTLP_ENDPOINT` is set
   - Check if OpenTelemetry collector is running

3. **Logs not appearing in Kibana**
   - Verify Filebeat is collecting container logs
   - Check Logstash configuration

### Health Checks
```bash
# Check Prometheus
curl http://localhost:9090/api/v1/query?query=up

# Check Grafana
curl http://localhost:3000/api/health

# Check Jaeger
curl http://localhost:16686/api/services

# Check Elasticsearch
curl http://localhost:9200/_cluster/health
```

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [ELK Stack Documentation](https://www.elastic.co/guide/)
