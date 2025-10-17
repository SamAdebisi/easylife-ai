#!/bin/bash

# Setup complete observability stack for EasyLife AI

echo "🚀 Setting up EasyLife AI Observability Stack..."

# Start core infrastructure
echo "📊 Starting core infrastructure..."
docker compose -f docker/docker-compose.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Start logging stack
echo "📝 Starting centralized logging..."
docker compose -f docker/logging/docker-compose.logging.yml up -d

# Start tracing stack
echo "🔍 Starting distributed tracing..."
docker compose -f docker/tracing/docker-compose.tracing.yml up -d

# Wait for all services
echo "⏳ Waiting for all services to be ready..."
sleep 20

# Check service health
echo "🔍 Checking service health..."

# Check Prometheus
if curl -s http://localhost:9090/api/v1/query?query=up > /dev/null; then
    echo "✅ Prometheus is running"
else
    echo "❌ Prometheus is not responding"
fi

# Check Grafana
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Grafana is running"
else
    echo "❌ Grafana is not responding"
fi

# Check MLflow
if curl -s http://localhost:5001 > /dev/null; then
    echo "✅ MLflow is running"
else
    echo "❌ MLflow is not responding"
fi

# Check Jaeger
if curl -s http://localhost:16686 > /dev/null; then
    echo "✅ Jaeger is running"
else
    echo "❌ Jaeger is not responding"
fi

# Check Elasticsearch
if curl -s http://localhost:9200 > /dev/null; then
    echo "✅ Elasticsearch is running"
else
    echo "❌ Elasticsearch is not responding"
fi

echo ""
echo "🎉 Observability stack setup complete!"
echo ""
echo "📊 Access URLs:"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana: http://localhost:3000 (admin/admin123)"
echo "  MLflow: http://localhost:5001"
echo "  Jaeger: http://localhost:16686"
echo "  Kibana: http://localhost:5601"
echo "  Elasticsearch: http://localhost:9200"
echo ""
echo "🔧 Next steps:"
echo "  1. Start your services with tracing enabled"
echo "  2. Configure Grafana dashboards"
echo "  3. Set up alerting rules"
echo "  4. Monitor service health and performance"
