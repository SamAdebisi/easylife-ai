#!/bin/bash

# Setup script for Advanced Features (Phase 10 & 11)
# EasyLife AI - Advanced Features Installation

set -e

echo "🚀 Setting up Advanced Features for EasyLife AI..."
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

echo "✅ Python and pip are available"

# Install advanced requirements
echo "📦 Installing advanced features requirements..."
pip3 install -r requirements-advanced.txt

# Install additional dependencies
echo "📦 Installing additional dependencies..."

# Kafka setup
if ! command -v kafka-topics &> /dev/null; then
    echo "📦 Setting up Kafka..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install kafka
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Please install Kafka manually on Linux"
    else
        echo "Please install Kafka manually for your OS"
    fi
fi

# Docker setup for advanced features
echo "🐳 Setting up Docker services for advanced features..."

# Create docker-compose for advanced features
cat > docker/docker-compose.advanced.yml << 'EOF'
version: '3.8'

services:
  # Kafka for streaming
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'

  # Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes

  # Elasticsearch for analytics
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data

  # Kibana for analytics visualization
  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    depends_on:
      - elasticsearch
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200

volumes:
  elasticsearch-data:
EOF

echo "✅ Docker compose file created for advanced features"

# Create directories
echo "📁 Creating necessary directories..."
mkdir -p logs/advanced
mkdir -p data/streaming
mkdir -p data/analytics
mkdir -p models/compressed
mkdir -p models/multimodal
mkdir -p config/advanced

# Set permissions
chmod +x examples/advanced_features_demo.py

# Create configuration files
echo "⚙️ Creating configuration files..."

# Create logging configuration
cat > config/advanced/logging.yaml << 'EOF'
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  detailed:
    format: '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/advanced/advanced_features.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  federated_learning:
    level: INFO
    handlers: [console, file]
    propagate: false

  edge_deployment:
    level: INFO
    handlers: [console, file]
    propagate: false

  streaming:
    level: INFO
    handlers: [console, file]
    propagate: false

  analytics:
    level: INFO
    handlers: [console, file]
    propagate: false

  automl:
    level: INFO
    handlers: [console, file]
    propagate: false

  explainable_ai:
    level: INFO
    handlers: [console, file]
    propagate: false

  model_compression:
    level: INFO
    handlers: [console, file]
    propagate: false

  multimodal_ai:
    level: INFO
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
EOF

# Create environment file
cat > .env.advanced << 'EOF'
# Advanced Features Environment Variables

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_GROUP_ID=easylife-ai
KAFKA_AUTO_OFFSET_RESET=latest

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Elasticsearch Configuration
ELASTICSEARCH_HOST=localhost:9200
ELASTICSEARCH_INDEX=easylife-ai

# Security Configuration
ENABLE_ENCRYPTION=true
ENABLE_AUTHENTICATION=true
API_KEY_REQUIRED=true

# Privacy Configuration
ENABLE_DIFFERENTIAL_PRIVACY=true
ENABLE_SECURE_AGGREGATION=true
PRIVACY_BUDGET=1.0

# Performance Configuration
ENABLE_GPU_ACCELERATION=true
BATCH_SIZE=32
MAX_WORKERS=4
MEMORY_LIMIT_GB=8

# Monitoring Configuration
ENABLE_METRICS=true
ENABLE_LOGGING=true
ENABLE_TRACING=true
LOG_LEVEL=INFO
EOF

echo "✅ Configuration files created"

# Create test script
cat > tests/test_advanced_features.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for Advanced Features
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_learning.fedavg import FedAvgConfig, FederatedLearningOrchestrator
from edge_deployment.model_optimizer import EdgeOptimizationConfig, EdgeModelOptimizer
from analytics.business_intelligence import BIConfig, BusinessIntelligenceManager
from automl.hyperparameter_optimizer import OptimizationConfig, AutoMLPipeline
from explainable_ai.model_explainer import ExplanationConfig, ExplainableAIManager
from model_compression.compression_techniques import CompressionConfig, ModelCompressor
from multimodal_ai.fusion_models import MultiModalConfig, MultiModalManager


class TestAdvancedFeatures(unittest.TestCase):
    """Test cases for advanced features."""

    def test_federated_learning_config(self):
        """Test federated learning configuration."""
        config = FedAvgConfig(num_rounds=10, num_clients=5)
        self.assertEqual(config.num_rounds, 10)
        self.assertEqual(config.num_clients, 5)

    def test_edge_deployment_config(self):
        """Test edge deployment configuration."""
        config = EdgeOptimizationConfig(target_device="mobile")
        self.assertEqual(config.target_device, "mobile")

    def test_analytics_config(self):
        """Test analytics configuration."""
        config = BIConfig(refresh_interval=300)
        self.assertEqual(config.refresh_interval, 300)

    def test_automl_config(self):
        """Test AutoML configuration."""
        config = OptimizationConfig(optimization_method="bayesian")
        self.assertEqual(config.optimization_method, "bayesian")

    def test_explainable_ai_config(self):
        """Test explainable AI configuration."""
        config = ExplanationConfig(method="shap")
        self.assertEqual(config.method, "shap")

    def test_model_compression_config(self):
        """Test model compression configuration."""
        config = CompressionConfig(target_compression_ratio=0.5)
        self.assertEqual(config.target_compression_ratio, 0.5)

    def test_multimodal_config(self):
        """Test multi-modal AI configuration."""
        config = MultiModalConfig(fusion_dim=512)
        self.assertEqual(config.fusion_dim, 512)


if __name__ == '__main__':
    unittest.main()
EOF

chmod +x tests/test_advanced_features.py

echo "✅ Test script created"

# Create documentation
cat > docs/ADVANCED_FEATURES_SETUP.md << 'EOF'
# Advanced Features Setup Guide

## Overview
This guide covers the setup and configuration of advanced features for EasyLife AI, including Phase 10 (Advanced Features) and Phase 11 (AI/ML Innovation).

## Prerequisites
- Python 3.8+
- pip3
- Docker (optional, for containerized services)
- Kafka (for streaming features)
- Redis (for caching)
- Elasticsearch (for analytics)

## Installation

### 1. Install Dependencies
```bash
pip3 install -r requirements-advanced.txt
```

### 2. Setup Services
```bash
# Start Docker services
docker-compose -f docker/docker-compose.advanced.yml up -d

# Or install services manually
# Kafka, Redis, Elasticsearch
```

### 3. Configuration
Copy and modify configuration files:
- `config/advanced_features.yaml`
- `config/advanced/logging.yaml`
- `.env.advanced`

### 4. Run Tests
```bash
python3 tests/test_advanced_features.py
```

### 5. Run Demo
```bash
python3 examples/advanced_features_demo.py
```

## Features

### Phase 10: Advanced Features
- **Federated Learning**: Privacy-preserving collaborative training
- **Edge Deployment**: Mobile and IoT optimization
- **Real-time Streaming**: Kafka-based processing
- **Advanced Analytics**: Business intelligence dashboards

### Phase 11: AI/ML Innovation
- **AutoML**: Automated model selection and tuning
- **Explainable AI**: SHAP and LIME integration
- **Model Compression**: Quantization and pruning
- **Multi-modal AI**: Text, image, and time series fusion

## Usage Examples

### Federated Learning
```python
from federated_learning.fedavg import FederatedLearningOrchestrator, FedAvgConfig

config = FedAvgConfig(num_rounds=100, num_clients=10)
orchestrator = FederatedLearningOrchestrator(config)
```

### Edge Deployment
```python
from edge_deployment.model_optimizer import EdgeModelOptimizer, EdgeOptimizationConfig

config = EdgeOptimizationConfig(target_device="mobile")
optimizer = EdgeModelOptimizer(config)
```

### Real-time Streaming
```python
from streaming.kafka_processor import StreamingManager, StreamingConfig

config = StreamingConfig(bootstrap_servers=["localhost:9092"])
manager = StreamingManager(config)
```

### Advanced Analytics
```python
from analytics.business_intelligence import BusinessIntelligenceManager, BIConfig

config = BIConfig(refresh_interval=300)
manager = BusinessIntelligenceManager(config)
```

### AutoML
```python
from automl.hyperparameter_optimizer import AutoMLPipeline, OptimizationConfig

config = OptimizationConfig(optimization_method="bayesian")
automl = AutoMLPipeline(config)
```

### Explainable AI
```python
from explainable_ai.model_explainer import ExplainableAIManager, ExplanationConfig

config = ExplanationConfig(method="shap")
manager = ExplainableAIManager(config)
```

### Model Compression
```python
from model_compression.compression_techniques import ModelCompressor, CompressionConfig

config = CompressionConfig(target_compression_ratio=0.5)
compressor = ModelCompressor(config)
```

### Multi-modal AI
```python
from multimodal_ai.fusion_models import MultiModalManager, MultiModalConfig

config = MultiModalConfig(fusion_dim=512)
manager = MultiModalManager(config)
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Kafka Connection**: Check if Kafka is running on localhost:9092
3. **Memory Issues**: Adjust batch sizes and memory limits
4. **GPU Issues**: Ensure CUDA is properly installed for GPU acceleration

### Logs
Check logs in `logs/advanced/` directory for detailed information.

## Support
For issues and questions, please refer to the main project documentation or create an issue in the repository.
EOF

echo "✅ Documentation created"

# Final setup
echo "🎯 Finalizing setup..."

# Create startup script
cat > scripts/start_advanced_features.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting Advanced Features for EasyLife AI..."

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose -f docker/docker-compose.advanced.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check Kafka
if curl -s http://localhost:9092 > /dev/null; then
    echo "✅ Kafka is running"
else
    echo "❌ Kafka is not accessible"
fi

# Check Redis
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Redis is not accessible"
fi

# Check Elasticsearch
if curl -s http://localhost:9200 > /dev/null; then
    echo "✅ Elasticsearch is running"
else
    echo "❌ Elasticsearch is not accessible"
fi

echo "🎉 Advanced features setup complete!"
echo "Run 'python3 examples/advanced_features_demo.py' to test the features."
EOF

chmod +x scripts/start_advanced_features.sh

echo "✅ Startup script created"

echo ""
echo "🎉 Advanced Features Setup Complete!"
echo "====================================="
echo ""
echo "📋 What was installed:"
echo "  ✅ Advanced Python dependencies"
echo "  ✅ Docker services configuration"
echo "  ✅ Configuration files"
echo "  ✅ Test scripts"
echo "  ✅ Documentation"
echo "  ✅ Demo examples"
echo ""
echo "🚀 Next steps:"
echo "  1. Start services: ./scripts/start_advanced_features.sh"
echo "  2. Run tests: python3 tests/test_advanced_features.py"
echo "  3. Run demo: python3 examples/advanced_features_demo.py"
echo ""
echo "📚 Documentation: docs/ADVANCED_FEATURES_SETUP.md"
echo ""
echo "🎯 EasyLife AI now includes cutting-edge AI/ML capabilities!"
