# NLP Sentiment Analysis Model Card

## Model Overview

**Model Name**: EasyLife AI Sentiment Classifier
**Version**: 1.0.0
**Type**: Binary Sentiment Classification
**Framework**: Scikit-learn with TF-IDF
**Last Updated**: October 2025

## Model Details

### Purpose
This model classifies text sentiment as positive (1) or negative (0) for customer reviews, social media posts, and feedback analysis.

### Architecture
- **Vectorizer**: TF-IDF with n-gram range (1,2)
- **Classifier**: Logistic Regression with L2 regularization
- **Features**: 10,000 most frequent terms
- **Preprocessing**: Lowercase, tokenization, stop word removal

### Training Data
- **Source**: IMDB Movie Reviews Dataset
- **Size**: 25,000 training samples, 25,000 test samples
- **Distribution**: 50% positive, 50% negative
- **Language**: English
- **Time Period**: 2000-2025

### Performance Metrics
```
┌─────────────────┬─────────────┬─────────────┐
│ Metric          │ Training    │ Test        │
├─────────────────┼─────────────┼─────────────┤
│ Accuracy        │ 0.89        │ 0.87        │
│ Precision       │ 0.88        │ 0.86        │
│ Recall          │ 0.90        │ 0.88        │
│ F1-Score        │ 0.89        │ 0.87        │
│ AUC-ROC         │ 0.95        │ 0.93        │
└─────────────────┴─────────────┴─────────────┘
```

## Usage

### API Endpoint
```bash
POST /predict
Content-Type: application/json

{
  "text": "This movie is absolutely fantastic!"
}
```

### Response
```json
{
  "label": 1,
  "confidence": 0.95,
  "sentiment": "positive"
}
```

### Python Usage
```python
from nlp_service.model import SentimentClassifier

classifier = SentimentClassifier()
result = classifier.predict("Great product!")
print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']}")
```

## Limitations

### Known Issues
1. **Domain Specificity**: Trained on movie reviews, may not generalize to other domains
2. **Sarcasm Detection**: Struggles with sarcastic or ironic statements
3. **Context Sensitivity**: Limited understanding of context and nuance
4. **Language Support**: English only, no multilingual support

### Bias Considerations
- **Training Data Bias**: IMDB dataset may contain demographic biases
- **Cultural Sensitivity**: May not account for cultural differences in sentiment expression
- **Temporal Bias**: Training data from specific time periods may not reflect current language use

## Monitoring & Maintenance

### Performance Monitoring
- **Accuracy Tracking**: Daily accuracy monitoring via MLflow
- **Drift Detection**: Weekly data drift analysis
- **Latency Monitoring**: <100ms response time target
- **Throughput**: 1000+ requests per second

### Retraining Schedule
- **Frequency**: Monthly retraining with new data
- **Trigger**: Performance degradation >5%
- **Validation**: A/B testing with shadow deployment
- **Rollback**: Automatic rollback on performance drop

### Data Requirements
- **Minimum Samples**: 1000 new samples for retraining
- **Quality Threshold**: 80% human-annotated accuracy
- **Label Distribution**: Balanced positive/negative samples
- **Freshness**: Data from last 3 months preferred

## Security & Privacy

### Data Protection
- **Input Sanitization**: Text preprocessing removes PII
- **No Storage**: Predictions not stored permanently
- **Encryption**: HTTPS-only API communication
- **Access Control**: API key authentication required

### Privacy Considerations
- **No Personal Data**: Model doesn't require personal information
- **Text Processing**: Input text processed in memory only
- **Audit Logging**: Request/response logging for compliance
- **GDPR Compliance**: Right to deletion and data portability

## Deployment

### Infrastructure
- **Container**: Docker with Python 3.10
- **Orchestration**: Kubernetes with HPA
- **Resources**: 512MB RAM, 0.5 CPU cores
- **Scaling**: 2-10 replicas based on load

### Dependencies
```
scikit-learn==1.5.2
numpy==2.1.1
pandas==2.2.2
fastapi==0.115.0
uvicorn==0.30.6
```

### Health Checks
- **Liveness**: `/health` endpoint
- **Readiness**: Model loading verification
- **Metrics**: `/metrics` Prometheus endpoint
- **Logging**: Structured JSON logging

## Contact & Support

- **Model Owner**: EasyLife AI Team
- **Repository**: [GitHub - NLP Service](https://github.com/SamAdebisi/easylife-ai/nlp_service)
- **Documentation**: [API Docs](http://localhost:8001/docs)
- **Issues**: [GitHub Issues](https://github.com/SamAdebisi/easylife-ai/issues)

---

*This model card provides transparency and accountability for the NLP sentiment analysis model. Regular updates ensure accuracy and reliability in production environments.*
