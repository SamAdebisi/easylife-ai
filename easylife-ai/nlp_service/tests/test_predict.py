from app.main import app
from fastapi.testclient import TestClient


def test_predict_endpoint_returns_label_and_confidence():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "I love this new feature"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["label"] in (0, 1)
        assert 0.0 <= payload["confidence"] <= 1.0


def test_metrics_endpoint_exposes_prediction_counter():
    with TestClient(app) as client:
        client.post(
            "/predict",
            json={"text": "I am not satisfied with the service"},
        )
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        assert "nlp_predictions_total" in metrics_response.text
