from fastapi.testclient import TestClient

from nlp_service.app.main import app


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


def test_explain_endpoint_returns_tokens():
    with TestClient(app) as client:
        response = client.post(
            "/explain",
            json={"text": "This product is fantastic!", "top_k": 3},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["label"] in (0, 1)
        assert 0.0 <= payload["confidence"] <= 1.0
        assert len(payload["tokens"]) <= 3
        if payload["tokens"]:
            assert all(item["contribution"] > 0 for item in payload["tokens"])
