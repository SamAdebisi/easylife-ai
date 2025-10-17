from fastapi.testclient import TestClient

from ts_forecasting.app.main import app


def test_forecast_endpoint_returns_values():
    client = TestClient(app)
    response = client.post("/forecast", json={"horizon": 10})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["timestamps"]) == 10
    assert len(payload["values"]) == 10
    # Values should be floats that can be cast without raising.
    assert all(isinstance(v, (int, float)) for v in payload["values"])
