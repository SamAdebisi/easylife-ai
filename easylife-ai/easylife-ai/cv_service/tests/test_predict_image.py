from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from cv_service.app.main import app, ensure_model_artifact


def _generate_image(blur: bool) -> bytes:
    image = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (118, 118), color=200, thickness=3)
    cv2.line(image, (0, 0), (127, 127), color=255, thickness=2)
    cv2.line(image, (0, 127), (127, 0), color=255, thickness=2)

    if blur:
        image = cv2.GaussianBlur(image, (9, 9), 3.0)

    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def test_predict_image_returns_classification():
    ensure_model_artifact()
    sharp_bytes = _generate_image(blur=False)

    with TestClient(app) as client:
        response = client.post(
            "/predict_image",
            files={"file": ("sharp.png", sharp_bytes, "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in (0, 1)
    assert payload["label_name"] in ("sharp", "blurred")
    assert payload["confidence"] >= 0.0
    assert payload["score"] >= 0.0
    assert payload["model_variant"] in ("threshold", "cnn")


def test_metrics_endpoint_exposes_counter():
    ensure_model_artifact()
    blurred_bytes = _generate_image(blur=True)

    with TestClient(app) as client:
        client.post(
            "/predict_image",
            files={"file": ("blurred.png", blurred_bytes, "image/png")},
        )
        metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200
    assert "cv_predictions_total" in metrics_response.text
