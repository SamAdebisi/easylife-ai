from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from recsys_service import data_utils
from recsys_service.app.main import app
from recsys_service.train import MODEL_PATH
from recsys_service.train import main as train_main


@pytest.fixture(scope="session", autouse=True)
def _ensure_artifacts() -> None:
    data_utils.ensure_datasets()
    if not MODEL_PATH.exists():
        train_main()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_version"] == "svd"


def test_recommendations_existing_user(client: TestClient) -> None:
    dataset = data_utils.ensure_datasets()
    user_id = dataset.train["user_id"].iat[0]
    response = client.post(
        "/recommendations",
        json={"user_id": user_id, "top_k": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == user_id
    assert len(payload["recommendations"]) == 5
    assert all("item_id" in item for item in payload["recommendations"])


def test_recommendations_new_user(client: TestClient) -> None:
    response = client.post(
        "/recommendations",
        json={"user_id": "new-user", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "new-user"
    assert len(payload["recommendations"]) == 3


def test_similar_items(client: TestClient) -> None:
    dataset = data_utils.ensure_datasets()
    item_id = dataset.items["item_id"].iat[0]
    response = client.get(f"/items/{item_id}/similar?top_k=4")
    assert response.status_code == 200
    payload = response.json()
    assert payload["item_id"] == item_id
    assert len(payload["similar_items"]) == 4


def test_similar_items_not_found(client: TestClient) -> None:
    response = client.get("/items/unknown-item/similar")
    assert response.status_code == 404
