from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


def test_predict_requires_features():
    client = TestClient(app)
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 422
