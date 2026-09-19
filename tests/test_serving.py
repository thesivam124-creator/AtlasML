from fastapi.testclient import TestClient
from src.atlasml.serving.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_predict_endpoint():
    payload = {
        "zone_id": 142,
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "lag_1": 25.0,
        "lag_4": 22.0,
        "rolling_mean_4": 24.5,
        "rolling_std_4": 3.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["zone_id"] == 142
    assert "prediction" in data
    assert "lower_bound" in data
    assert "upper_bound" in data
    assert data["lower_bound"] <= data["prediction"] <= data["upper_bound"]
    assert data["model_used"] in ["gbm", "ridge"]
