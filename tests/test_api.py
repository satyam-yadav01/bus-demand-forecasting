
from fastapi.testclient import TestClient

from src.api.app import app


def test_home_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "route_id": "R01",
        "departure_datetime": "2025-10-15T08:00:00",
        "planned_capacity": 40,
        "season": "Monsoon",
        "peak_period": "Peak",
        "hist_route_time_avg": 42.5,
        "route_time_history_count": 4,
        "hist_route_avg": 38.2,
        "route_history_count": 20,
        "hist_global_time_avg": 35.7,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["route_id"] == "R01"
    assert result["planned_capacity"] == 40
    assert result["predicted_passenger_demand"] >= 0
    assert result["capacity_status"] in [
        "Insufficient capacity",
        "Capacity appears sufficient",
    ]


def test_prediction_rejects_invalid_capacity():
    payload = {
        "route_id": "R01",
        "departure_datetime": "2025-10-15T08:00:00",
        "planned_capacity": 0,
        "season": "Monsoon",
        "peak_period": "Peak",
        "hist_route_time_avg": 42.5,
        "route_time_history_count": 4,
        "hist_route_avg": 38.2,
        "route_history_count": 20,
        "hist_global_time_avg": 35.7,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
