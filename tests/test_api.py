"""
Tests for src.serving.app

Coverage:
    - GET /health returns 200 with degraded status when no model loaded
    - POST /predict returns 503 when no model loaded
    - POST /predict returns correct prediction with a mocked model
    - POST /predict returns 1 (failure) when probability > threshold
    - POST /predict returns 0 (normal) when probability < threshold
    - POST /predict validates required fields (422 on bad input)
    - Response schema fields are all present
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.serving.app import app

client = TestClient(app)

# ── Full valid payload ─────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "machine_id": 1,
    "volt": 170.0,      "rotate": 450.0,
    "pressure": 95.0,   "vibration": 40.0,
    # rolling 3h
    "volt_mean_3h": 170.5,  "volt_std_3h": 1.2,
    "rotate_mean_3h": 450.1,"rotate_std_3h": 0.8,
    "pressure_mean_3h": 95.1,"pressure_std_3h": 0.5,
    "vibration_mean_3h": 40.1,"vibration_std_3h": 0.3,
    # rolling 24h
    "volt_mean_24h": 170.2, "volt_std_24h": 2.1,
    "rotate_mean_24h": 449.9,"rotate_std_24h": 1.9,
    "pressure_mean_24h": 95.2,"pressure_std_24h": 1.1,
    "vibration_mean_24h": 40.0,"vibration_std_24h": 0.9,
    # lags
    "volt_lag1": 170.1, "volt_lag2": 169.8, "volt_lag3": 170.3,
    "rotate_lag1": 450.2,"rotate_lag2": 449.8,"rotate_lag3": 450.0,
    "pressure_lag1": 94.8,"pressure_lag2": 95.2,"pressure_lag3": 95.0,
    "vibration_lag1": 40.2,"vibration_lag2": 39.8,"vibration_lag3": 40.1,
    # deltas
    "volt_delta": 0.5, "rotate_delta": -0.3,
    "pressure_delta": 0.2, "vibration_delta": -0.1,
    # error counts
    "error1_count": 0, "error2_count": 1, "error3_count": 0,
    "error4_count": 0, "error5_count": 0,
    # component ages
    "hours_since_comp1": 120, "hours_since_comp2": 240,
    "hours_since_comp3": 60,  "hours_since_comp4": 180,
    # machine metadata
    "model_id": 2, "age": 7,
}


# ── Health endpoint ────────────────────────────────────────────────────────


def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200


def test_health_degraded_when_no_model():
    r = client.get("/health")
    body = r.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert "model_version" in body


# ── Predict endpoint — no model ────────────────────────────────────────────


def test_predict_returns_503_when_no_model():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 503


def test_predict_returns_422_on_missing_field():
    """Pydantic validation: missing machine_id must return 422."""
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "machine_id"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_returns_422_on_invalid_machine_id():
    """machine_id must be between 1 and 100."""
    bad = {**VALID_PAYLOAD, "machine_id": 999}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


# ── Predict endpoint — with mocked model ──────────────────────────────────


@pytest.fixture(autouse=False)
def mock_model_high_prob():
    """Injects a mock model returning probability 0.82 (above threshold 0.35)."""
    import src.serving.app as m
    mock = MagicMock()
    mock.predict.return_value = np.array([0.82])
    m._model         = mock
    m._model_version = "test-v1"
    yield
    m._model         = None
    m._model_version = "not_loaded"


@pytest.fixture(autouse=False)
def mock_model_low_prob():
    """Injects a mock model returning probability 0.10 (below threshold 0.35)."""
    import src.serving.app as m
    mock = MagicMock()
    mock.predict.return_value = np.array([0.10])
    m._model         = mock
    m._model_version = "test-v1"
    yield
    m._model         = None
    m._model_version = "not_loaded"


def test_predict_failure_when_high_probability(mock_model_high_prob):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 1
    assert body["prediction_label"] == "FAILURE EXPECTED"
    assert body["failure_probability"] == pytest.approx(0.82, abs=1e-3)
    assert body["machine_id"] == 1
    assert body["model_version"] == "test-v1"
    assert body["threshold_used"] == 0.35


def test_predict_normal_when_low_probability(mock_model_low_prob):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 0
    assert body["prediction_label"] == "NORMAL OPERATION"
    assert body["failure_probability"] == pytest.approx(0.10, abs=1e-3)


def test_predict_response_has_all_required_fields(mock_model_high_prob):
    r = client.post("/predict", json=VALID_PAYLOAD)
    body = r.json()
    required_fields = [
        "machine_id", "failure_probability", "prediction",
        "prediction_label", "prediction_window_hours",
        "model_version", "threshold_used",
    ]
    for field in required_fields:
        assert field in body, f"Missing field in response: {field}"


def test_health_healthy_when_model_loaded(mock_model_high_prob):
    r = client.get("/health")
    assert r.json()["status"] == "healthy"
    assert r.json()["model_loaded"] is True