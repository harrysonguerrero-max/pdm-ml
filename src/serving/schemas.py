"""
Pydantic v2 request/response schemas for the Predictive Maintenance API.

Design decisions:
- All pre-computed features are required in the request body.
  The API is a pure scoring endpoint — it does not do feature engineering.
  Feature engineering happens in the training pipeline and must be replicated
  by the caller before sending the request.
- machine_id is included for traceability in logs and responses.
- Strict field validation (ge/le constraints) catches bad inputs early.
"""
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """All pre-computed features for a single machine at a single timestamp."""

    # Identity
    machine_id: int = Field(..., ge=1, le=100, description="Machine identifier (1-100)")

    # Raw telemetry
    volt:      float = Field(..., description="Voltage reading (V)")
    rotate:    float = Field(..., description="Rotation speed (RPM)")
    pressure:  float = Field(..., description="Pressure (PSI)")
    vibration: float = Field(..., description="Vibration level")

    # Rolling mean — 3h window
    volt_mean_3h:      float = Field(...)
    rotate_mean_3h:    float = Field(...)
    pressure_mean_3h:  float = Field(...)
    vibration_mean_3h: float = Field(...)

    # Rolling std — 3h window
    volt_std_3h:      float = Field(...)
    rotate_std_3h:    float = Field(...)
    pressure_std_3h:  float = Field(...)
    vibration_std_3h: float = Field(...)

    # Rolling mean — 24h window
    volt_mean_24h:      float = Field(...)
    rotate_mean_24h:    float = Field(...)
    pressure_mean_24h:  float = Field(...)
    vibration_mean_24h: float = Field(...)

    # Rolling std — 24h window
    volt_std_24h:      float = Field(...)
    rotate_std_24h:    float = Field(...)
    pressure_std_24h:  float = Field(...)
    vibration_std_24h: float = Field(...)

    # Lag features — 1h
    volt_lag1:      float = Field(...)
    rotate_lag1:    float = Field(...)
    pressure_lag1:  float = Field(...)
    vibration_lag1: float = Field(...)

    # Lag features — 2h
    volt_lag2:      float = Field(...)
    rotate_lag2:    float = Field(...)
    pressure_lag2:  float = Field(...)
    vibration_lag2: float = Field(...)

    # Lag features — 3h
    volt_lag3:      float = Field(...)
    rotate_lag3:    float = Field(...)
    pressure_lag3:  float = Field(...)
    vibration_lag3: float = Field(...)

    # Rate of change (delta)
    volt_delta:      float = Field(...)
    rotate_delta:    float = Field(...)
    pressure_delta:  float = Field(...)
    vibration_delta: float = Field(...)

    # Error counts (past 24h)
    error1_count: int = Field(default=0, ge=0)
    error2_count: int = Field(default=0, ge=0)
    error3_count: int = Field(default=0, ge=0)
    error4_count: int = Field(default=0, ge=0)
    error5_count: int = Field(default=0, ge=0)

    # Component maintenance age (hours since last replacement)
    hours_since_comp1: int = Field(default=0, ge=0)
    hours_since_comp2: int = Field(default=0, ge=0)
    hours_since_comp3: int = Field(default=0, ge=0)
    hours_since_comp4: int = Field(default=0, ge=0)

    # Machine metadata
    age:      int = Field(..., ge=0, description="Machine age in years")


class PredictionResponse(BaseModel):
    """Fully auditable prediction response."""
    machine_id:              int
    failure_probability:     float = Field(..., description="Probability of failure [0, 1]")
    prediction:              int   = Field(..., description="1 = failure expected, 0 = normal")
    prediction_label:        str   = Field(..., description="Human-readable prediction")
    prediction_window_hours: int   = Field(..., description="Forecast horizon (hours)")
    model_version:           str   = Field(..., description="MLflow model version used")
    threshold_used:          float = Field(..., description="Decision threshold applied")


class HealthResponse(BaseModel):
    """API liveness and readiness check."""
    status:        str  = Field(..., description="healthy | degraded")
    model_loaded:  bool = Field(..., description="True if model loaded from Registry")
    model_version: str  = Field(..., description="Current loaded model version")