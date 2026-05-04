"""
FastAPI serving application for the Predictive Maintenance system.

Endpoints:
    GET  /health  — liveness + readiness check
    POST /predict — failure probability for a single machine

Architecture decisions:
- Model loaded ONCE at startup from MLflow Registry (not from a file path).
  This ensures the serving container always uses the 'Production' version
  tracked in the Registry, making rollback a Registry operation only.

- Degraded mode: if the Registry is unavailable at startup, the API
  still starts and returns 503 on /predict. This prevents crashing the
  container on a temporary MLflow connectivity issue.

- Decision threshold = 0.35 (configurable):
  FN (missed failure) costs ~5-10x more than FP (false alarm).
  Lower threshold increases recall at the cost of precision.
  This is the correct trade-off for predictive maintenance.

- Feature ordering: features are extracted from the request dict
  and sorted alphabetically before inference. The model was trained
  with sorted column names (see trainer.py _feature_columns).
  Order MUST match training — this is a common production bug.
"""
import numpy as np
from fastapi import FastAPI, HTTPException
from loguru import logger
import mlflow
import mlflow.pyfunc

from src.config import settings
from src.serving.schemas import HealthResponse, PredictionRequest, PredictionResponse

# Decision threshold — lower than 0.5 because FN >> FP in cost
_THRESHOLD = 0.35

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Predicts whether a machine will fail in the next "
        f"{settings.prediction_window_hours} hours. "
        "Model loaded from MLflow Registry (Production stage)."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Module-level model state — loaded once at startup
_model         = None
_model_version = "not_loaded"


@app.on_event("startup")
async def load_model() -> None:
    """
    Loads the Production model from MLflow Registry at startup.
    Runs in degraded mode if Registry is unreachable.
    """
    global _model, _model_version
    model_uri = f"models:/{settings.model_registry_name}/{settings.model_stage}"
    logger.info(f"Loading model from Registry: {model_uri}")

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        _model = mlflow.pyfunc.load_model(model_uri)

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            settings.model_registry_name,
            stages=[settings.model_stage],
        )
        _model_version = versions[0].version if versions else "unknown"
        logger.info(f"Model loaded successfully — version: {_model_version}")

    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        logger.warning("API running in DEGRADED mode — /predict will return 503")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness check",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    """
    Returns API health status.
    - status = 'healthy' if model is loaded.
    - status = 'degraded' if model failed to load at startup.
    """
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_version=_model_version,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict machine failure in next 24h",
    tags=["Predictions"],
)
async def predict(req: PredictionRequest) -> PredictionResponse:
    """
    Scores a single machine at a single point in time.

    The caller is responsible for pre-computing all features
    (rolling stats, lags, deltas) before sending the request.
    The API is a pure scoring endpoint.

    Feature ordering: sorted alphabetically, matching training.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Check MLflow Registry and restart the API.",
        )

    try:
        # Exclude machine_id — not a feature, only for response traceability
        raw_features = req.model_dump(exclude={"machine_id"})

        # Sort keys alphabetically — MUST match training column order
        feature_vector = np.array(
            [raw_features[k] for k in sorted(raw_features.keys())]
        ).reshape(1, -1)

        raw_output = _model.predict(feature_vector)

        # Handle both numpy arrays and pandas Series
        prob = float(
            raw_output.values[0]
            if hasattr(raw_output, "values")
            else raw_output[0]
        )

        prediction = 1 if prob >= _THRESHOLD else 0

        return PredictionResponse(
            machine_id=req.machine_id,
            failure_probability=round(prob, 4),
            prediction=prediction,
            prediction_label="FAILURE EXPECTED" if prediction == 1 else "NORMAL OPERATION",
            prediction_window_hours=settings.prediction_window_hours,
            model_version=_model_version,
            threshold_used=_THRESHOLD,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Prediction error for machine {req.machine_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))