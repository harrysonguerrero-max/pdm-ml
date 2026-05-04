"""
config.py
=========
Centralised configuration using Pydantic Settings.

Design decision:
    All pipeline parameters live here — no magic numbers scattered
    across modules. Every module receives a Config object, never
    reads environment variables directly.
"""

from pathlib import Path
from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """Training pipeline configuration."""

    # Paths
    data_dir:   Path = Field(default=Path("data/raw"))
    output_dir: Path = Field(default=Path("data/processed"))

    # MLflow
    experiment_name:  str   = Field(default="pdm-predictive-maintenance")
    model_name:       str   = Field(default="pdm-failure-predictor")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")

    # Labeling
    window_hours: int = Field(default=24, ge=1, le=168,
                              description="Hours ahead to predict failure")

    # Feature engineering
    rolling_windows: list[int] = Field(default=[3, 24],
                                       description="Rolling window sizes in hours")
    lag_hours:       list[int] = Field(default=[1, 2, 3],
                                       description="Lag feature offsets in hours")

    # Temporal split
    cutoff_date: str = Field(default="2015-10-01",
                             description="ISO date — train < cutoff, test >= cutoff")

    # Model hyperparameters
    model_params: dict = Field(default_factory=lambda: {
        "n_estimators":     500,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "eval_metric":      "aucpr",
        "early_stopping_rounds": 30,
        "random_state":     42,
        # scale_pos_weight is computed dynamically from data — not hardcoded here
    })

    # Registry
    pr_auc_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    register_model:   bool  = Field(default=True)

    # Prediction threshold
    # 0.35 preferred over 0.5 — reduces false negatives (costly in maintenance)
    decision_threshold: float = Field(default=0.35)

    class Config:
        arbitrary_types_allowed = True