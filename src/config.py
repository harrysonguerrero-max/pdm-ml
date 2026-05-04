"""
Centralized configuration using Pydantic Settings.
Values are loaded from .env file — no hardcoded credentials.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # MLflow
    mlflow_tracking_uri:    str  = "http://localhost:5000"
    mlflow_experiment_name: str  = "predictive-maintenance"
    model_registry_name:    str  = "pdm-xgboost"
    model_stage:            str  = "Production"

    # Paths
    data_raw_path:       Path = Path("data/raw")
    data_processed_path: Path = Path("data/processed")

    # Training
    train_cutoff_date:          str   = "2015-10-01"
    prediction_window_hours:    int   = 24
    promotion_pr_auc_threshold: float = 0.50

    # 🆕 Feature engineering — usados en engineering.py
    rolling_windows: list[int] = [3, 24]
    lag_hours:       list[int] = [1, 2, 3]

    # 🆕 Model hyperparameters — usados en trainer.py
    xgb_n_estimators:        int   = 500
    xgb_max_depth:           int   = 6
    xgb_learning_rate:       float = 0.05
    xgb_subsample:           float = 0.8
    xgb_colsample_bytree:    float = 0.8
    xgb_min_child_weight:    int   = 10
    xgb_early_stopping:      int   = 30

    # 🆕 Serving
    # 0.35 preferido sobre 0.5 — reduce falsos negativos (costosos en mantenimiento)
    decision_threshold: float = 0.35


settings = Settings()
