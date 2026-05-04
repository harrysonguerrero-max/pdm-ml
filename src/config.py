"""
Centralized configuration using Pydantic Settings.
Values are loaded from .env file — no hardcoded credentials.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    mlflow_tracking_uri:    str  = "http://localhost:5000"
    mlflow_experiment_name: str  = "predictive-maintenance"
    model_registry_name:    str  = "pdm-xgboost"
    model_stage:            str  = "Production"

    data_raw_path:       Path = Path("data/raw")
    data_processed_path: Path = Path("data/processed")

    train_cutoff_date:          str   = "2015-10-01"
    prediction_window_hours:    int   = 24
    promotion_pr_auc_threshold: float = 0.50


settings = Settings()
