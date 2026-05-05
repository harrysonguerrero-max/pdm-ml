"""
List all registered model versions with their stage and run ID.

Usage:
    python scripts/registry_list.py
    make registry-list
"""
import mlflow
from mlflow.tracking import MlflowClient
from src.config import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
client = MlflowClient()

versions = client.search_model_versions(f"name='{settings.model_registry_name}'")
versions = sorted(versions, key=lambda v: int(v.version))

for v in versions:
    print(f"v{v.version:>3} | {v.current_stage:<12} | run={v.run_id[:8]}")
