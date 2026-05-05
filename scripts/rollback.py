"""
MLflow Model Registry rollback utility.

Demotes the current Production model version and promotes the previous one.
Safe to run multiple times — idempotent if there is no previous version.

Usage:
    python scripts/rollback.py
    make rollback
"""
import sys
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient

from src.config import settings


def rollback() -> bool:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    name = settings.model_registry_name

    all_versions = client.search_model_versions(f"name='{name}'")
    all_versions = sorted(all_versions, key=lambda v: int(v.version))

    production = [v for v in all_versions if v.current_stage == "Production"]
    archived   = [v for v in all_versions if v.current_stage == "Archived"]

    if not production:
        logger.error("No Production model found — nothing to roll back.")
        return False

    current = production[0]
    logger.info(f"Current Production: version={current.version} | run={current.run_id}")

    if not archived:
        logger.error("No Archived versions found — cannot roll back further.")
        return False

    previous = archived[-1]  # most recent archived = last promoted before current
    logger.info(f"Rolling back to: version={previous.version} | run={previous.run_id}")

    # Demote current Production → Archived
    client.transition_model_version_stage(
        name=name, version=current.version,
        stage="Archived", archive_existing_versions=False,
    )

    # Promote previous Archived → Production
    client.transition_model_version_stage(
        name=name, version=previous.version,
        stage="Production", archive_existing_versions=False,
    )

    logger.info(
        f"Rollback complete — "
        f"v{current.version} → Archived | v{previous.version} → Production"
    )
    return True


if __name__ == "__main__":
    success = rollback()
    sys.exit(0 if success else 1)
