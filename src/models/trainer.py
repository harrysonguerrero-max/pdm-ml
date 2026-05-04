"""
Training module with full MLflow tracking and Model Registry integration.

Key decisions documented inline:

1. XGBoost chosen over LightGBM or neural nets:
   - Tabular data → tree ensembles consistently win vs DL at this scale.
   - Fast training (<5 min on CPU), interpretable with SHAP.
   - scale_pos_weight handles class imbalance directly in the loss.

2. scale_pos_weight over SMOTE:
   - SMOTE generates synthetic minority samples.
   - If applied before temporal split, synthetic samples can carry
     information from the future → leakage.
   - scale_pos_weight adjusts the gradient weighting → zero leakage risk.
   - Formula: scale_pos_weight = count(negatives) / count(positives)

3. Temporal split — NEVER random shuffle:
   - Train on past, evaluate on future. Simulates real deployment.
   - Random k-fold would produce optimistic metrics that break in prod.

4. Decision threshold = 0.35 (not 0.5):
   - In maintenance: FN (missed failure) >> FP (false alarm) in cost.
   - Lower threshold increases recall at the cost of precision.
   - Documented and tunable via environment variable.

5. MLflow Model Registry:
   - Every experiment tracked: params, metrics, artifacts.
   - Model promoted to 'Production' stage if PR-AUC >= threshold.
   - Serving loads directly from Registry — no manual file management.
"""
import json

import mlflow
import mlflow.xgboost
import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger

from src.config import settings
from src.models.evaluator import evaluate_model, plot_confusion_matrix

# Columns that must NOT be used as features
_NON_FEATURE = {"machineID", "datetime", "target"}


def temporal_train_test_split(
    df: pl.DataFrame,
    cutoff_date: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splits the dataset by a strict temporal boundary.

    Everything BEFORE cutoff_date → train.
    Everything AFTER  cutoff_date → test.

    Never use random split on time-series data.
    The model must never see the future during training.
    """
    cutoff_date = cutoff_date or settings.train_cutoff_date
    train = df.filter(pl.col("datetime") < pl.lit(cutoff_date).str.to_datetime(format="%Y-%m-%d"))
    test  = df.filter(pl.col("datetime") >= pl.lit(cutoff_date).str.to_datetime(format="%Y-%m-%d"))

    logger.info(
        f"Temporal split at {cutoff_date} — "
        f"Train: {train.shape[0]:,} rows | Test: {test.shape[0]:,} rows"
    )
    return train, test


def _feature_columns(df: pl.DataFrame) -> list[str]:
    """Returns sorted list of feature column names (excludes IDs and target)."""
    return sorted([c for c in df.columns if c not in _NON_FEATURE])


def train_xgboost(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    params: dict | None = None,
) -> tuple[xgb.XGBClassifier, dict, str]:
    """
    Trains an XGBoost classifier and logs everything to MLflow.

    Returns:
        (trained_model, metrics_dict, mlflow_run_id)
    """
    feat_cols = _feature_columns(train_df)

    X_train = train_df.select(feat_cols).to_numpy()
    y_train = train_df["target"].to_numpy()
    X_test  = test_df.select(feat_cols).to_numpy()
    y_test  = test_df["target"].to_numpy()

    # Compute scale_pos_weight from training set only (never test set)
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw   = round(n_neg / max(n_pos, 1), 2)
    logger.info(f"Class balance — negatives: {n_neg:,} | positives: {n_pos:,} | scale_pos_weight: {spw}")

    base_params = {
        "n_estimators":      300,
        "max_depth":         6,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "scale_pos_weight":  spw,
        "eval_metric":       "aucpr",   # optimise for PR-AUC during training
        "random_state":      42,
        "n_jobs":            -1,
        "verbosity":         0,
    }
    final_params = {**base_params, **(params or {})}

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="xgboost-temporal-split") as run:
        run_id = run.info.run_id

        # Log all params
        mlflow.log_params(final_params)
        mlflow.log_param("n_features",    len(feat_cols))
        mlflow.log_param("train_rows",    len(X_train))
        mlflow.log_param("test_rows",     len(X_test))
        mlflow.log_param("cutoff_date",   settings.train_cutoff_date)
        mlflow.log_param("feature_names", json.dumps(feat_cols))

        # Train
        model = xgb.XGBClassifier(**final_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate and log metrics
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Log artifacts
        cm_path = plot_confusion_matrix(model, X_test, y_test)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # Log the model itself
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=X_test[:1],
        )

        logger.info(
            f"Run {run_id} complete — "
            f"PR-AUC={metrics['pr_auc']} | "
            f"F2={metrics['f2_score']} | "
            f"Recall={metrics['recall']}"
        )

    return model, metrics, run_id


def promote_model(run_id: str, pr_auc: float) -> bool:
    """
    Registers the model in MLflow Model Registry.
    Promotes to 'Production' stage only if PR-AUC >= threshold.

    Why threshold-gated promotion?
    Prevents accidentally deploying a degraded model if the pipeline
    is re-run with bad data or wrong features.

    Returns:
        True if promoted, False if threshold not met.
    """
    threshold = settings.promotion_pr_auc_threshold

    if pr_auc < threshold:
        logger.warning(
            f"Model NOT promoted — PR-AUC {pr_auc:.4f} < threshold {threshold:.4f}"
        )
        return False

    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=settings.model_registry_name,
    )

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=settings.model_registry_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,  # auto-archive previous Production version
    )

    logger.info(
        f"Model promoted to Production — "
        f"name={settings.model_registry_name} | version={result.version}"
    )
    return True