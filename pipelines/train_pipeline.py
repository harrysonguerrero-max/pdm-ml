"""
train_pipeline.py
=================
Entry point for the end-to-end training pipeline.

Usage:
    python pipelines/train_pipeline.py
    python pipelines/train_pipeline.py --config configs/train_config.yaml  (optional)

What this script does:
    1. Load raw CSVs from data/raw/
    2. Validate schema
    3. Build feature table (rolling stats, lags, error counts, maint age)
    4. Label rows with failure window
    5. Temporal train/test split
    6. Train XGBoost with MLflow tracking
    7. Evaluate on test set (PR-AUC, F2, precision, recall)
    8. Register model in MLflow Model Registry if metrics pass threshold
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from loguru import logger

from src.config import settings
from src.data.loader import load_all
from src.features.engineering import build_feature_table, label_failures
from src.models.trainer import train_and_track


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PdM ML Training Pipeline")
    parser.add_argument(
        "--cutoff",
        type=str,
        default=settings.train_cutoff_date,
        help=f"Temporal split date (default: {settings.train_cutoff_date})",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=settings.prediction_window_hours,
        help=f"Failure prediction window in hours (default: {settings.prediction_window_hours})",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        default=False,
        help="Skip MLflow Model Registry promotion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Permite override de cutoff y window desde CLI sin tocar .env
    cutoff       = args.cutoff
    window_hours = args.window_hours
    register     = not args.no_register

    logger.info("=" * 55)
    logger.info("  PDM TRAINING PIPELINE")
    logger.info("=" * 55)
    logger.info(f"  Cutoff date   : {cutoff}")
    logger.info(f"  Window hours  : {window_hours}h")
    logger.info(f"  Register model: {register}")
    logger.info(f"  MLflow URI    : {settings.mlflow_tracking_uri}")
    logger.info("=" * 55)

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    logger.info("Step 1/6 — Loading raw data")
    tables = load_all(settings.data_raw_path)
    # Schema validation ya ocurre dentro de load_all() — no se duplica

    # ── Step 2: Log data stats ────────────────────────────────────────────────
    logger.info("Step 2/6 — Data summary")
    tel = tables["telemetry"]
    n_machines   = tel["machineID"].n_unique()
    span_days = tel.select(
        (pl.col("datetime").max() - pl.col("datetime").min()).dt.total_days()
    ).item()
    logger.info(f"  telemetry : {tel.shape[0]:,} rows | {n_machines} machines | {span_days} days")
    logger.info(f"  failures  : {tables['failures'].shape[0]} events")
    logger.info(f"  errors    : {tables['errors'].shape[0]:,} events")

    # ── Step 3: Feature engineering ──────────────────────────────────────────
    logger.info("Step 3/6 — Feature engineering")
    features_df = build_feature_table(tables, window_hours=window_hours)
    logger.info(f"  Shape: {features_df.shape[0]:,} rows × {features_df.shape[1]} cols")

    # ── Step 4: Label failures ────────────────────────────────────────────────
    logger.info("Step 4/6 — Labeling failure windows")
    labeled_df = label_failures(
        features_df,
        tables["failures"],
        window_hours=window_hours,
    )
    n_pos      = int(labeled_df["target"].sum())
    n_tot      = labeled_df.shape[0]
    pos_rate   = n_pos / n_tot * 100
    logger.info(f"  Positives : {n_pos:,} ({pos_rate:.2f}%)")
    logger.info(f"  Negatives : {n_tot - n_pos:,}")
    logger.info(f"  scale_pos_weight ≈ {round((n_tot - n_pos) / max(n_pos, 1), 1)}")

    # ── Step 5: Temporal split ────────────────────────────────────────────────
    logger.info(f"Step 5/6 — Temporal split at {cutoff}")
    cutoff_dt = pl.lit(cutoff).str.to_date()
    train_df  = labeled_df.filter(pl.col("datetime").cast(pl.Date) < cutoff_dt)
    test_df   = labeled_df.filter(pl.col("datetime").cast(pl.Date) >= cutoff_dt)

    logger.info(f"  Train : {train_df.shape[0]:,} rows")
    logger.info(f"  Test  : {test_df.shape[0]:,} rows")

    # Guardar splits como parquet para referencia / reproducibilidad
    settings.data_processed_path.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(settings.data_processed_path / "train.parquet")
    test_df.write_parquet(settings.data_processed_path / "test.parquet")
    logger.info(f"  Splits saved → {settings.data_processed_path}")

    # ── Step 6: Train + MLflow ────────────────────────────────────────────────
    logger.info("Step 6/6 — Training with MLflow tracking")
    run_id = train_and_track(
        train_df=train_df,
        test_df=test_df,
        n_positives=n_pos,
        n_negatives=n_tot - n_pos,
        register=register,
    )

    logger.info("=" * 55)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Run ID   : {run_id}")
    logger.info(f"  MLflow   : {settings.mlflow_tracking_uri}")
    logger.info(f"  Experiment: {settings.mlflow_experiment_name}")
    logger.info("=" * 55)


if __name__ == "__main__":
    try:
        main()
        logger.info("Pipeline completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
