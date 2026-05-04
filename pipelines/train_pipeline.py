"""
train_pipeline.py
=================
Entry point for the end-to-end training pipeline.

Execution:
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

# ── Make src importable when running from project root ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import TrainConfig
from src.data.loader import load_raw_data, validate_schema
from src.features.engineering import build_feature_table, label_failures
from src.models.trainer import train_and_track
from src.models.evaluator import evaluate_model, log_evaluation_report


# ── Constants ────────────────────────────────────────────────────────────────
PIPELINE_NAME = "pdm-training-pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PdM ML Training Pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CSV files (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to write processed artifacts (default: data/processed)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="pdm-predictive-maintenance",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=24,
        help="Failure prediction window in hours (default: 24)",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2015-10-01",
        help="Temporal split date — train < cutoff, test >= cutoff (default: 2015-10-01)",
    )
    parser.add_argument(
        "--pr-auc-threshold",
        type=float,
        default=0.30,
        help="Minimum PR-AUC to register model in MLflow Registry (default: 0.30)",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        default=True,
        help="Register model in MLflow Model Registry if threshold is met",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Config object ────────────────────────────────────────────────────────
    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment,
        window_hours=args.window_hours,
        cutoff_date=args.cutoff_date,
        pr_auc_threshold=args.pr_auc_threshold,
        register_model=args.register,
    )

    logger.info(f"Starting pipeline: {PIPELINE_NAME}")
    logger.info(f"Config: {config.model_dump()}")

    # ── Step 1: Load raw data ────────────────────────────────────────────────
    logger.info("Step 1/6 — Loading raw data")
    tables = load_raw_data(config.data_dir)
    logger.info(
        f"  telemetry: {tables['telemetry'].shape[0]:,} rows | "
        f"  failures:  {tables['failures'].shape[0]} events | "
        f"  machines:  {tables['machines'].shape[0]}"
    )

    # ── Step 2: Validate schema ──────────────────────────────────────────────
    logger.info("Step 2/6 — Validating schema")
    validate_schema(tables)
    logger.info("  Schema OK")

    # ── Step 3: Build feature table ──────────────────────────────────────────
    logger.info("Step 3/6 — Building feature table")
    features_df = build_feature_table(tables, config)
    logger.info(f"  Feature table: {features_df.shape[0]:,} rows × {features_df.shape[1]} cols")

    # ── Step 4: Label failures ───────────────────────────────────────────────
    logger.info("Step 4/6 — Labeling failure windows")
    labeled_df = label_failures(features_df, tables["failures"], config.window_hours)
    n_pos = int(labeled_df["target"].sum())
    n_neg = labeled_df.shape[0] - n_pos
    pos_rate = n_pos / labeled_df.shape[0] * 100
    logger.info(
        f"  Labeled: {labeled_df.shape[0]:,} rows | "
        f"positives: {n_pos:,} ({pos_rate:.2f}%) | "
        f"negatives: {n_neg:,}"
    )

    # ── Step 5: Temporal train/test split ────────────────────────────────────
    logger.info(f"Step 5/6 — Temporal split at {config.cutoff_date}")
    train_df = labeled_df.filter(
        pl.col("datetime") < pl.lit(config.cutoff_date).str.to_date()
    )
    test_df = labeled_df.filter(
        pl.col("datetime") >= pl.lit(config.cutoff_date).str.to_date()
    )
    logger.info(
        f"  Train: {train_df.shape[0]:,} rows | "
        f"  Test:  {test_df.shape[0]:,} rows"
    )

    # ── Save processed splits for reference ─────────────────────────────────
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(config.output_dir / "train.parquet")
    test_df.write_parquet(config.output_dir / "test.parquet")
    logger.info(f"  Splits saved to {config.output_dir}")

    # ── Step 6: Train + MLflow tracking ─────────────────────────────────────
    logger.info("Step 6/6 — Training with MLflow tracking")
    run_id, model = train_and_track(
        train_df=train_df,
        test_df=test_df,
        config=config,
    )
    logger.info(f"  MLflow run_id: {run_id}")

    # ── Final report ─────────────────────────────────────────────────────────
    logger.info("Pipeline completed successfully")
    logger.info(f"  MLflow UI: http://localhost:5000 → Experiment: {config.experiment_name}")
    logger.info(f"  Run ID   : {run_id}")


if __name__ == "__main__":
    # ── This import needs to be here to avoid circular import issues
    import polars as pl
    main()