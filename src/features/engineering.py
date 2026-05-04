"""
Feature engineering: rolling stats, lags, and rate-of-change.
All features are BACKWARD-LOOKING — zero data leakage.

Responsibilities:
    This module ONLY handles temporal telemetry features:
        - rolling mean/std (3h, 24h windows)
        - lag values (1h, 2h, 3h)
        - rate of change (delta)

    Everything else (error counts, component ages, target, machine
    metadata) lives in src.data.preprocessor — single responsibility.

New columns added by build_all_features (~32 total):
  rolling mean/std per sensor (3h, 24h) = 16
  lag values per sensor (1h, 2h, 3h)   = 12
  rate of change (delta) per sensor     =  4
"""
import polars as pl
from loguru import logger

_SENSORS = ["volt", "rotate", "pressure", "vibration"]
_WINDOWS = [3, 24]
_LAGS    = [1, 2, 3]


def build_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Rolling mean and std per sensor per machine (3h and 24h windows)."""
    df = df.sort(["machineID", "datetime"])
    exprs = []
    for s in _SENSORS:
        for w in _WINDOWS:
            exprs += [
                pl.col(s)
                  .rolling_mean(window_size=w)
                  .over("machineID")
                  .alias(f"{s}_mean_{w}h"),
                pl.col(s)
                  .rolling_std(window_size=w)
                  .over("machineID")
                  .fill_null(0.0)
                  .alias(f"{s}_std_{w}h"),
            ]
    df = df.with_columns(exprs)
    logger.info(f"Rolling features añadidas: {len(exprs)} columnas")
    return df


def build_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Lag values 1h, 2h, 3h per sensor per machine.
    forward_fill + backward_fill handles NaN at series boundaries.
    """
    exprs = [
        pl.col(s)
          .shift(lag)
          .over("machineID")
          .forward_fill()
          .backward_fill()
          .alias(f"{s}_lag{lag}")
        for s in _SENSORS
        for lag in _LAGS
    ]
    df = df.with_columns(exprs)
    logger.info(f"Lag features añadidas: {len(exprs)} columnas")
    return df


def build_rate_of_change(df: pl.DataFrame) -> pl.DataFrame:
    """
    First-order delta (value[t] - value[t-1]) per sensor.
    Abrupt changes often immediately precede failures.
    """
    exprs = [
        (pl.col(s) - pl.col(s).shift(1).over("machineID"))
          .fill_null(0.0)
          .alias(f"{s}_delta")
        for s in _SENSORS
    ]
    df = df.with_columns(exprs)
    logger.info(f"Delta features añadidas: {len(exprs)} columnas")
    return df


def build_all_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Applies rolling, lag, and delta features in correct order.
    Input must already be sorted by (machineID, datetime).
    Called by build_feature_table after preprocessor runs.
    """
    logger.info("=== Feature engineering (telemetry) ===")
    df = df.sort(["machineID", "datetime"])
    df = build_rolling_features(df)
    df = build_lag_features(df)
    df = build_rate_of_change(df)
    logger.info(f"Total columnas tras feature engineering: {df.shape[1]}")
    return df


# ── Public API — importada por train_pipeline.py ──────────────────────────

def build_feature_table(
    tables: dict[str, pl.DataFrame],
    window_hours: int = 24,
) -> pl.DataFrame:
    """
    Full pipeline: preprocessor → telemetry features.

    Step 1 — preprocessor.build_preprocessed_table():
        joins all tables, error counts, component ages,
        target label, model_id, age.

    Step 2 — build_all_features():
        rolling stats, lags, rate-of-change on top of step 1.

    Why this separation?
        preprocessor = joins and business logic (testable in isolation).
        engineering  = pure signal extraction from telemetry time series.
        Each has its own test file and single responsibility.

    Args:
        tables:       Dict with keys: telemetry, errors, maint,
                      machines, failures.
        window_hours: Passed to preprocessor for target labeling.
                      Default matches settings.prediction_window_hours.

    Returns:
        DataFrame with all features INCLUDING target column.
    """
    from src.data.preprocessor import build_preprocessed_table

    logger.info("=== Full feature pipeline ===")

    # Step 1 — preprocessor handles everything except telemetry features
    df = build_preprocessed_table(tables)

    # Step 2 — add rolling, lag, delta on top
    df = build_all_features(df)

    logger.info(f"Final feature table: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def label_failures(
    features_df: pl.DataFrame,
    failures: pl.DataFrame,
    window_hours: int = 24,
) -> pl.DataFrame:
    """
    Called by train_pipeline.py after build_feature_table().
    Target is already included by preprocessor — returns as-is.
    Kept for pipeline interface compatibility.
    """
    if "target" in features_df.columns:
        logger.info("Target already present (from preprocessor) — skipping")
        return features_df

    # Fallback si se llama sin haber pasado por build_feature_table
    from src.data.preprocessor import _add_target
    logger.warning("Target not found — running _add_target as fallback")
    return _add_target(features_df, failures, window_hours)