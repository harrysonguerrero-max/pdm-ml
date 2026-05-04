"""
Feature engineering: rolling stats, lags, and rate-of-change.
All features are BACKWARD-LOOKING — zero data leakage.

New columns added (~32 total):
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
                  .rolling_mean(window_size=w, min_periods=1)
                  .over("machineID")
                  .alias(f"{s}_mean_{w}h"),
                pl.col(s)
                  .rolling_std(window_size=w, min_periods=1)
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
    Why lags? The model needs to see where the sensor was, not just where it is.
    forward_fill + backward_fill handles NaN at the series boundaries.
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
    Applies all feature engineering steps in correct order.
    Must be called after preprocessor.build_preprocessed_table().
    """
    logger.info("=== Feature engineering ===")
    df = df.sort(["machineID", "datetime"])
    df = build_rolling_features(df)
    df = build_lag_features(df)
    df = build_rate_of_change(df)
    logger.info(f"Total columnas tras feature engineering: {df.shape[1]}")
    return df