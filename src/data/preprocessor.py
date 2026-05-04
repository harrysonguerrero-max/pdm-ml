"""
Preprocessing: joins all tables, creates error counts,
component age features, and the binary target label.

Target definition:
    failure_in_24h = 1 if machine fails within next 24 hours, else 0.
    Built by joining failures table on (machineID, datetime window).

Why 24h window?
    Operationally useful: gives maintenance teams time to intervene.
    Narrower windows increase precision but reduce recall.
"""
import polars as pl
from loguru import logger
from src.config import settings


def _add_error_counts(telemetry: pl.DataFrame, errors: pl.DataFrame) -> pl.DataFrame:
    """Count each error type in past 24h per machine per hour."""
    error_ids = errors["errorID"].unique().sort().to_list()
    result = telemetry
    for eid in error_ids:
        e = errors.filter(pl.col("errorID") == eid).select(["datetime", "machineID"])
        col = eid.replace("error", "error") + "_count"
        joined = (
            result
            .join(e.rename({"datetime": "err_dt"}), on="machineID", how="left")
            .filter(
                (pl.col("err_dt") <= pl.col("datetime")) &
                (pl.col("err_dt") > pl.col("datetime") - pl.duration(hours=24))
            )
            .group_by(["machineID", "datetime"])
            .agg(pl.len().alias(col))
        )
        result = result.join(joined, on=["machineID", "datetime"], how="left")
        result = result.with_columns(pl.col(col).fill_null(0).cast(pl.Int32))
    logger.info(f"Error count cols: {error_ids}")
    return result


def _add_component_ages(telemetry: pl.DataFrame, maint: pl.DataFrame) -> pl.DataFrame:
    """Hours since last replacement per component per machine."""
    comps = maint["comp"].unique().sort().to_list()
    result = telemetry
    for comp in comps:
        col = f"hours_since_{comp}"
        m = maint.filter(pl.col("comp") == comp).select(["datetime", "machineID"])
        joined = (
            result
            .join(m.rename({"datetime": "maint_dt"}), on="machineID", how="left")
            .filter(pl.col("maint_dt") <= pl.col("datetime"))
            .with_columns(
                ((pl.col("datetime") - pl.col("maint_dt")).dt.total_hours()).alias(col)
            )
            .group_by(["machineID", "datetime"])
            .agg(pl.col(col).min())
        )
        result = result.join(joined, on=["machineID", "datetime"], how="left")
        result = result.with_columns(pl.col(col).fill_null(0).cast(pl.Int32))
    logger.info(f"Component age cols: {comps}")
    return result


def _add_target(telemetry: pl.DataFrame, failures: pl.DataFrame, window_hours: int) -> pl.DataFrame:
    """Binary target: 1 if machine fails in next N hours."""
    f = failures.select(["machineID", "datetime"]).rename({"datetime": "fail_dt"})
    joined = (
        telemetry
        .join(f, on="machineID", how="left")
        .filter(
            (pl.col("fail_dt") > pl.col("datetime")) &
            (pl.col("fail_dt") <= pl.col("datetime") + pl.duration(hours=window_hours))
        )
        .group_by(["machineID", "datetime"])
        .agg(pl.len().alias("_has_failure"))
        .with_columns(pl.lit(1).cast(pl.Int8).alias("target"))
        .select(["machineID", "datetime", "target"])
    )
    result = telemetry.join(joined, on=["machineID", "datetime"], how="left")
    result = result.with_columns(pl.col("target").fill_null(0).cast(pl.Int8))
    pos = result["target"].sum()
    logger.info(f"Target: {pos:,} positives / {result.shape[0]:,} rows ({pos/result.shape[0]*100:.2f}%)")
    return result


def _add_machine_metadata(df: pl.DataFrame, machines: pl.DataFrame) -> pl.DataFrame:
    """Join model type and age (static per machine)."""
    meta = (
        machines
        .with_columns(
            pl.col("model").str.extract(r"(\d+)", 0).cast(pl.Int32).alias("model_id")
        )
        .select(["machineID", "model_id", "age"])
    )
    return df.join(meta, on="machineID", how="left")


def build_preprocessed_table(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Joins all tables into a single ML-ready feature table."""
    logger.info("=== Preprocessing ===")
    df = tables["telemetry"].sort(["machineID", "datetime"])
    df = _add_machine_metadata(df, tables["machines"])
    df = _add_error_counts(df, tables["errors"])
    df = _add_component_ages(df, tables["maint"])
    df = _add_target(df, tables["failures"], settings.prediction_window_hours)
    logger.info(f"Preprocessed table shape: {df.shape}")
    return df
