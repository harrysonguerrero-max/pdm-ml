"""
Data loading and schema validation.
Uses Polars for fast CSV reading.
Validates required columns at load time — fail fast, fail loud.
"""

from pathlib import Path

import polars as pl
from loguru import logger

from src.config import settings

_REQUIRED: dict[str, list[str]] = {
    "telemetry": ["datetime", "machineID", "volt", "rotate", "pressure", "vibration"],
    "errors": ["datetime", "machineID", "errorID"],
    "maint": ["datetime", "machineID", "comp"],
    "failures": ["datetime", "machineID", "failure"],
    "machines": ["machineID", "model", "age"],
}

_FILES: dict[str, str] = {
    "telemetry": "PdM_telemetry.csv",
    "errors": "PdM_errors.csv",
    "maint": "PdM_maint.csv",
    "failures": "PdM_failures.csv",
    "machines": "PdM_machines.csv",
}


def _validate_columns(df: pl.DataFrame, name: str) -> None:
    missing = [c for c in _REQUIRED[name] if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] columnas faltantes: {missing}")


def _parse_datetime(df: pl.DataFrame) -> pl.DataFrame:
    if df.schema.get("datetime") == pl.Utf8:
        df = df.with_columns(pl.col("datetime").str.to_datetime(format="%Y-%m-%d %H:%M:%S"))
    return df


def load_telemetry(raw_path: Path | None = None) -> pl.DataFrame:
    raw_path = raw_path or settings.data_raw_path
    path = raw_path / _FILES["telemetry"]
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    df = _parse_datetime(pl.read_csv(path, try_parse_dates=False))
    _validate_columns(df, "telemetry")
    logger.info(f"Telemetry: {df.shape}")
    return df


def _load(name: str, raw_path: Path | None = None) -> pl.DataFrame:
    raw_path = raw_path or settings.data_raw_path
    path = raw_path / _FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    df = pl.read_csv(path)
    if "datetime" in df.columns:
        df = _parse_datetime(df)
    _validate_columns(df, name)
    logger.info(f"{name}: {df.shape}")
    return df


def load_all(raw_path: Path | None = None) -> dict[str, pl.DataFrame]:
    logger.info("=== Loading all datasets ===")
    return {
        "telemetry": load_telemetry(raw_path),
        "errors": _load("errors", raw_path),
        "maint": _load("maint", raw_path),
        "failures": _load("failures", raw_path),
        "machines": _load("machines", raw_path),
    }
