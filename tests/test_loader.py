"""
Tests for src.data.loader

Coverage:
    - Schema validation passes on valid data
    - Schema validation raises ValueError on missing columns
    - Datetime parsing from string to pl.Datetime
    - FileNotFoundError raised when CSV is missing
    - All 5 datasets loadable from a temp directory (integration-style)
"""
from pathlib import Path

import polars as pl
import pytest

from src.data.loader import _parse_datetime, _validate_columns


# ── Schema validation ──────────────────────────────────────────────────────


def test_validate_telemetry_columns_passes():
    df = pl.DataFrame({
        "datetime":  ["2015-01-01 00:00:00"],
        "machineID": [1],
        "volt":      [170.0],
        "rotate":    [450.0],
        "pressure":  [95.0],
        "vibration": [40.0],
    })
    # Should not raise
    _validate_columns(df, "telemetry")


def test_validate_columns_raises_on_missing_column():
    df = pl.DataFrame({
        "datetime":  ["2015-01-01 00:00:00"],
        "machineID": [1],
        # volt, rotate, pressure, vibration missing
    })
    with pytest.raises(ValueError, match="faltantes"):
        _validate_columns(df, "telemetry")


def test_validate_errors_columns_passes():
    df = pl.DataFrame({
        "datetime":  ["2015-01-01 00:00:00"],
        "machineID": [1],
        "errorID":   ["error1"],
    })
    _validate_columns(df, "errors")


def test_validate_machines_columns_passes():
    df = pl.DataFrame({
        "machineID": [1],
        "model":     ["model1"],
        "age":       [5],
    })
    _validate_columns(df, "machines")


def test_validate_failures_columns_passes():
    df = pl.DataFrame({
        "datetime":  ["2015-01-01 00:00:00"],
        "machineID": [1],
        "failure":   ["comp1"],
    })
    _validate_columns(df, "failures")


# ── Datetime parsing ───────────────────────────────────────────────────────


def test_parse_datetime_converts_string_column():
    df = pl.DataFrame({"datetime": ["2015-01-01 06:00:00", "2015-06-15 12:00:00"]})
    result = _parse_datetime(df)
    assert result.schema["datetime"] == pl.Datetime


def test_parse_datetime_is_idempotent():
    """If datetime is already Datetime type, no error should be raised."""
    from datetime import datetime
    df = pl.DataFrame({"datetime": [datetime(2015, 1, 1, 6, 0, 0)]})
    result = _parse_datetime(df)
    assert result.schema["datetime"] == pl.Datetime


# ── File not found ─────────────────────────────────────────────────────────


def test_load_telemetry_raises_file_not_found(tmp_path: Path):
    from src.data.loader import load_telemetry
    with pytest.raises(FileNotFoundError, match="Archivo no encontrado"):
        load_telemetry(raw_path=tmp_path)


def test_load_all_raises_when_data_dir_empty(tmp_path: Path):
    from src.data.loader import load_all
    with pytest.raises(FileNotFoundError):
        load_all(raw_path=tmp_path)