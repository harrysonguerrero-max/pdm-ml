"""
Shared pytest fixtures — pure in-memory DataFrames.
No real CSV files needed to run the test suite.

Design decision:
    Fixtures are minimal but realistic enough to exercise
    the actual logic. They use real column names and types
    matching the Kaggle dataset schema.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest


def _datetimes(start: str, n: int) -> list[datetime]:
    base = datetime.fromisoformat(start)
    return [base + timedelta(hours=i) for i in range(n)]


@pytest.fixture
def telemetry_df() -> pl.DataFrame:
    """10 hourly rows for 2 machines — minimal realistic telemetry."""
    dts = _datetimes("2015-01-01 00:00:00", 10)
    rows = [
        {
            "datetime": dt,
            "machineID": mid,
            "volt": 170.0 + mid,
            "rotate": 450.0 + mid,
            "pressure": 95.0 + mid,
            "vibration": 40.0 + mid,
        }
        for mid in [1, 2]
        for dt in dts
    ]
    return pl.DataFrame(rows)


@pytest.fixture
def machines_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "machineID": [1, 2],
            "model": ["model1", "model2"],
            "age": [5, 10],
        }
    )


@pytest.fixture
def errors_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": _datetimes("2015-01-01 02:00:00", 3),
            "machineID": [1, 1, 2],
            "errorID": ["error1", "error2", "error1"],
        }
    )


@pytest.fixture
def maint_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": _datetimes("2015-01-01 00:00:00", 4),
            "machineID": [1, 1, 2, 2],
            "comp": ["comp1", "comp2", "comp1", "comp3"],
        }
    )


@pytest.fixture
def failures_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": _datetimes("2015-01-01 05:00:00", 2),
            "machineID": [1, 2],
            "failure": ["comp1", "comp2"],
        }
    )


@pytest.fixture
def all_tables(telemetry_df, machines_df, errors_df, maint_df, failures_df) -> dict:
    """Convenience fixture: all 5 tables in a single dict."""
    return {
        "telemetry": telemetry_df,
        "machines": machines_df,
        "errors": errors_df,
        "maint": maint_df,
        "failures": failures_df,
    }


@pytest.fixture
def base_feature_df() -> pl.DataFrame:
    """
    30-row single-machine DataFrame for testing feature engineering.
    Must have at least 24 rows to test 24h rolling windows without all-nulls.
    """
    base = datetime(2015, 1, 1)
    import random

    random.seed(42)
    return pl.DataFrame(
        {
            "machineID": [1] * 30,
            "datetime": [base + timedelta(hours=i) for i in range(30)],
            "volt": [170.0 + random.uniform(-2, 2) for _ in range(30)],
            "rotate": [450.0 + random.uniform(-5, 5) for _ in range(30)],
            "pressure": [95.0 + random.uniform(-1, 1) for _ in range(30)],
            "vibration": [40.0 + random.uniform(-1, 1) for _ in range(30)],
        }
    )
