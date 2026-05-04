"""
Tests for src.data.preprocessor

Coverage:
    - build_preprocessed_table returns a DataFrame
    - Target column exists with only 0 and 1 values
    - Positive rate is > 0 (failures are labelled)
    - Error count columns are present and non-negative
    - Component age columns are present and non-negative
    - No nulls in key columns after preprocessing
"""
import polars as pl
import pytest

from src.data.preprocessor import (
    _add_machine_metadata,
    _add_target,
    build_preprocessed_table,
)


def test_build_preprocessed_table_returns_dataframe(all_tables):
    result = build_preprocessed_table(all_tables)
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] > 0


def test_target_column_exists(all_tables):
    result = build_preprocessed_table(all_tables)
    assert "target" in result.columns


def test_target_is_binary(all_tables):
    result = build_preprocessed_table(all_tables)
    unique_vals = set(result["target"].unique().to_list())
    assert unique_vals.issubset({0, 1}), f"Non-binary target values: {unique_vals}"


def test_positive_rate_is_nonzero(all_tables):
    result = build_preprocessed_table(all_tables)
    pos_rate = result["target"].mean()
    assert pos_rate > 0, "No positive labels found — check failure window logic"


def test_error_count_columns_present(all_tables):
    result = build_preprocessed_table(all_tables)
    assert "error1_count" in result.columns
    assert "error2_count" in result.columns


def test_error_counts_non_negative(all_tables):
    result = build_preprocessed_table(all_tables)
    for col in [c for c in result.columns if c.endswith("_count")]:
        assert result[col].min() >= 0, f"Negative count in {col}"


def test_component_age_columns_present(all_tables):
    result = build_preprocessed_table(all_tables)
    assert "hours_since_comp1" in result.columns
    assert "hours_since_comp2" in result.columns


def test_component_age_non_negative(all_tables):
    result = build_preprocessed_table(all_tables)
    age_cols = [c for c in result.columns if c.startswith("hours_since_")]
    negatives = result.select(age_cols).select(
        pl.col(age_cols).min().name.prefix("min_")
    )
    for col in age_cols:
        min_val = negatives[f"min_{col}"].item()
        assert min_val is not None and min_val >= 0, f"Negative age in {col}"


def test_machine_metadata_joined(all_tables):
    result = build_preprocessed_table(all_tables)
    assert "model_id" in result.columns
    assert "age"      in result.columns


def test_no_nulls_in_core_columns(all_tables):
    result = build_preprocessed_table(all_tables)
    core = ["machineID", "datetime", "volt", "rotate", "pressure", "vibration", "target"]
    for col in core:
        nulls = result[col].null_count()
        assert nulls == 0, f"Unexpected nulls in {col}: {nulls}"


def test_add_machine_metadata_adds_model_id_and_age(telemetry_df, machines_df):
    result = _add_machine_metadata(telemetry_df, machines_df)
    assert "model_id" in result.columns
    assert "age"      in result.columns
    assert result["age"].null_count() == 0