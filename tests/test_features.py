"""
Tests for src.features.engineering

Coverage:
    - Rolling features add exactly 16 columns
    - Lag features add exactly 12 columns
    - Delta features add exactly 4 columns
    - build_all_features adds exactly 32 columns total
    - No null values remain after all feature engineering
    - Column naming convention is correct
    - Features are backward-looking (row 0 lag = filled, not null)
"""
import polars as pl

from src.features.engineering import (
    build_all_features,
    build_lag_features,
    build_rate_of_change,
    build_rolling_features,
)

# ── Rolling features ───────────────────────────────────────────────────────


def test_rolling_features_add_16_columns(base_feature_df):
    result = build_rolling_features(base_feature_df)
    new_cols = result.shape[1] - base_feature_df.shape[1]
    assert new_cols == 16, f"Expected 16 rolling columns, got {new_cols}"


def test_rolling_column_names_correct(base_feature_df):
    result = build_rolling_features(base_feature_df)
    expected = [
        "volt_mean_3h", "volt_std_3h", "volt_mean_24h", "volt_std_24h",
        "rotate_mean_3h", "rotate_std_3h",
        "vibration_mean_3h", "vibration_std_3h",
        "pressure_mean_24h", "pressure_std_24h",
    ]
    for col in expected:
        assert col in result.columns, f"Missing rolling column: {col}"


def test_rolling_no_nulls(base_feature_df):
    result = build_rolling_features(base_feature_df)
    for col in [c for c in result.columns if "mean" in c or "std" in c]:
        assert result[col].null_count() == 0, f"Null in rolling column: {col}"


# ── Lag features ───────────────────────────────────────────────────────────


def test_lag_features_add_12_columns(base_feature_df):
    result = build_lag_features(base_feature_df)
    new_cols = result.shape[1] - base_feature_df.shape[1]
    assert new_cols == 12, f"Expected 12 lag columns, got {new_cols}"


def test_lag_column_names_correct(base_feature_df):
    result = build_lag_features(base_feature_df)
    for sensor in ["volt", "rotate", "pressure", "vibration"]:
        for lag in [1, 2, 3]:
            col = f"{sensor}_lag{lag}"
            assert col in result.columns, f"Missing lag column: {col}"


def test_lag_no_nulls_after_fill(base_feature_df):
    """Boundary NaN values must be filled — no nulls allowed after engineering."""
    result = build_lag_features(base_feature_df)
    for col in [c for c in result.columns if "lag" in c]:
        assert result[col].null_count() == 0, f"Null in lag column: {col}"


# ── Delta / rate-of-change features ───────────────────────────────────────


def test_delta_features_add_4_columns(base_feature_df):
    result = build_rate_of_change(base_feature_df)
    new_cols = result.shape[1] - base_feature_df.shape[1]
    assert new_cols == 4, f"Expected 4 delta columns, got {new_cols}"


def test_delta_column_names_correct(base_feature_df):
    result = build_rate_of_change(base_feature_df)
    for sensor in ["volt", "rotate", "pressure", "vibration"]:
        assert f"{sensor}_delta" in result.columns


def test_delta_first_row_is_zero(base_feature_df):
    """First delta per machine should be 0.0 (fill_null=0.0)."""
    result = build_rate_of_change(base_feature_df)
    assert result["volt_delta"][0] == 0.0


# ── Full pipeline ──────────────────────────────────────────────────────────


def test_build_all_features_adds_32_columns(base_feature_df):
    result = build_all_features(base_feature_df)
    new_cols = result.shape[1] - base_feature_df.shape[1]
    assert new_cols == 32, f"Expected 32 new feature columns, got {new_cols}"


def test_build_all_features_no_nulls(base_feature_df):
    result = build_all_features(base_feature_df)
    feat_cols = [c for c in result.columns if c not in ("machineID", "datetime")]
    for col in feat_cols:
        assert result[col].null_count() == 0, f"Null found in feature column: {col}"


def test_build_all_features_preserves_row_count(base_feature_df):
    result = build_all_features(base_feature_df)
    assert result.shape[0] == base_feature_df.shape[0]


def test_build_all_features_sorted_by_machine_datetime(base_feature_df):
    """Output must be sorted — important for lag correctness."""
    result = build_all_features(base_feature_df)
    dts = result.filter(pl.col("machineID") == 1)["datetime"].to_list()
    assert dts == sorted(dts), "Output not sorted by datetime for machine 1"


# ── build_feature_table + label_failures ──────────────────────────────────

def test_build_feature_table_returns_dataframe(all_tables):
    """build_feature_table runs full pipeline including preprocessor."""
    from src.features.engineering import build_feature_table
    result = build_feature_table(all_tables, window_hours=24)
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] > 0
    assert "target" in result.columns


def test_build_feature_table_has_engineered_columns(all_tables):
    """build_feature_table adds rolling + lag + delta on top of preprocessor."""
    from src.features.engineering import build_feature_table
    result = build_feature_table(all_tables, window_hours=24)
    assert "volt_mean_3h" in result.columns
    assert "volt_lag1" in result.columns
    assert "volt_delta" in result.columns


def test_label_failures_skips_if_target_exists(base_feature_df):
    """label_failures returns as-is when target column already present."""
    from src.features.engineering import label_failures
    df_with_target = base_feature_df.with_columns(
        pl.lit(0).cast(pl.Int8).alias("target")
    )
    failures_dummy = pl.DataFrame({
        "datetime":  [base_feature_df["datetime"][0]],
        "machineID": [1],
        "failure":   ["comp1"],
    })
    result = label_failures(df_with_target, failures_dummy, window_hours=24)
    # Must return the same df without modification
    assert "target" in result.columns
    assert result.shape == df_with_target.shape


def test_label_failures_fallback_adds_target(base_feature_df):
    """label_failures runs _add_target when target column is missing."""
    from src.features.engineering import label_failures
    failures = pl.DataFrame({
        "datetime":  [base_feature_df["datetime"][2]],
        "machineID": [1],
        "failure":   ["comp1"],
    })
    result = label_failures(base_feature_df, failures, window_hours=24)
    assert "target" in result.columns
    unique_vals = set(result["target"].unique().to_list())
    assert unique_vals.issubset({0, 1})
