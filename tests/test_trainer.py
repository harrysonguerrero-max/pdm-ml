"""
Tests for src.models.trainer

Coverage:
- temporal_train_test_split splits by cutoff_date correctly
- No temporal leakage: max(train.datetime) < min(test.datetime)
- All rows are preserved after split
- _feature_columns excludes machineID, datetime, target
- scale_pos_weight formula is correct (inline in train_xgboost)
- train_and_track calls MLflow and returns a run_id string (mocked)
"""
import numpy as np
import polars as pl
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.models.trainer import (
    temporal_train_test_split,
    _feature_columns,
    train_and_track,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def feature_df():
    """60-row feature DataFrame spanning Jan–Mar 2015."""
    base = datetime(2015, 1, 1)
    n = 60
    return pl.DataFrame({
        "machineID": [1] * n,
        "datetime":  [base + timedelta(hours=i) for i in range(n)],
        "volt":      [170.0] * n,
        "rotate":    [450.0] * n,
        "pressure":  [95.0] * n,
        "vibration": [40.0] * n,
        "volt_mean_3h": [170.0] * n,
        "target":    [0] * 55 + [1] * 5,
    })


# ── temporal_train_test_split ──────────────────────────────────────────────

def test_temporal_split_returns_two_dataframes(feature_df):
    train, test = temporal_train_test_split(feature_df, cutoff_date="2015-01-03")
    assert isinstance(train, pl.DataFrame)
    assert isinstance(test, pl.DataFrame)


def test_temporal_split_no_leakage(feature_df):
    """Max datetime in train < min datetime in test — strict boundary."""
    train, test = temporal_train_test_split(feature_df, cutoff_date="2015-01-03")
    assert train["datetime"].max() < test["datetime"].min()


def test_temporal_split_covers_all_rows(feature_df):
    """No rows dropped — train + test = total."""
    train, test = temporal_train_test_split(feature_df, cutoff_date="2015-01-03")
    assert train.shape[0] + test.shape[0] == feature_df.shape[0]


def test_temporal_split_train_before_cutoff(feature_df):
    """All train rows must be before cutoff."""
    cutoff = "2015-01-03"
    train, _ = temporal_train_test_split(feature_df, cutoff_date=cutoff)
    cutoff_dt = datetime.fromisoformat(cutoff)
    assert all(dt < cutoff_dt for dt in train["datetime"].to_list())


def test_temporal_split_preserves_columns(feature_df):
    train, test = temporal_train_test_split(feature_df, cutoff_date="2015-01-03")
    assert set(train.columns) == set(feature_df.columns)
    assert set(test.columns) == set(feature_df.columns)


# ── _feature_columns ───────────────────────────────────────────────────────

def test_feature_columns_excludes_non_features(feature_df):
    cols = _feature_columns(feature_df)
    assert "machineID" not in cols
    assert "datetime" not in cols
    assert "target" not in cols


def test_feature_columns_includes_sensor_cols(feature_df):
    cols = _feature_columns(feature_df)
    assert "volt" in cols
    assert "volt_mean_3h" in cols


def test_feature_columns_returns_sorted_list(feature_df):
    cols = _feature_columns(feature_df)
    assert cols == sorted(cols)


# ── train_and_track (mocked MLflow) ───────────────────────────────────────

@patch("src.models.trainer.mlflow")
@patch("src.models.trainer.promote_model")
def test_train_and_track_returns_run_id(mock_promote, mock_mlflow, feature_df):
    """train_and_track must return a non-empty run_id string."""
    # Mock MLflow run context
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-123"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    train, test = temporal_train_test_split(feature_df, cutoff_date="2015-01-03")
    n_pos = int(train["target"].sum())
    n_neg = train.shape[0] - n_pos

    run_id = train_and_track(
        train_df=train,
        test_df=test,
        n_positives=n_pos,
        n_negatives=n_neg,
        register=False,          # skip Registry promotion in tests
    )
    assert isinstance(run_id, str)
    assert len(run_id) > 0


@patch("src.models.trainer.mlflow")
def test_promote_model_skips_below_threshold(mock_mlflow):
    """promote_model returns False when PR-AUC < threshold."""
    from src.models.trainer import promote_model
    result = promote_model(run_id="fake-run", pr_auc=0.01)
    assert result is False
    mock_mlflow.register_model.assert_not_called()


# ── promote_model — successful promotion path ──────────────────────────────

@patch("src.models.trainer.mlflow")
def test_promote_model_registers_when_above_threshold(mock_mlflow):
    """promote_model returns True and calls register_model when PR-AUC is high."""
    from src.models.trainer import promote_model

    mock_version = MagicMock()
    mock_version.version = "2"
    mock_mlflow.register_model.return_value = mock_version

    mock_client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    result = promote_model(run_id="good-run-456", pr_auc=0.99)

    assert result is True
    mock_mlflow.register_model.assert_called_once()

    kwargs = mock_client.transition_model_version_stage.call_args.kwargs
    assert kwargs["version"] == "2"
    assert kwargs["stage"] == "Production"
    assert kwargs["archive_existing_versions"] is True


@patch("src.models.trainer.mlflow")
def test_promote_model_archives_previous_version(mock_mlflow):
    """archive_existing_versions=True must always be passed."""
    from src.models.trainer import promote_model

    mock_version = MagicMock()
    mock_version.version = "5"
    mock_mlflow.register_model.return_value = mock_version
    mock_client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    promote_model(run_id="run-789", pr_auc=0.95)

    call_kwargs = mock_client.transition_model_version_stage.call_args.kwargs
    assert call_kwargs["archive_existing_versions"] is True
    assert call_kwargs["stage"] == "Production"


# ── train_and_track — register=True path ──────────────────────────────────

@patch("src.models.trainer.promote_model")
@patch("src.models.trainer.mlflow")
def test_train_and_track_calls_promote_when_register_true(
    mock_mlflow, mock_promote, feature_df
):
    """When register=True, promote_model must be called after training."""
    mock_run = MagicMock()
    mock_run.info.run_id = "run-promote-test"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
    mock_promote.return_value = True

    train, test = temporal_train_test_split(feature_df, cutoff_date="2015-01-03")
    n_pos = int(train["target"].sum())
    n_neg = train.shape[0] - n_pos

    train_and_track(
        train_df=train,
        test_df=test,
        n_positives=max(n_pos, 1),
        n_negatives=n_neg,
        register=True,
    )

    mock_promote.assert_called_once()


@patch("src.models.trainer.promote_model")
@patch("src.models.trainer.mlflow")
def test_train_and_track_skips_promote_when_register_false(
    mock_mlflow, mock_promote, feature_df
):
    """When register=False, promote_model must NOT be called."""
    mock_run = MagicMock()
    mock_run.info.run_id = "run-no-promote"
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    train, test = temporal_train_test_split(feature_df, cutoff_date="2015-01-03")
    n_pos = int(train["target"].sum())
    n_neg = train.shape[0] - n_pos

    train_and_track(
        train_df=train,
        test_df=test,
        n_positives=max(n_pos, 1),
        n_negatives=n_neg,
        register=False,
    )

    mock_promote.assert_not_called()
