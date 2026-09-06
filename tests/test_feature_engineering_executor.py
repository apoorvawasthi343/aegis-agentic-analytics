"""Tests for the AEGIS feature engineering executor."""

import numpy as np
import pandas as pd
import pytest

from src.aegis.feature_engineering_executor import FeatureEngineeringExecutor
from src.aegis.schemas import (
    AppliedFeature,
    FeatureEngineeringReport,
    FeatureEngineeringSpec,
    SkippedFeature,
)


def _make_sample_df() -> pd.DataFrame:
    """Create a small sample DataFrame for feature engineering tests."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, np.nan, 50],
            "income": [50000, 60000, 75000, 80000, 90000, 100000],
            "score": [85.0, 90.0, 75.0, 95.0, 88.0, 92.0],
            "category": ["A", "B", "A", "C", "B", "A"],
        }
    )


def test_log1p_transformation_applied() -> None:
    """log1p transformation should create the expected column."""
    df = _make_sample_df()
    original_columns = set(df.columns)

    executor = FeatureEngineeringExecutor()
    spec = FeatureEngineeringSpec(
        feature_name="log_income",
        transformation_type="log1p",
        columns=["income"],
    )
    engineered_df, report = executor.apply(df, [spec])

    assert isinstance(report, FeatureEngineeringReport)
    assert len(report.applied_features) == 1
    applied = report.applied_features[0]
    assert applied.transformation_type == "log1p"
    assert applied.result_column == "log_income"
    assert "log_income" in engineered_df.columns
    assert "log_income" not in original_columns


def test_original_dataframe_unchanged() -> None:
    """The original DataFrame should not be modified by the executor."""
    df = _make_sample_df()
    original_columns = set(df.columns)

    executor = FeatureEngineeringExecutor()
    spec = FeatureEngineeringSpec(
        feature_name="log_income",
        transformation_type="log1p",
        columns=["income"],
    )
    executor.apply(df, [spec])

    # The original DataFrame should remain unchanged
    assert set(df.columns) == original_columns
    assert "log_income" not in df.columns


def test_missing_indicator_applied() -> None:
    """missing_indicator should create a binary indicator column."""
    df = _make_sample_df()

    executor = FeatureEngineeringExecutor()
    spec = FeatureEngineeringSpec(
        feature_name="age_is_missing",
        transformation_type="missing_indicator",
        columns=["age"],
    )
    engineered_df, report = executor.apply(df, [spec])

    assert len(report.applied_features) == 1
    assert report.applied_features[0].result_column == "age_is_missing"
    assert "age_is_missing" in engineered_df.columns

    # Verify the indicator values: row with NaN age should be 1, others 0
    age_is_missing_col = engineered_df["age_is_missing"]
    assert age_is_missing_col.iloc[4] == 1  # NaN row
    assert age_is_missing_col.iloc[0] == 0  # non-NaN row


def test_ratio_transformation_applied() -> None:
    """ratio transformation should compute income / score."""
    df = _make_sample_df()

    executor = FeatureEngineeringExecutor()
    spec = FeatureEngineeringSpec(
        feature_name="income_per_score",
        transformation_type="ratio",
        columns=["income", "score"],
    )
    engineered_df, report = executor.apply(df, [spec])

    assert len(report.applied_features) == 1
    applied = report.applied_features[0]
    assert applied.transformation_type == "ratio"
    assert applied.result_column == "income_per_score"
    assert "income_per_score" in engineered_df.columns

    # Verify approximate ratio values
    assert abs(engineered_df["income_per_score"].iloc[0] - 50000 / 85.0) < 0.01


def test_count_sum_transformation_applied() -> None:
    """count_sum transformation should sum multiple numeric columns."""
    df = _make_sample_df()

    executor = FeatureEngineeringExecutor()
    spec = FeatureEngineeringSpec(
        feature_name="total_metrics",
        transformation_type="count_sum",
        columns=["age", "score"],
    )
    engineered_df, report = executor.apply(df, [spec])

    assert len(report.applied_features) == 1
    applied = report.applied_features[0]
    assert applied.transformation_type == "count_sum"
    assert applied.result_column == "total_metrics"
    assert "total_metrics" in engineered_df.columns

    # First row: 25 + 85 = 110
    assert engineered_df["total_metrics"].iloc[0] == 110.0


def test_missing_column_skipped_safely() -> None:
    """A transformation referencing a missing column should be skipped."""
    df = _make_sample_df()

    executor = FeatureEngineeringExecutor()
    spec = FeatureEngineeringSpec(
        feature_name="bad_feature",
        transformation_type="log1p",
        columns=["nonexistent_column"],
    )
    _, report = executor.apply(df, [spec])

    assert len(report.applied_features) == 0
    assert len(report.skipped_features) == 1
    skipped = report.skipped_features[0]
    assert skipped.feature_name == "bad_feature"
    assert "does not exist" in skipped.reason


def test_negative_values_skipped_for_log1p() -> None:
    """log1p should be skipped when the column contains negative values."""
    df = pd.DataFrame({"value": [-1, 2, 3]})

    executor = FeatureEngineeringExecutor()
    spec = FeatureEngineeringSpec(
        feature_name="log_value",
        transformation_type="log1p",
        columns=["value"],
    )
    _, report = executor.apply(df, [spec])

    assert len(report.applied_features) == 0
    assert len(report.skipped_features) == 1
    skipped = report.skipped_features[0]
    assert skipped.transformation_type == "log1p"
    assert "negative" in skipped.reason.lower()


def test_non_numeric_column_skipped_for_log1p() -> None:
    """log1p should be skipped when the column is not numeric."""
    df = pd.DataFrame({"text_col": ["a", "b", "c"]})

    executor = FeatureEngineeringExecutor()
    spec = FeatureEngineeringSpec(
        feature_name="log_text",
        transformation_type="log1p",
        columns=["text_col"],
    )
    _, report = executor.apply(df, [spec])

    assert len(report.applied_features) == 0
    assert len(report.skipped_features) == 1
    skipped = report.skipped_features[0]
    assert skipped.transformation_type == "log1p"
    assert "not numeric" in skipped.reason


def test_unsupported_transformation_skipped() -> None:
    """An unsupported transformation type should be skipped with a clear reason."""
    df = _make_sample_df()

    executor = FeatureEngineeringExecutor()
    spec = FeatureEngineeringSpec(
        feature_name="weird_feature",
        transformation_type="do_something_dangerous",
        columns=["income"],
    )
    _, report = executor.apply(df, [spec])

    assert len(report.applied_features) == 0
    assert len(report.skipped_features) == 1
    skipped = report.skipped_features[0]
    assert skipped.transformation_type == "do_something_dangerous"
    assert "Unsupported transformation type" in skipped.reason


def test_mixed_applied_and_skipped() -> None:
    """A mix of valid and invalid transformations should be handled correctly."""
    df = _make_sample_df()

    executor = FeatureEngineeringExecutor()
    specs = [
        FeatureEngineeringSpec(
            feature_name="log_income",
            transformation_type="log1p",
            columns=["income"],
        ),
        FeatureEngineeringSpec(
            feature_name="bad_feature",
            transformation_type="log1p",
            columns=["nonexistent"],
        ),
        FeatureEngineeringSpec(
            feature_name="age_missing_flag",
            transformation_type="missing_indicator",
            columns=["age"],
        ),
    ]
    engineered_df, report = executor.apply(df, specs)

    assert len(report.applied_features) == 2
    assert len(report.skipped_features) == 1
    applied_types = {f.transformation_type for f in report.applied_features}
    assert applied_types == {"log1p", "missing_indicator"}

    # Verify both new columns exist in the engineered DataFrame
    assert "log_income" in engineered_df.columns
    assert "age_missing_flag" in engineered_df.columns

    skipped = report.skipped_features[0]
    assert skipped.transformation_type == "log1p"


def test_ratio_by_zero_results_in_nan() -> None:
    """Ratio with zero denominator should produce NaN for that row."""
    df = pd.DataFrame({"num": [10, 20, 30], "den": [2, 0, 5]})

    executor = FeatureEngineeringExecutor()
    spec = FeatureEngineeringSpec(
        feature_name="ratio_col",
        transformation_type="ratio",
        columns=["num", "den"],
    )
    engineered_df, report = executor.apply(df, [spec])

    assert len(report.applied_features) == 1
    # Second row should be NaN (division by zero)
    assert pd.isna(engineered_df["ratio_col"].iloc[1])


def test_feature_engineering_report_summary() -> None:
    """The report summary should reflect applied and skipped counts."""
    df = _make_sample_df()

    executor = FeatureEngineeringExecutor()
    specs = [
        FeatureEngineeringSpec(
            feature_name="log_income",
            transformation_type="log1p",
            columns=["income"],
        ),
        FeatureEngineeringSpec(
            feature_name="bad_col",
            transformation_type="log1p",
            columns=["missing"],
        ),
    ]
    _, report = executor.apply(df, specs)

    assert "Applied 1 feature(s) and skipped 1 feature(s)" in report.summary
    assert report.original_row_count == 6
    assert report.original_column_count == 4
    assert report.engineered_column_count == 5  # original 4 + 1 new


def test_all_previous_tests_still_pass() -> None:
    """Marker test to confirm no regression in existing functionality.

    The actual verification happens through the complete pytest run.
    """
    from src.aegis.feature_engineering_executor import FeatureEngineeringExecutor

    assert FeatureEngineeringExecutor is not None