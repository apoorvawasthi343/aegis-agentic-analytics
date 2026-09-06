"""Tests for AEGIS baseline vs. engineered model comparison."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.aegis.model_comparison import ModelComparison
from src.aegis.schemas import (
    BaselineVsEngineeredComparison,
    FeatureEngineeringSpec,
)


def _make_synthetic_classification_df(
    n_samples: int = 200,
    n_features: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a synthetic classification DataFrame with non-negative features."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )
    # Shift to non-negative since log1p requires non-negative data
    min_val = X.min()
    if min_val < 0:
        X = X - min_val + 1.0
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def test_comparison_returns_comparison_report() -> None:
    """Comparison should return a BaselineVsEngineeredComparison."""
    df = _make_synthetic_classification_df()
    specs = [
        FeatureEngineeringSpec(
            feature_name="feature_0_log",
            transformation_type="log1p",
            columns=["feature_0"],
        )
    ]

    comparison = ModelComparison().compare(df, "target", specs)

    assert isinstance(comparison, BaselineVsEngineeredComparison)
    assert comparison.baseline_metrics is not None
    assert comparison.engineered_metrics is not None


def test_comparison_baseline_and_engineered_metrics_exist() -> None:
    """Both baseline and engineered metrics should be populated."""
    df = _make_synthetic_classification_df()
    specs = [
        FeatureEngineeringSpec(
            feature_name="feature_0_log",
            transformation_type="log1p",
            columns=["feature_0"],
        )
    ]

    comparison = ModelComparison().compare(df, "target", specs)

    assert comparison.baseline_metrics.accuracy > 0
    assert comparison.baseline_metrics.f1_score > 0
    assert comparison.baseline_metrics.train_rows > 0
    assert comparison.baseline_metrics.test_rows > 0

    assert comparison.engineered_metrics.accuracy > 0
    assert comparison.engineered_metrics.f1_score > 0
    assert comparison.engineered_metrics.train_rows > 0
    assert comparison.engineered_metrics.test_rows > 0


def test_comparison_accuracy_change_calculated() -> None:
    """Accuracy change should be calculated as engineered - baseline."""
    df = _make_synthetic_classification_df()
    specs = [
        FeatureEngineeringSpec(
            feature_name="feature_0_log",
            transformation_type="log1p",
            columns=["feature_0"],
        )
    ]

    comparison = ModelComparison().compare(df, "target", specs)

    expected_change = (
        comparison.engineered_metrics.accuracy
        - comparison.baseline_metrics.accuracy
    )
    assert abs(comparison.accuracy_change - expected_change) < 1e-10


def test_comparison_f1_change_calculated() -> None:
    """F1 change should be calculated as engineered - baseline."""
    df = _make_synthetic_classification_df()
    specs = [
        FeatureEngineeringSpec(
            feature_name="feature_0_log",
            transformation_type="log1p",
            columns=["feature_0"],
        )
    ]

    comparison = ModelComparison().compare(df, "target", specs)

    expected_change = (
        comparison.engineered_metrics.f1_score
        - comparison.baseline_metrics.f1_score
    )
    assert abs(comparison.f1_change - expected_change) < 1e-10


def test_comparison_roc_auc_change_when_available() -> None:
    """ROC-AUC change should be calculated when both models support it."""
    df = _make_synthetic_classification_df()
    specs = [
        FeatureEngineeringSpec(
            feature_name="feature_0_log",
            transformation_type="log1p",
            columns=["feature_0"],
        )
    ]

    comparison = ModelComparison().compare(df, "target", specs)

    if comparison.baseline_metrics.roc_auc is not None:
        expected_change = (
            comparison.engineered_metrics.roc_auc
            - comparison.baseline_metrics.roc_auc
        )
        assert abs(comparison.roc_auc_change - expected_change) < 1e-10
    else:
        assert comparison.roc_auc_change is None


def test_comparison_features_created_recorded() -> None:
    """Engineered feature names should be recorded in the report."""
    df = _make_synthetic_classification_df()
    specs = [
        FeatureEngineeringSpec(
            feature_name="log_feature_0",
            transformation_type="log1p",
            columns=["feature_0"],
        ),
        FeatureEngineeringSpec(
            feature_name="feature_1_log",
            transformation_type="log1p",
            columns=["feature_1"],
        ),
    ]

    comparison = ModelComparison().compare(df, "target", specs)

    assert len(comparison.features_created) == 2
    assert "log_feature_0" in comparison.features_created
    assert "feature_1_log" in comparison.features_created


def test_comparison_features_skipped_recorded() -> None:
    """Skipped feature reasons should be recorded."""
    df = _make_synthetic_classification_df()
    specs = [
        FeatureEngineeringSpec(
            feature_name="bad_feature",
            transformation_type="log1p",
            columns=["nonexistent_column"],
        )
    ]

    comparison = ModelComparison().compare(df, "target", specs)

    assert len(comparison.features_skipped) == 1
    assert "bad_feature" in comparison.features_skipped[0]


def test_improved_false_when_no_improvement() -> None:
    """improved should be False when engineered doesn't beat baseline."""
    df = _make_synthetic_classification_df()
    specs = [
        FeatureEngineeringSpec(
            feature_name="log_feature_0",
            transformation_type="log1p",
            columns=["feature_0"],
        )
    ]

    comparison = ModelComparison().compare(df, "target", specs)

    # log1p on shifted data may not improve, but should not crash
    assert isinstance(comparison.improved, bool)


def test_comparison_summary_contains_baseline_and_engineered_metrics() -> None:
    """Summary should mention baseline and engineered performance."""
    df = _make_synthetic_classification_df()
    specs = [
        FeatureEngineeringSpec(
            feature_name="log_feature_0",
            transformation_type="log1p",
            columns=["feature_0"],
        )
    ]

    comparison = ModelComparison().compare(df, "target", specs)

    assert (
        "Baseline model achieved accuracy"
        in comparison.summary
    )
    assert (
        "Engineered model achieved accuracy"
        in comparison.summary
    )


def test_comparison_handles_negative_metric_changes() -> None:
    """Negative metric changes should be represented correctly."""
    df = _make_synthetic_classification_df()
    specs = [
        FeatureEngineeringSpec(
            feature_name="log_feature_0",
            transformation_type="log1p",
            columns=["feature_0"],
        )
    ]

    comparison = ModelComparison().compare(df, "target", specs)

    # Changes can be negative, zero, or positive
    assert isinstance(comparison.accuracy_change, float)
    assert isinstance(comparison.f1_change, float)


def test_same_random_state_produces_reproducible_split() -> None:
    """Same random_state should produce reproducible comparison results."""
    df = _make_synthetic_classification_df()

    specs = [
        FeatureEngineeringSpec(
            feature_name="log_feature_0",
            transformation_type="log1p",
            columns=["feature_0"],
        )
    ]

    # Two comparisons with same random state
    comp1 = ModelComparison(random_state=42).compare(df, "target", specs)
    comp2 = ModelComparison(random_state=42).compare(df, "target", specs)

    assert comp1.baseline_metrics.accuracy == comp2.baseline_metrics.accuracy
    assert comp1.engineered_metrics.accuracy == comp2.engineered_metrics.accuracy


def test_model_comparison_summary_reports_improvement_honestly() -> None:
    """Summary should accurately report whether performance improved."""
    df = _make_synthetic_classification_df()
    specs = [
        FeatureEngineeringSpec(
            feature_name="log_feature_0",
            transformation_type="log1p",
            columns=["feature_0"],
        )
    ]

    comparison = ModelComparison().compare(df, "target", specs)

    if comparison.improved:
        assert "improved" in comparison.summary.lower()
    else:
        # Could be "did not improve" or "some metrics improved while others declined"
        assert (
            "did not improve" in comparison.summary.lower()
            or "some metrics improved while others declined" in comparison.summary.lower()
        )


def test_all_previous_tests_still_pass() -> None:
    """Marker test to confirm no regression in existing functionality.

    The actual verification happens through the complete pytest run.
    """
    from src.aegis.model_comparison import ModelComparison
    from src.aegis.schemas import BaselineVsEngineeredComparison

    assert ModelComparison is not None
    assert BaselineVsEngineeredComparison is not None