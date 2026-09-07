"""Tests for the AEGIS Critic Agent."""

import pytest
import pandas as pd
from src.aegis.critic_agent import CriticAgent
from src.aegis.schemas import (
    BaselineVsEngineeredComparison,
    ModelMetrics,
)


def _make_comparison(
    accuracy_change: float = 0.0,
    f1_change: float = 0.0,
    roc_auc_change: float = 0.0,
    features_created: list[str] | None = None,
    features_skipped: list[str] | None = None,
) -> BaselineVsEngineeredComparison:
    """Helper to create a BaselineVsEngineeredComparison for testing."""
    baseline = ModelMetrics(
        accuracy=0.85,
        f1_score=0.8496,
        roc_auc=0.93,
        train_rows=160,
        test_rows=40,
    )
    engineered = ModelMetrics(
        accuracy=baseline.accuracy + accuracy_change,
        f1_score=baseline.f1_score + f1_change,
        roc_auc=baseline.roc_auc + roc_auc_change,
        train_rows=160,
        test_rows=40,
    )
    return BaselineVsEngineeredComparison(
        baseline_metrics=baseline,
        engineered_metrics=engineered,
        features_created=features_created or [],
        features_skipped=features_skipped or [],
        accuracy_change=accuracy_change,
        f1_change=f1_change,
        roc_auc_change=roc_auc_change if roc_auc_change != 0.0 else None,
        improved=f1_change > 0,
        summary="test comparison",
    )


def test_clear_improvement_accepts() -> None:
    """Clear improvement should result in accept decision."""
    comparison = _make_comparison(
        accuracy_change=0.05,
        f1_change=0.03,
        roc_auc_change=0.02,
        features_created=["log_feature_0", "log_feature_1"],
    )

    critic = CriticAgent()
    report = critic.review(comparison)

    assert report.decision == "accept"
    assert report.accepted_features == ["log_feature_0", "log_feature_1"]
    assert len(report.rejected_features) == 0
    assert report.performance_improved is True


def test_no_improvement_rejects() -> None:
    """No improvement should result in reject decision."""
    comparison = _make_comparison(
        accuracy_change=0.0,
        f1_change=0.0,
        features_created=["log_feature_0"],
    )

    critic = CriticAgent()
    report = critic.review(comparison)

    assert report.decision == "reject"
    assert report.accepted_features == []
    assert report.rejected_features == ["log_feature_0"]
    assert report.performance_improved is False


def test_mixed_metrics_review() -> None:
    """Mixed results (some metrics improve, others decline) should be review."""
    comparison = _make_comparison(
        accuracy_change=0.03,
        f1_change=-0.01,
        features_created=["log_feature_0"],
    )

    critic = CriticAgent()
    report = critic.review(comparison)

    assert report.decision == "review"
    assert report.accepted_features == ["log_feature_0"]
    assert report.rejected_features == []


def test_no_created_features_rejects() -> None:
    """No created features should result in reject."""
    comparison = _make_comparison(
        features_created=[],
        features_skipped=["bad_feature: column not found"],
    )

    critic = CriticAgent()
    report = critic.review(comparison)

    assert report.decision == "reject"
    assert report.accepted_features == []
    assert "bad_feature: column not found" in report.rejected_features
    assert any("No engineered features" in r for r in report.reasons)


def test_accept_leaves_skipped_features_in_rejected() -> None:
    """When accepting, skipped features should still appear in rejected list."""
    comparison = _make_comparison(
        accuracy_change=0.03,
        f1_change=0.02,
        features_created=["good_feature"],
        features_skipped=["bad_feature: missing column"],
    )

    critic = CriticAgent()
    report = critic.review(comparison)

    assert report.decision == "accept"
    assert report.accepted_features == ["good_feature"]
    assert "bad_feature: missing column" in report.rejected_features


# --- New tests for the two issues ---


def test_mixed_roc_auc_decline_is_review() -> None:
    """ROC-AUC decline with accuracy/F1 improvement should produce 'review'.

    Simulates the actual run:
        accuracy_change = +0.0167
        f1_change       = +0.0082
        roc_auc_change  = -0.0682
    """
    comparison = _make_comparison(
        accuracy_change=0.0167,
        f1_change=0.0082,
        roc_auc_change=-0.0682,
        features_created=["log_monthly_charges", "log_total_charges"],
        features_skipped=[],
    )

    critic = CriticAgent()
    report = critic.review(comparison)

    assert report.decision == "review"
    assert any("ROC-AUC" in r and "declined" in r for r in report.reasons)


def test_all_metrics_improve_accept() -> None:
    """All three metrics improving should still produce 'accept'."""
    comparison = _make_comparison(
        accuracy_change=0.02,
        f1_change=0.015,
        roc_auc_change=0.01,
        features_created=["log_x"],
    )

    critic = CriticAgent()
    report = critic.review(comparison)

    assert report.decision == "accept"


def test_no_metrics_improve_reject() -> None:
    """All metrics declining should produce 'reject'."""
    comparison = _make_comparison(
        accuracy_change=-0.02,
        f1_change=-0.01,
        roc_auc_change=-0.03,
        features_created=["log_x"],
    )

    critic = CriticAgent()
    report = critic.review(comparison)

    assert report.decision == "reject"


def test_feature_accounting_proposed_equals_created_plus_skipped() -> None:
    """The feature engineering report must satisfy:
    proposed = created + skipped.
    """
    from src.aegis.feature_engineering_executor import FeatureEngineeringExecutor
    from src.aegis.schemas import FeatureEngineeringSpec

    specs = [
        # Valid: should be created
        FeatureEngineeringSpec(
            feature_name="log_age",
            transformation_type="log1p",
            columns=["age"],
        ),
        FeatureEngineeringSpec(
            feature_name="log_tenure",
            transformation_type="log1p",
            columns=["tenure_months"],
        ),
        # Invalid: column does not exist -> must be skipped with reason
        FeatureEngineeringSpec(
            feature_name="log_nonexistent",
            transformation_type="log1p",
            columns=["_no_such_column_"],
        ),
        # Invalid: not a supported type -> must be skipped with reason
        FeatureEngineeringSpec(
            feature_name="bad_transform",
            transformation_type="invalid_type_xyz",
            columns=["age"],
        ),
        # Valid: missing indicator
        FeatureEngineeringSpec(
            feature_name="age_is_missing",
            transformation_type="missing_indicator",
            columns=["age"],
        ),
    ]

    executor = FeatureEngineeringExecutor()
    df = pd.DataFrame(
        {
            "age": [1, 2, 3, None, 5],
            "tenure_months": [12, 24, 36, 48, 60],
            "churn": [0, 1, 0, 1, 0],
        }
    )

    engineered_df, report = executor.apply(df, specs)

    created = [f.feature_name for f in report.applied_features]
    skipped = [f.feature_name for f in report.skipped_features]

    # proposed = created + skipped
    assert len(specs) == len(created) + len(skipped), (
        f"proposed={len(specs)}, created={len(created)}, skipped={len(skipped)}"
    )

    # Every skipped feature must carry a reason
    for sf in report.skipped_features:
        assert sf.reason, f"Skipped feature '{sf.feature_name}' has no reason"

    # Verify specific expectations
    assert "log_age" in created
    assert "log_tenure" in created
    assert "log_nonexistent" in skipped
    assert "bad_transform" in skipped