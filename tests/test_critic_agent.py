"""Tests for the AEGIS Critic Agent."""

import pytest

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
    assert report.performance_improved is False


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