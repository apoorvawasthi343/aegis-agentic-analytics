"""Baseline vs. engineered model comparison for AEGIS.

Compares a baseline Logistic Regression model trained on original
features vs. the same modeling approach trained on engineered features.
Uses the same train/test split, preprocessing, and evaluation approach
so the comparison is fair and apples-to-apples.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.aegis.feature_engineering_executor import FeatureEngineeringExecutor
from src.aegis.schemas import (
    BaselineVsEngineeredComparison,
    FeatureEngineeringSpec,
    ModelMetrics,
    ModelingReport,
)
from src.aegis.modeling_agent import ModelingAgent


class ModelComparison:
    """Compares baseline vs. feature-engineered model performance."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        max_iter: int = 1000,
    ) -> None:
        """Initialize the model comparison.

        Args:
            test_size: Fraction of data to hold out for testing.
            random_state: Random seed for train/test split.
            max_iter: max_iter passed to LogisticRegression.
        """
        self.modeling_agent = ModelingAgent(
            test_size=test_size,
            random_state=random_state,
            max_iter=max_iter,
        )
        self.feature_executor = FeatureEngineeringExecutor()

    def compare(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_specs: list[FeatureEngineeringSpec],
    ) -> BaselineVsEngineeredComparison:
        """Run baseline vs. engineered model comparison.

        Args:
            df: The full dataset including the target column.
            target_column: Name of the target column to predict.
            feature_specs: Feature engineering specs to apply before
                training the engineered model.

        Returns:
            A BaselineVsEngineeredComparison with before/after metrics.
        """
        # Train baseline model on original features
        baseline_report = self.modeling_agent.analyze(
            df=df,
            target_column=target_column,
        )

        # Apply feature engineering
        engineered_df, fe_report = self.feature_executor.apply(
            df=df,
            specs=feature_specs,
        )

        # Train engineered model on engineered features
        engineered_report = self.modeling_agent.analyze(
            df=engineered_df,
            target_column=target_column,
        )

        # Calculate metric changes
        accuracy_change = (
            engineered_report.metrics.accuracy - baseline_report.metrics.accuracy
        )
        f1_change = (
            engineered_report.metrics.f1_score - baseline_report.metrics.f1_score
        )

        roc_auc_change: Optional[float] = None
        if (
            baseline_report.metrics.roc_auc is not None
            and engineered_report.metrics.roc_auc is not None
        ):
            roc_auc_change = (
                engineered_report.metrics.roc_auc
                - baseline_report.metrics.roc_auc
            )

        # Determine if overall improvement
        baseline_score = baseline_report.metrics.f1_score
        engineered_score = engineered_report.metrics.f1_score
        improved = engineered_score > baseline_score

        # Build summary
        summary_parts = [
            f"Baseline model achieved accuracy {baseline_report.metrics.accuracy:.4f} "
            f"and F1 {baseline_report.metrics.f1_score:.4f}.",
            f"Engineered model achieved accuracy {engineered_report.metrics.accuracy:.4f} "
            f"and F1 {engineered_report.metrics.f1_score:.4f}.",
        ]

        if roc_auc_change is not None:
            summary_parts.append(
                f"ROC-AUC changed from {baseline_report.metrics.roc_auc:.4f} to "
                f"{engineered_report.metrics.roc_auc:.4f} "
                f"(Δ {roc_auc_change:+.4f})."
            )

        if improved:
            summary_parts.append(
                "The engineered features improved model performance."
            )
        elif accuracy_change > 0 and f1_change <= 0:
            summary_parts.append(
                "Accuracy improved but F1 did not; some metrics improved while others declined."
            )
        elif accuracy_change <= 0 and f1_change > 0:
            summary_parts.append(
                "Accuracy declined but F1 improved; some metrics improved while others declined."
            )
        else:
            summary_parts.append(
                "The engineered features did not improve model performance."
            )

        summary = " ".join(summary_parts)

        features_created = [f.feature_name for f in fe_report.applied_features]
        features_skipped = [
            f"{f.feature_name}: {f.reason}" for f in fe_report.skipped_features
        ]

        return BaselineVsEngineeredComparison(
            baseline_metrics=baseline_report.metrics,
            engineered_metrics=engineered_report.metrics,
            features_created=features_created,
            features_skipped=features_skipped,
            accuracy_change=accuracy_change,
            f1_change=f1_change,
            roc_auc_change=roc_auc_change,
            improved=improved,
            summary=summary,
        )