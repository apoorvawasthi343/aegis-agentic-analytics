"""AEGIS Orchestrator - pipeline orchestration for the AEGIS framework.

Orchestrates the complete data science pipeline:
1. Load CSV
2. Profile dataset
3. Data quality analysis
4. EDA analysis
5. Feature engineering (plan + execute via ModelComparison)
6. Model comparison (baseline vs engineered)
7. Critic review

All components are injectable for testability.

Design note: the orchestrator coordinates components; it does NOT
independently invent feature-engineering decisions. If a feature
planner is not provided, no features are engineered and the modeling
comparison runs on original features only. The _default_feature_planner
exists as an explicitly optional deterministic fallback, disabled by
default.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from src.aegis.data_quality_agent import DataQualityAgent
from src.aegis.eda_agent import EDAAgent
from src.aegis.feature_engineering_executor import FeatureEngineeringExecutor
from src.aegis.loader import load_csv
from src.aegis.model_comparison import ModelComparison
from src.aegis.profiler import profile_dataset
from src.aegis.schemas import (
    AppliedFeature,
    BaselineVsEngineeredComparison,
    CriticReport,
    DataQualityReport,
    DatasetProfile,
    EDAReport,
    FeatureEngineeringReport,
    FeatureEngineeringSpec,
    OrchestrationResult,
    SkippedFeature,
)
from src.aegis.critic_agent import CriticAgent


class AEGISOrchestrator:
    """Orchestrates the complete AEGIS data science pipeline.

    Pipeline stages:
    1. Load CSV
    2. Profile dataset
    3. Data quality analysis
    4. EDA analysis
    5. Feature engineering (plan + execute)
    6. Model comparison (baseline vs engineered)
    7. Critic review

    All components are injectable for testability. The orchestrator
    does NOT make any LLM calls itself; LLM integration happens
    inside the individual agents if they are configured with an
    LLM client.

    Feature engineering:
    - If a feature_planner is provided, it generates FeatureEngineeringSpecs.
    - If no feature_planner is provided, feature_specs is empty and the
      modeling comparison runs on original features only.
    - The _default_feature_planner is an OPTIONAL deterministic fallback
      that is NOT used by default. Pass it explicitly if you want
      automatic log1p / missing_indicator suggestions.
    """

    def __init__(
        self,
        data_quality_agent: DataQualityAgent,
        eda_agent: EDAAgent,
        feature_executor: FeatureEngineeringExecutor,
        model_comparison: ModelComparison,
        critic_agent: CriticAgent,
        *,
        feature_planner: Callable[[pd.DataFrame, str], list[FeatureEngineeringSpec]] | None = None,
        loader: Callable[[str], pd.DataFrame] = load_csv,
        profiler: Callable[[pd.DataFrame, str | None], DatasetProfile] = profile_dataset,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            data_quality_agent: Agent for data quality analysis.
            eda_agent: Agent for exploratory data analysis.
            feature_executor: Executor for applying feature engineering specs.
            model_comparison: Comparator for baseline vs engineered models.
            critic_agent: Critic for reviewing feature engineering results.
            feature_planner: Optional callable that generates feature specs
                from a DataFrame and target column. If None, no features
                are engineered and the comparison uses original features only.
                Use _default_feature_planner for automatic suggestions, but
                that is NOT the default — you must opt in explicitly.
            loader: Callable that loads a CSV file into a DataFrame.
            profiler: Callable that creates a DatasetProfile from a DataFrame.
        """
        self.data_quality_agent = data_quality_agent
        self.eda_agent = eda_agent
        self.feature_executor = feature_executor
        self.model_comparison = model_comparison
        self.critic_agent = critic_agent
        self.feature_planner = feature_planner
        self.loader = loader
        self.profiler = profiler

    def run(
        self,
        file_path: str,
        target_column: str,
    ) -> OrchestrationResult:
        """Run the complete AEGIS pipeline.

        Args:
            file_path: Path to the CSV file to analyze.
            target_column: Name of the target column for modeling.

        Returns:
            OrchestrationResult with all pipeline outputs.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the target column is not found in the data.
        """
        # Stage 1: Load data
        df = self.loader(file_path)

        # Stage 2: Profile
        profile = self.profiler(df, target_column=target_column)

        # Stage 3: Data quality analysis
        data_quality_report = self.data_quality_agent.analyze(profile)

        # Stage 4: EDA analysis
        eda_report = self.eda_agent.analyze(profile)

        # Stage 5: Feature engineering planning
        # If no feature_planner is provided, feature_specs is empty.
        # The orchestrator does NOT invent features on its own.
        feature_specs: list[FeatureEngineeringSpec] = []
        if self.feature_planner is not None:
            feature_specs = self.feature_planner(df, target_column)

        # Stage 6: Model comparison (applies features + trains models)
        comparison = self.model_comparison.compare(
            df=df,
            target_column=target_column,
            feature_specs=feature_specs,
        )

        # Stage 7: Critic review
        critic_report = self.critic_agent.review(comparison)

        # Build orchestration result
        return OrchestrationResult(
            dataset_profile=profile,
            data_quality_report=data_quality_report,
            eda_report=eda_report,
            feature_engineering_report=FeatureEngineeringReport(
                original_row_count=profile.row_count,
                original_column_count=profile.column_count,
                engineered_row_count=profile.row_count,
                engineered_column_count=profile.column_count
                + len(comparison.features_created),
                applied_features=[
                    AppliedFeature(
                        feature_name=name,
                        transformation_type="auto",
                        columns=[],
                        result_column=name,
                    )
                    for name in comparison.features_created
                ],
                skipped_features=[
                    SkippedFeature(
                        feature_name=name.split(":")[0].strip(),
                        transformation_type="auto",
                        columns=[],
                        reason=name,
                    )
                    for name in comparison.features_skipped
                ],
                summary=f"Created {len(comparison.features_created)} features, "
                f"skipped {len(comparison.features_skipped)}.",
            ),
            created_features=comparison.features_created,
            skipped_features=comparison.features_skipped,
            modeling_comparison=comparison,
            critic_report=critic_report,
            summary=(
                f"Pipeline completed. Data quality: {len(data_quality_report.findings)} "
                f"issue(s). EDA: {len(eda_report.findings)} finding(s). "
                f"Features: {len(comparison.features_created)} created, "
                f"{len(comparison.features_skipped)} skipped. "
                f"Modeling: baseline accuracy {comparison.baseline_metrics.accuracy:.4f} "
                f"→ engineered {comparison.engineered_metrics.accuracy:.4f}. "
                f"Critic decision: {critic_report.decision}."
            ),
        )


def _default_feature_planner(
    df: pd.DataFrame,
    target_column: str,
) -> list[FeatureEngineeringSpec]:
    """Generate default feature engineering specs for numeric columns.

    Creates log1p features for positive numeric columns and missing
    indicators for columns with any missing values.

    NOTE: This is an explicitly optional deterministic fallback.
    It is NOT used by the orchestrator by default — you must pass it
    explicitly via feature_planner=_default_feature_planner if you
    want automatic feature suggestions without a Feature Engineering
    Agent.
    """
    specs: list[FeatureEngineeringSpec] = []
    feature_cols = [c for c in df.columns if c != target_column]

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Log1p for positive numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            if (df[col].dropna() > 0).any() and (df[col].dropna() >= 0).all():
                specs.append(
                    FeatureEngineeringSpec(
                        feature_name=f"log_{col}",
                        transformation_type="log1p",
                        columns=[col],
                    )
                )

        # Missing indicator for columns with missing values
        if df[col].isna().any():
            specs.append(
                FeatureEngineeringSpec(
                    feature_name=f"{col}_is_missing",
                    transformation_type="missing_indicator",
                    columns=[col],
                )
            )

    return specs