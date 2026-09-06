"""AEGISOrchestrator - pipeline orchestration for the AEGIS framework.

Orchestrates the data pipeline without LLM calls or complex ML models.
Uses a simple linear pipeline: load → profile → data quality → EDA →
feature engineering → modeling comparison → critic review.

All components are injectable for testability.
"""

from __future__ import annotations

import pandas as pd
from src.aegis.schemas import (
    AppliedFeature,
    BaselineVsEngineeredComparison,
    CriticReport,
    DataQualityReport,
    DatasetProfile,
    EDAReport,
    FeatureEngineeringReport,
    ModelingReport,
    OrchestrationResult,
    SkippedFeature,
)
from src.aegis.agents.data_quality_agent import DataQualityAgent
from src.aegis.agents.eda_agent import EDAAgent
from src.aegis.agents.feature_engineering_agent import FeatureEngineeringAgent
from src.aegis.pipelines.model_comparison import ModelComparison
from src.aegis.pipelines.critic_agent import CriticAgent as CriticAgent_
from src.aegis.pipelines.orchestration import AEGISOrchestrator


class AEGISOrchestrator:
    """Orchestrates the AEGIS pipeline.

    The pipeline is:
    1. Load CSV
    2. Profile dataset
    3. Data quality analysis
    4. EDA analysis
    5. Feature engineering (plan + execute)
    6. Model comparison (baseline vs engineered)
    7. Critic review
    """

    def __init__(
        self,
        data_quality_agent: DataQualityAgent,
        eda_agent: EDAAgent,
        feature_eng_agent: FeatureEngineeringAgent,
        model_comparison: ModelComparison,
        critic_agent: CriticAgent_,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            data_quality_agent: Agent for data quality analysis.
            eda_agent: Agent for exploratory data analysis.
            feature_eng_agent: Agent for feature engineering recommendations.
            model_comparison: Comparator for baseline vs engineered models.
            critic_agent: Critic for reviewing feature engineering results.
        """
        self.data_quality_agent = data_quality_agent
        self.eda_agent = eda_agent
        self.feature_eng_agent = feature_eng_agent
        self.model_comparison = model_comparison
        self.critic_agent = critic_agent

    def run(self, file_path: str, target_column: str) -> OrchestrationResult:
        """Run the complete AEGIS pipeline.

        Args:
            file_path: Path to the CSV file.
            target_column: Name of the target column for modeling.

        Returns:
            OrchestrationResult with all pipeline outputs.
        """
        # 1. Load and profile
        df = _load_csv(file_path)
        profile = _create_profile(df, target_column)

        # 2. Data quality
        data_quality_report = self.data_quality_agent.analyze(profile)

        # 3. EDA
        eda_report = self.eda_agent.analyze(profile)

        # 4. Feature engineering
        feature_specs = self.feature_eng_agent.recommend(df, target_column)
        feature_eng_report = self._apply_feature_engineering(
            df, feature_specs
        )

        # 5. Modeling comparison
        comparison = self.model_comparison.compare(
            df=df,
            target_column=target_column,
            specs=feature_specs,
        )

        # 6. Critic review
        critic_report = self.critic_agent.review(comparison)

        # Build orchestration result
        return OrchestrationResult(
            dataset_profile=profile,
            data_quality_report=data_quality_report,
            eda_report=eda_report,
            feature_engineering_report=feature_eng_report,
            created_features=comparison.features_created,
            skipped_features=comparison.features_skipped,
            modeling_comparison=comparison,
            critic_report=critic_report,
            summary=_build_summary(
                data_quality_report,
                eda_report,
                feature_eng_report,
                comparison,
                critic_report,
            ),
        )

    def _apply_feature_engineering(
        self,
        df: pd.DataFrame,
        specs: list[FeatureEngineeringSpec],
    ) -> FeatureEngineeringReport:
        """Apply feature engineering specs to the DataFrame."""
        # Simple implementation: apply each spec directly
        results: list[AppliedFeature] = []
        skipped: list[SkippedFeature] = []

        for spec in specs:
            if spec.transformation_type == "log":
                try:
                    col = spec.columns[0]
                    result_col = spec.feature_name
                    df[result_col] = df[col].apply(
                        lambda x: np.log(x) if pd.notna(x) and x > 0 else None
                    )
                    results.append(
                        AppliedFeature(
                            feature_name=spec.feature_name,
                            transformation_type="log",
                            columns=spec.columns,
                            result_column=result_col,
                        )
                    )
                except Exception as e:
                    skipped.append(
                        SkippedFeature(
                            feature_name=spec.feature_name,
                            transformation_type="log",
                            columns=spec.columns,
                            reason=str(e),
                        )
                    )
            elif spec.transformation_type == "ratio":
                try:
                    num_col, den_col = spec.columns
                    result_col = spec.feature_name
                    df[result_col] = df[num_col] / df[den_col]
                    results.append(
                        AppliedFeature(
                            feature_name=spec.feature_name,
                            transformation_type="ratio",
                            columns=spec.columns,
                            result_column=result_col,
                        )
                    )
                except Exception as e:
                    skipped.append(
                        SkippedFeature(
                            feature_name=spec.feature_name,
                            transformation_type="ratio",
                            columns=spec.columns,
                            reason=str(e),
                        )
                    )
            elif spec.transformation_type == "interaction":
                try:
                    col1, col2 = spec.columns
                    result_col = spec.feature_name
                    df[result_col] = df[col1] * df[col2]
                    results.append(
                        AppliedFeature(
                            feature_name=spec.feature_name,
                            transformation_type="interaction",
                            columns=spec.columns,
                            result_column=result_col,
                        )
                    )
                except Exception as e:
                    skipped.append(
                        SkippedFeature(
                            feature_name=spec.feature_name,
                            transformation_type="interaction",
                            columns=spec.columns,
                            reason=str(e),
                        )
                    )
            else:
                skipped.append(
                    SkippedFeature(
                        feature_name=spec.feature_name,
                        transformation_type=spec.transformation_type,
                        columns=spec.columns,
                        reason=f"Unknown transformation type: {spec.transformation_type}",
                    )
                )

        return FeatureEngineeringReport(
            original_row_count=df.shape[0],
            original_column_count=df.shape[1] - len(results),
            engineered_row_count=df.shape[0],
            engineered_column_count=df.shape[1],
            applied_features=results,
            skipped_features=skipped,
            summary=f"Applied {len(results)} features, skipped {len(skipped)}.",
        )


def _load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    return pd.read_csv(file_path)


def _create_profile(df: pd.DataFrame, target_column: str) -> DatasetProfile:
    """Create dataset profile from DataFrame."""
    numeric_stats: dict[str, NumericStats] = {}
    categorical_stats: dict[str, CategoricalStat] = {}

    for col in df.columns:
        if col == target_column:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            numeric_stats[col] = NumericStats(
                mean=float(series.mean()) if len(series) > 0 else None,
                median=float(series.median()) if len(series) > 0 else None,
                std=float(series.std()) if len(series) > 0 else None,
                min=float(series.min()) if len(series) > 0 else None,
                max=float(series.max()) if len(series) > 0 else None,
            )
        else:
            value_counts = df[col].value_counts()
            most_frequent_value = value_counts.index[0] if len(value_counts) > 0 else None
            most_frequent_count = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            categorical_stats[col] = CategoricalStat(
                most_frequent_value=str(most_frequent_value) if most_frequent_value is not None else None,
                most_frequent_count=most_frequent_count,
            )

    missing_values = {
        col: int(df[col].isna().sum()) for col in df.columns
    }
    unique_values = {
        col: int(df[col].nunique()) for col in df.columns
    }
    data_types = {
        col: str(df[col].dtype) for col in df.columns
    }

    target_distribution: Optional[Dict[str, int]] = None
    if target_column and target_column in df.columns:
        target_dist = df[target_column].value_counts()
        target_distribution = {
            str(k): int(v) for k, v in target_dist.items()
        }

    return DatasetProfile(
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_row_count=int(df.duplicated().sum()),
        missing_values_by_column=missing_values,
        unique_values_by_column=unique_values,
        data_types_by_column=data_types,
        numeric_statistics=numeric_stats,
        categorical_statistics=categorical_stats,
        target_column=target_column,
        target_distribution=target_distribution,
    )


def _build_summary(
    data_quality_report: DataQualityReport,
    eda_report: EDAReport,
    feature_eng_report: FeatureEngineeringReport,
    comparison: BaselineVsEngineeredComparison,
    critic_report: CriticReport,
) -> str:
    """Build overall summary string."""
    parts = [
        f"Data Quality: {len(data_quality_report.findings)} issue(s) found.",
        f"EDA: {len(eda_report.findings)} finding(s) discovered.",
        f"Feature Engineering: {len(feature_eng_report.applied_features)} features created, "
        f"{len(feature_eng_report.skipped_features)} skipped.",
        f"Modeling: Baseline accuracy {comparison.baseline_metrics.accuracy:.4f} → "
        f"Engineered accuracy {comparison.engineered_metrics.accuracy:.4f} "
        f"(Δ {comparison.accuracy_change:+.4f}).",
        f"Critic: Decision = {critic_report.decision.upper()}.",
    ]
    return ". ".join(parts)