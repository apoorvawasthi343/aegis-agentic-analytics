"""Pydantic schemas for AEGIS orchestration results.

Provides the OrchestrationResult schema that collects outputs from all
AEGIS pipeline stages into a single structured result.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class NumericStats(BaseModel):
    """Statistics for a single numeric column."""

    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


class CategoricalStat(BaseModel):
    """Statistics for a single non-numeric (categorical) column."""

    most_frequent_value: Any = None
    most_frequent_count: int = 0


class DatasetProfile(BaseModel):
    """Structured representation of a dataset profile."""

    row_count: int
    column_count: int
    duplicate_row_count: int
    missing_values_by_column: dict[str, int]
    unique_values_by_column: dict[str, int]
    data_types_by_column: dict[str, str]
    numeric_statistics: dict[str, NumericStats]
    categorical_statistics: dict[str, CategoricalStat]
    target_column: Optional[str] = None
    target_distribution: Optional[dict[str, int]] = None
    summary: str = ""


class DataQualityFinding(BaseModel):
    """A single data quality issue detected in a dataset."""

    issue_type: str
    severity: str
    column: Optional[str] = None
    evidence: str
    recommendation: str


class DataQualityReport(BaseModel):
    """Structured report of data quality findings for a dataset."""

    findings: list[DataQualityFinding]
    summary: str


class EDAFinding(BaseModel):
    """A single exploratory data analysis finding in a dataset."""

    finding_type: str
    importance: str
    columns: list[str]
    evidence: str
    interpretation: str
    modeling_implication: Optional[str] = None


class EDAReport(BaseModel):
    """Structured exploratory data analysis report for a dataset."""

    findings: list[EDAFinding]
    summary: str


class ModelMetrics(BaseModel):
    """Performance metrics for a trained classification model."""

    accuracy: float
    f1_score: float
    roc_auc: Optional[float] = None
    train_rows: int
    test_rows: int


class ModelingReport(BaseModel):
    """Structured report of a baseline modeling run."""

    model_name: str
    task_type: str
    target_column: str
    metrics: ModelMetrics
    notes: str


class FeatureEngineeringSpec(BaseModel):
    """Specification for a single feature engineering transformation."""

    feature_name: str
    transformation_type: str
    columns: list[str]
    parameters: Optional[dict[str, Any]] = None


class AppliedFeature(BaseModel):
    """A feature engineering transformation that was successfully applied."""

    feature_name: str
    transformation_type: str
    columns: list[str]
    result_column: str


class SkippedFeature(BaseModel):
    """A requested feature engineering transformation that was skipped."""

    feature_name: str
    transformation_type: str
    columns: list[str]
    reason: str


class FeatureEngineeringReport(BaseModel):
    """Report from applying feature engineering transformations to a dataset."""

    original_row_count: int
    original_column_count: int
    engineered_row_count: int
    engineered_column_count: int
    applied_features: list[AppliedFeature]
    skipped_features: list[SkippedFeature]
    summary: str


class BaselineVsEngineeredComparison(BaseModel):
    """Comparison of baseline vs. feature-engineered model performance."""

    baseline_metrics: ModelMetrics
    engineered_metrics: ModelMetrics
    features_created: list[str]
    features_skipped: list[str]
    accuracy_change: float
    f1_change: float
    roc_auc_change: Optional[float] = None
    improved: bool
    summary: str


class CriticReport(BaseModel):
    """Result of a critic's review of a feature-engineering comparison."""

    decision: str
    accepted_features: list[str]
    rejected_features: list[str]
    reasons: list[str]
    performance_improved: bool
    leakage_warning: bool
    summary: str


class OrchestrationResult(BaseModel):
    """Complete orchestration result from the AEGIS pipeline.

    Collects outputs from all pipeline stages:
    profiling → data quality → EDA → feature engineering →
    modeling comparison → critic review
    """

    dataset_profile: DatasetProfile
    data_quality_report: DataQualityReport
    eda_report: EDAReport
    feature_engineering_report: FeatureEngineeringReport
    created_features: list[str]
    skipped_features: list[str]
    modeling_comparison: BaselineVsEngineeredComparison
    critic_report: CriticReport
    summary: str = ""