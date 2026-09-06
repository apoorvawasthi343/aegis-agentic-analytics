"""Pydantic schemas for AEGIS modeling results."""

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
    """A single exploratory data analysis finding in a dataset.

    EDAFindings capture patterns and insights useful for understanding
    the data before modeling, such as distributions, relationships,
    imbalance, and potential predictive signals.
    """

    finding_type: str
    """Category of the finding (e.g. 'distribution', 'imbalance', 'relationship')."""

    importance: str
    """How important this finding is for modeling: 'low', 'medium', or 'high'."""

    columns: list[str]
    """Columns involved in or related to this finding."""

    evidence: str
    """Observable data evidence supporting this finding."""

    interpretation: str
    """What this finding means in plain language."""

    modeling_implication: str | None = None
    """Suggested impact or next step for modeling, or None if not applicable."""


class EDAReport(BaseModel):
    """Structured exploratory data analysis report for a dataset.

    EDAReports collect EDAFindings along with an overall summary
    of the most important patterns discovered.
    """

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
    """Specification for a single requested feature engineering transformation.

    Used by callers (e.g. an LLM-based planner) to request a specific
    transformation that the executor will attempt to apply safely.
    """

    feature_name: str
    """Human-readable name for the engineered feature column."""

    transformation_type: str
    """Type of transformation: 'log1p', 'ratio', 'missing_indicator', 'count_sum'."""

    columns: list[str]
    """Columns involved in the transformation."""

    parameters: dict[str, Any] = {}
    """Optional extra parameters for the transformation."""


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