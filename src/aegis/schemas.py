"""Pydantic schemas for AEGIS dataset profiling results."""

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
