"""Tests for the AEGIS Data Quality Agent."""

import pytest

from src.aegis.data_quality_agent import DataQualityAgent
from src.aegis.schemas import (
    CategoricalStat,
    DataQualityFinding,
    DataQualityReport,
    DatasetProfile,
    NumericStats,
)


def _make_profile(
    missing_values_by_column: dict[str, int],
    row_count: int = 100,
) -> DatasetProfile:
    """Helper to build a minimal DatasetProfile for testing."""
    return DatasetProfile(
        row_count=row_count,
        column_count=len(missing_values_by_column),
        duplicate_row_count=0,
        missing_values_by_column=missing_values_by_column,
        unique_values_by_column={},
        data_types_by_column={},
        numeric_statistics={},
        categorical_statistics={},
    )


def _finding_attrs(finding: DataQualityFinding) -> dict:
    """Extract the most important attributes from a finding."""
    return {
        "issue_type": finding.issue_type,
        "severity": finding.severity,
        "column": finding.column,
        "evidence": finding.evidence,
        "recommendation": finding.recommendation,
    }


def test_missing_values_creates_one_finding():
    """A column with missing values produces one DataQualityFinding."""
    profile = _make_profile({"age": 10}, row_count=100)

    report = DataQualityAgent().analyze(profile)

    assert len(report.findings) == 1
    finding = report.findings[0]

    attrs = _finding_attrs(finding)
    assert attrs["issue_type"] == "missing_values"
    assert attrs["column"] == "age"
    assert attrs["severity"] == "medium"
    assert "10 missing value" in attrs["evidence"]
    assert "10.00%" in attrs["evidence"]


def test_no_missing_values_creates_no_findings():
    """A profile with no missing values produces zero findings."""
    profile = _make_profile({}, row_count=50)

    report = DataQualityAgent().analyze(profile)

    assert len(report.findings) == 0
    assert report.summary == "No missing-value issues detected."


@pytest.mark.parametrize(
    "missing_count,row_count,expected_severity",
    [
        (2, 100, "low"),      # 2%
        (5, 100, "low"),      # 5% (boundary: <= 5% is still low)
        (6, 100, "medium"),   # 6%
        (20, 100, "medium"),  # 20% (boundary: <= 20% is still medium)
        (21, 100, "high"),    # 21%
        (50, 100, "high"),    # 50%
    ],
)
def test_missing_value_severity_thresholds(missing_count, row_count, expected_severity):
    """Severity is assigned correctly at the defined thresholds."""
    profile = _make_profile({"col": missing_count}, row_count=row_count)

    report = DataQualityAgent().analyze(profile)

    assert len(report.findings) == 1
    assert report.findings[0].severity == expected_severity


def test_summary_reports_number_of_missing_value_issues():
    """Summary reports how many missing-value issues were detected."""
    profile = _make_profile({"a": 1, "b": 2, "c": 0}, row_count=100)

    report = DataQualityAgent().analyze(profile)

    assert len(report.findings) == 2
    assert "2 missing-value issue" in report.summary
