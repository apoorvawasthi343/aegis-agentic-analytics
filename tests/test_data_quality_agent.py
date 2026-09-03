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
    missing_values_by_column: dict[str, int] | None = None,
    duplicate_row_count: int = 0,
    row_count: int = 100,
    unique_values_by_column: dict[str, int] | None = None,
) -> DatasetProfile:
    """Helper to build a minimal DatasetProfile for testing."""
    return DatasetProfile(
        row_count=row_count,
        column_count=max(
            len(missing_values_by_column or {}),
            len(unique_values_by_column or {}),
            1,
        ),
        duplicate_row_count=duplicate_row_count,
        missing_values_by_column=missing_values_by_column or {},
        unique_values_by_column=unique_values_by_column or {},
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
    profile = _make_profile(missing_values_by_column={"age": 10}, row_count=100)

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
    """A profile with no missing values and no duplicates produces zero findings."""
    profile = _make_profile(
        missing_values_by_column={}, duplicate_row_count=0, row_count=50
    )

    report = DataQualityAgent().analyze(profile)

    assert len(report.findings) == 0
    assert report.summary == "No data-quality issues detected."


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
    profile = _make_profile(
        missing_values_by_column={"col": missing_count}, row_count=row_count
    )

    report = DataQualityAgent().analyze(profile)

    assert len(report.findings) == 1
    assert report.findings[0].severity == expected_severity


def test_summary_reports_total_number_of_findings():
    """Summary reports the total number of data-quality findings detected."""
    profile = _make_profile(
        missing_values_by_column={"a": 1, "b": 2},
        duplicate_row_count=0,
        row_count=100,
    )

    report = DataQualityAgent().analyze(profile)

    assert len(report.findings) == 2
    assert "2 data-quality issue" in report.summary


def test_duplicate_rows_creates_one_finding():
    """Duplicate rows produce one DataQualityFinding with issue_type 'duplicate_rows'."""
    profile = _make_profile(duplicate_row_count=2, row_count=100)

    report = DataQualityAgent().analyze(profile)

    assert len(report.findings) == 1
    finding = report.findings[0]

    attrs = _finding_attrs(finding)
    assert attrs["issue_type"] == "duplicate_rows"
    assert attrs["column"] is None
    assert attrs["severity"] == "low"
    assert "2 duplicate row" in attrs["evidence"]
    assert "2.00%" in attrs["evidence"]


def test_no_duplicate_rows_creates_no_duplicate_finding():
    """Zero duplicate rows produces no duplicate-row finding."""
    profile = _make_profile(duplicate_row_count=0, row_count=50)

    report = DataQualityAgent().analyze(profile)

    duplicate_findings = [f for f in report.findings if f.issue_type == "duplicate_rows"]
    assert len(duplicate_findings) == 0


@pytest.mark.parametrize(
    "duplicate_count,row_count,expected_severity",
    [
        (1, 100, "low"),      # 1%
        (2, 100, "low"),      # 2% (boundary: <= 2% is still low)
        (3, 100, "medium"),   # 3%
        (10, 100, "medium"),  # 10% (boundary: <= 10% is still medium)
        (11, 100, "high"),    # 11%
        (30, 100, "high"),    # 30%
    ],
)
def test_duplicate_row_severity_thresholds(duplicate_count, row_count, expected_severity):
    """Duplicate-row severity is assigned correctly at the defined thresholds."""
    profile = _make_profile(
        duplicate_row_count=duplicate_count, row_count=row_count
    )

    report = DataQualityAgent().analyze(profile)

    duplicate_findings = [f for f in report.findings if f.issue_type == "duplicate_rows"]
    assert len(duplicate_findings) == 1
    assert duplicate_findings[0].severity == expected_severity


def test_combined_missing_and_duplicate_findings():
    """Missing values and duplicate rows produce findings for both issues."""
    profile = _make_profile(
        missing_values_by_column={"age": 10},
        duplicate_row_count=5,
        row_count=100,
    )

    report = DataQualityAgent().analyze(profile)

    assert len(report.findings) == 2

    issue_types = {f.issue_type for f in report.findings}
    assert issue_types == {"missing_values", "duplicate_rows"}

    assert report.summary == "2 data-quality issue(s) detected."


def test_high_cardinality_nearly_unique_column_is_flagged():
    """A column with 96% unique values is flagged as high cardinality (medium)."""
    profile = _make_profile(
        unique_values_by_column={"code": 96},
        row_count=100,
    )

    report = DataQualityAgent().analyze(profile)

    high_card_findings = [f for f in report.findings if f.issue_type == "high_cardinality"]
    assert len(high_card_findings) == 1

    attrs = _finding_attrs(high_card_findings[0])
    assert attrs["column"] == "code"
    assert attrs["severity"] == "medium"
    assert "96 unique value" in attrs["evidence"]
    assert "96.00%" in attrs["evidence"]


def test_high_cardinality_fully_unique_column_is_flagged():
    """A column with 100% unique values is flagged as high cardinality (high)."""
    profile = _make_profile(
        unique_values_by_column={"id": 100},
        row_count=100,
    )

    report = DataQualityAgent().analyze(profile)

    high_card_findings = [f for f in report.findings if f.issue_type == "high_cardinality"]
    assert len(high_card_findings) == 1

    attrs = _finding_attrs(high_card_findings[0])
    assert attrs["column"] == "id"
    assert attrs["severity"] == "high"
    assert "100 unique value" in attrs["evidence"]
    assert "100.00%" in attrs["evidence"]


def test_low_cardinality_column_is_not_flagged():
    """A column with 50% unique values is not flagged as high cardinality."""
    profile = _make_profile(
        unique_values_by_column={"category": 50},
        row_count=100,
    )

    report = DataQualityAgent().analyze(profile)

    high_card_findings = [f for f in report.findings if f.issue_type == "high_cardinality"]
    assert len(high_card_findings) == 0


def test_combined_missing_and_high_cardinality_findings():
    """Missing values and high cardinality produce findings for both issues."""
    profile = _make_profile(
        missing_values_by_column={"age": 5},
        unique_values_by_column={"id": 98},
        row_count=100,
    )

    report = DataQualityAgent().analyze(profile)

    issue_types = {f.issue_type for f in report.findings}
    assert "missing_values" in issue_types
    assert "high_cardinality" in issue_types

    assert len(report.findings) == 2
    assert report.summary == "2 data-quality issue(s) detected."


def test_row_count_zero_does_not_flag_high_cardinality():
    """When row_count is 0, high cardinality columns are not flagged."""
    profile = _make_profile(
        unique_values_by_column={"id": 10},
        row_count=0,
    )

    report = DataQualityAgent().analyze(profile)

    high_card_findings = [f for f in report.findings if f.issue_type == "high_cardinality"]
    assert len(high_card_findings) == 0
