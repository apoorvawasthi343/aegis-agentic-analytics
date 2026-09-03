"""Tests for the AEGIS Exploratory Data Analysis Agent."""

import pytest

from src.aegis.eda_agent import EDAAgent
from src.aegis.schemas import (
    CategoricalStat,
    DatasetProfile,
    EDAReport,
)


def _make_profile(
    row_count: int = 100,
    target_column: str | None = None,
    target_distribution: dict[str, int] | None = None,
    categorical_statistics: dict[str, CategoricalStat] | None = None,
) -> DatasetProfile:
    """Helper to build a minimal DatasetProfile for EDA tests."""
    return DatasetProfile(
        row_count=row_count,
        column_count=max(
            1,
            (1 if target_column else 0)
            + (1 if target_distribution else 0)
            + len(categorical_statistics or {}),
        ),
        duplicate_row_count=0,
        missing_values_by_column={},
        unique_values_by_column={},
        data_types_by_column={},
        numeric_statistics={},
        categorical_statistics=categorical_statistics or {},
        target_column=target_column,
        target_distribution=target_distribution or None,
    )


def _finding_attrs(finding) -> dict:
    """Extract the most important attributes from an EDAFinding."""
    return {
        "finding_type": finding.finding_type,
        "importance": finding.importance,
        "columns": finding.columns,
        "evidence": finding.evidence,
        "interpretation": finding.interpretation,
        "modeling_implication": finding.modeling_implication,
    }


# ---------------------------------------------------------------------------
# PART 1: Target imbalance
# ---------------------------------------------------------------------------

def test_balanced_target_produces_no_imbalance_finding() -> None:
    """A balanced target should not be flagged as imbalanced."""
    profile = _make_profile(
        target_column="churn",
        target_distribution={"no": 50, "yes": 50},
        row_count=100,
    )

    report = EDAAgent().analyze(profile)
    imbalance_findings = [
        f for f in report.findings if f.finding_type == "target_imbalance"
    ]
    assert len(imbalance_findings) == 0


def test_imbalance_majority_below_60_percent_is_not_flagged() -> None:
    """A majority below 60% should not be flagged as imbalanced."""
    profile = _make_profile(
        target_column="label",
        target_distribution={"a": 59, "b": 41},
        row_count=100,
    )

    report = EDAAgent().analyze(profile)
    imbalance_findings = [
        f for f in report.findings if f.finding_type == "target_imbalance"
    ]
    assert len(imbalance_findings) == 0


def test_imbalance_60_to_74_percent_importance_is_low() -> None:
    """Majority between 60% and 74.99% should be flagged as low importance."""
    profile = _make_profile(
        target_column="status",
        target_distribution={"normal": 70, "anomalous": 30},
        row_count=100,
    )

    report = EDAAgent().analyze(profile)
    imbalance_findings = [
        f for f in report.findings if f.finding_type == "target_imbalance"
    ]

    assert len(imbalance_findings) == 1
    attrs = _finding_attrs(imbalance_findings[0])
    assert attrs["importance"] == "low"
    assert attrs["finding_type"] == "target_imbalance"
    assert "normal" in attrs["evidence"]
    assert "70.00%" in attrs["evidence"]


def test_imbalance_75_to_89_percent_importance_is_medium() -> None:
    """Majority between 75% and 89.99% should be flagged as medium importance."""
    profile = _make_profile(
        target_column="outcome",
        target_distribution={"positive": 85, "negative": 15},
        row_count=100,
    )

    report = EDAAgent().analyze(profile)
    imbalance_findings = [
        f for f in report.findings if f.finding_type == "target_imbalance"
    ]

    assert len(imbalance_findings) == 1
    attrs = _finding_attrs(imbalance_findings[0])
    assert attrs["importance"] == "medium"
    assert "positive" in attrs["evidence"]
    assert "85.00%" in attrs["evidence"]


def test_imbalance_90_percent_or_more_importance_is_high() -> None:
    """Majority of 90% or more should be flagged as high importance."""
    profile = _make_profile(
        target_column="target",
        target_distribution={"majority": 95, "minority": 5},
        row_count=100,
    )

    report = EDAAgent().analyze(profile)
    imbalance_findings = [
        f for f in report.findings if f.finding_type == "target_imbalance"
    ]

    assert len(imbalance_findings) == 1
    attrs = _finding_attrs(imbalance_findings[0])
    assert attrs["importance"] == "high"
    assert "majority" in attrs["evidence"]
    assert "95.00%" in attrs["evidence"]


def test_no_target_supplied_produces_no_imbalance_finding() -> None:
    """When no target column is present, target imbalance should not be detected."""
    profile = _make_profile(row_count=100)

    report = EDAAgent().analyze(profile)
    imbalance_findings = [
        f for f in report.findings if f.finding_type == "target_imbalance"
    ]
    assert len(imbalance_findings) == 0


def test_imbalance_finding_interpretation_and_implication() -> None:
    """Target imbalance findings should include interpretation and modeling implication."""
    profile = _make_profile(
        target_column="is_fraud",
        target_distribution={"no": 92, "yes": 8},
        row_count=100,
    )

    report = EDAAgent().analyze(profile)
    imbalance_findings = [
        f for f in report.findings if f.finding_type == "target_imbalance"
    ]

    assert len(imbalance_findings) == 1
    attrs = _finding_attrs(imbalance_findings[0])
    assert "disproportionate share" in attrs["interpretation"].lower()
    assert "accuracy alone may be misleading" in attrs["modeling_implication"].lower()


# ---------------------------------------------------------------------------
# PART 2: Dominant category detection
# ---------------------------------------------------------------------------

def test_dominant_categorical_column_is_flagged() -> None:
    """A categorical column with >=80% of rows in one category should be flagged."""
    profile = _make_profile(
        row_count=100,
        categorical_statistics={
            "status": CategoricalStat(most_frequent_value="active", most_frequent_count=90)
        },
    )

    report = EDAAgent().analyze(profile)
    dominant_findings = [
        f for f in report.findings if f.finding_type == "dominant_category"
    ]

    assert len(dominant_findings) == 1
    attrs = _finding_attrs(dominant_findings[0])
    assert attrs["columns"] == ["status"]
    assert attrs["importance"] == "medium"
    assert "active" in attrs["evidence"]
    assert "90" in attrs["evidence"]
    assert "90.00%" in attrs["evidence"]


def test_dominant_category_below_80_percent_is_not_flagged() -> None:
    """A categorical column below 80% should not be flagged."""
    profile = _make_profile(
        row_count=100,
        categorical_statistics={
            "channel": CategoricalStat(most_frequent_value="web", most_frequent_count=70)
        },
    )

    report = EDAAgent().analyze(profile)
    dominant_findings = [
        f for f in report.findings if f.finding_type == "dominant_category"
    ]
    assert len(dominant_findings) == 0


def test_dominant_category_80_to_89_percent_importance_is_low() -> None:
    """Dominant ratio between 80% and 89.99% should be low importance."""
    profile = _make_profile(
        row_count=100,
        categorical_statistics={
            "source": CategoricalStat(most_frequent_value="organic", most_frequent_count=85)
        },
    )

    report = EDAAgent().analyze(profile)
    dominant_findings = [
        f for f in report.findings if f.finding_type == "dominant_category"
    ]

    assert len(dominant_findings) == 1
    assert _finding_attrs(dominant_findings[0])["importance"] == "low"


def test_dominant_category_90_to_94_percent_importance_is_medium() -> None:
    """Dominant ratio between 90% and 94.99% should be medium importance."""
    profile = _make_profile(
        row_count=100,
        categorical_statistics={
            "country": CategoricalStat(most_frequent_value="US", most_frequent_count=92)
        },
    )

    report = EDAAgent().analyze(profile)
    dominant_findings = [
        f for f in report.findings if f.finding_type == "dominant_category"
    ]

    assert len(dominant_findings) == 1
    assert _finding_attrs(dominant_findings[0])["importance"] == "medium"


def test_dominant_category_95_percent_and_above_importance_is_high() -> None:
    """Dominant ratio of 95% or more should be high importance."""
    profile = _make_profile(
        row_count=100,
        categorical_statistics={
            "currency": CategoricalStat(most_frequent_value="USD", most_frequent_count=98)
        },
    )

    report = EDAAgent().analyze(profile)
    dominant_findings = [
        f for f in report.findings if f.finding_type == "dominant_category"
    ]

    assert len(dominant_findings) == 1
    attrs = _finding_attrs(dominant_findings[0])
    assert attrs["importance"] == "high"
    assert "USD" in attrs["evidence"]


def test_dominant_category_zero_most_frequent_count_is_not_flagged() -> None:
    """A categorical column with zero most frequent count should not produce a finding."""
    profile = _make_profile(
        row_count=100,
        categorical_statistics={
            "empty_col": CategoricalStat(most_frequent_value=None, most_frequent_count=0)
        },
    )

    report = EDAAgent().analyze(profile)
    dominant_findings = [
        f for f in report.findings if f.finding_type == "dominant_category"
    ]
    assert len(dominant_findings) == 0


def test_dominant_category_interpretation_and_implication() -> None:
    """Dominant category findings should include interpretation and modeling implication."""
    profile = _make_profile(
        row_count=100,
        categorical_statistics={
            "flag": CategoricalStat(most_frequent_value="none", most_frequent_count=97)
        },
    )

    report = EDAAgent().analyze(profile)
    dominant_findings = [
        f for f in report.findings if f.finding_type == "dominant_category"
    ]

    assert len(dominant_findings) == 1
    attrs = _finding_attrs(dominant_findings[0])
    assert "heavily concentrated" in attrs["interpretation"].lower()
    assert "limited variation" in attrs["modeling_implication"].lower()


# ---------------------------------------------------------------------------
# Combined behavior
# ---------------------------------------------------------------------------

def test_combined_imbalance_and_dominant_category_findings() -> None:
    """Imbalanced target and dominant categorical column should both be reported."""
    profile = _make_profile(
        row_count=100,
        target_column="churn",
        target_distribution={"no": 88, "yes": 12},
        categorical_statistics={
            "segment": CategoricalStat(most_frequent_value="general", most_frequent_count=91)
        },
    )

    report = EDAAgent().analyze(profile)

    assert len(report.findings) == 2
    finding_types = {f.finding_type for f in report.findings}
    assert finding_types == {"target_imbalance", "dominant_category"}
    imbalance = [f for f in report.findings if f.finding_type == "target_imbalance"][0]
    dominant = [f for f in report.findings if f.finding_type == "dominant_category"][0]

    assert imbalance.columns == ["churn"]
    assert dominant.columns == ["segment"]
    assert report.summary == "2 EDA finding(s) detected."


def test_no_findings_summary_when_profile_is_balanced() -> None:
    """A clean balanced profile should produce no findings and the expected summary."""
    profile = _make_profile(
        row_count=100,
        target_column="churn",
        target_distribution={"no": 50, "yes": 50},
    )

    report = EDAAgent().analyze(profile)

    assert report.findings == []
    assert report.summary == "No EDA findings detected."


def test_summary_counts_total_findings() -> None:
    """The summary should report the total number of EDA findings detected."""
    profile = _make_profile(
        row_count=100,
        target_column="churn",
        target_distribution={"no": 90, "yes": 10},
        categorical_statistics={
            "flag": CategoricalStat(most_frequent_value="none", most_frequent_count=90)
        },
    )

    report = EDAAgent().analyze(profile)

    assert len(report.findings) == 2
    assert report.summary == "2 EDA finding(s) detected."