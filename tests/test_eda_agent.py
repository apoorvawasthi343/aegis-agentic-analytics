"""Tests for the AEGIS Exploratory Data Analysis Agent."""

import json

import pytest

from src.aegis.eda_agent import EDAAgent
from src.aegis.llm import LLMClient
from src.aegis.schemas import (
    CategoricalStat,
    DatasetProfile,
    EDAFinding,
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


# ---------------------------------------------------------------------------
# LLM integration tests
# ---------------------------------------------------------------------------

class FakeLLMClient(LLMClient):
    """Minimal fake LLM client for EDA tests."""

    def __init__(self, response_text: str = "") -> None:
        self.response_text = response_text
        self.received_prompt: str | None = None
        self.received_schema: type | None = None

    def generate(
        self,
        prompt: str,
        response_schema: type | None = None,
    ) -> str:
        self.received_prompt = prompt
        self.received_schema = response_schema
        return self.response_text


def test_eda_agent_works_without_llm_client() -> None:
    """EDAAgent should still work when no LLM client is provided."""
    profile = _make_profile(
        target_column="churn",
        target_distribution={"no": 90, "yes": 10},
        row_count=100,
    )

    agent = EDAAgent()
    report = agent.analyze(profile)

    imbalance_findings = [
        f for f in report.findings if f.finding_type == "target_imbalance"
    ]
    assert len(imbalance_findings) == 1
    assert imbalance_findings[0].importance == "high"
    assert "LLM" not in report.summary


def test_eda_llm_finding_is_merged_into_report() -> None:
    """With a fake LLM client, the LLM EDA finding is added to the final report."""
    llm_client = FakeLLMClient(
        response_text=json.dumps(
            {
                "findings": [
                    {
                        "finding_type": "possible_predictive_signal",
                        "importance": "medium",
                        "columns": ["age", "income"],
                        "evidence": "The profile shows a moderate numeric spread in both columns.",
                        "interpretation": "Age and income may carry useful signal for modeling.",
                        "modeling_implication": "Consider exploring interactions between these features.",
                    }
                ],
                "summary": "LLM EDA analysis completed.",
            }
        )
    )

    profile = _make_profile(
        row_count=100,
        categorical_statistics={
            "status": CategoricalStat(most_frequent_value="active", most_frequent_count=90)
        },
    )

    agent = EDAAgent(llm_client=llm_client)
    report = agent.analyze(profile)

    finding_types = {f.finding_type for f in report.findings}
    assert "dominant_category" in finding_types
    assert "possible_predictive_signal" in finding_types

    assert len(report.findings) == 2
    assert report.summary == "2 EDA finding(s) detected."

    # Verify the LLM client received EDAReport as the response_schema
    assert llm_client.received_schema is EDAReport


def test_eda_prompt_is_passed_to_fake_client() -> None:
    """The EDA prompt built from the profile is actually passed to the LLM client."""
    llm_client = FakeLLMClient(
        response_text=json.dumps({"findings": [], "summary": ""})
    )

    profile = _make_profile(row_count=10)

    agent = EDAAgent(llm_client=llm_client)
    agent.analyze(profile)

    assert llm_client.received_prompt is not None
    assert "row_count" in llm_client.received_prompt
    assert "10" in llm_client.received_prompt
    assert "exploratory data analyst" in llm_client.received_prompt


def test_eda_invalid_llm_json_does_not_crash_agent() -> None:
    """Invalid JSON from the EDA LLM does not crash the agent."""
    llm_client = FakeLLMClient(response_text="this is not json")

    profile = _make_profile(row_count=10)

    agent = EDAAgent(llm_client=llm_client)
    report = agent.analyze(profile)

    assert len(report.findings) == 0
    assert "LLM reasoning was requested" in report.summary


def test_eda_deterministic_findings_survive_llm_failure() -> None:
    """Deterministic EDA findings are still present when LLM validation fails."""
    llm_client = FakeLLMClient(response_text="not valid json")

    profile = _make_profile(
        row_count=100,
        categorical_statistics={
            "flag": CategoricalStat(most_frequent_value="none", most_frequent_count=90)
        },
    )

    agent = EDAAgent(llm_client=llm_client)
    report = agent.analyze(profile)

    dominant_findings = [
        f for f in report.findings if f.finding_type == "dominant_category"
    ]
    assert len(dominant_findings) == 1
    assert dominant_findings[0].importance == "medium"

    assert "LLM reasoning was requested" in report.summary


def test_eda_valid_structured_llm_json_is_accepted() -> None:
    """Valid structured JSON matching EDAReport is accepted."""
    valid_json = EDAReport(
        findings=[
            EDAFinding(
                finding_type="llm_eda_finding",
                importance="low",
                columns=["validated_column"],
                evidence="This finding was accepted by Pydantic validation.",
                interpretation="An LLM-derived EDA observation.",
                modeling_implication="Use validated findings.",
            )
        ],
        summary="LLM validated structured EDA output.",
    ).model_dump_json()

    llm_client = FakeLLMClient(response_text=valid_json)

    profile = _make_profile(row_count=10)

    agent = EDAAgent(llm_client=llm_client)
    report = agent.analyze(profile)

    llm_findings = [
        f for f in report.findings if f.finding_type == "llm_eda_finding"
    ]
    assert len(llm_findings) == 1
    assert llm_findings[0].columns == ["validated_column"]


def test_eda_invalid_structured_output_falls_back_safely() -> None:
    """Invalid structured LLM output does not crash; deterministic findings remain."""
    # Simulate LLM returning malformed JSON that would fail Pydantic validation
    llm_client = FakeLLMClient(response_text='{"invalid": "structure"}')

    profile = _make_profile(
        row_count=100,
        categorical_statistics={
            "category": CategoricalStat(most_frequent_value="A", most_frequent_count=92)
        },
    )

    agent = EDAAgent(llm_client=llm_client)
    report = agent.analyze(profile)

    # Deterministic findings must still be present
    dominant_findings = [
        f for f in report.findings if f.finding_type == "dominant_category"
    ]
    assert len(dominant_findings) == 1
    assert dominant_findings[0].importance == "medium"

    # LLM findings should NOT be present (validation failed)
    llm_findings = [
        f for f in report.findings if f.finding_type not in ("dominant_category",)
    ]
    assert len(llm_findings) == 0

    # Summary should note the LLM validation failure
    assert "LLM reasoning was requested" in report.summary