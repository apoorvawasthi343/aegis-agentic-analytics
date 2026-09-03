"""Data Quality Agent for AEGIS automated dataset analysis."""

from src.aegis.llm import LLMClient
from src.aegis.prompts import build_data_quality_prompt
from src.aegis.schemas import DataQualityFinding, DataQualityReport, DatasetProfile


class DataQualityAgent:
    """Agent that analyzes dataset profiles and produces data quality reports.

    Currently implements deterministic detection of:

    * Missing values per column (``profile.missing_values_by_column``)
    * Duplicate rows (``profile.duplicate_row_count``)
    * High-cardinality / possible identifier columns
      (``profile.unique_values_by_column``)
    * Constant columns / near-zero variance
      (``profile.unique_values_by_column``)

    If an LLM client is provided, the agent also requests LLM-based reasoning
    and merges those findings with the deterministic ones. If the LLM response
    cannot be validated, the deterministic findings are preserved and the
    summary notes the failure.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """Initialize the agent.

        Args:
            llm_client: Optional LLM client to use for LLM-based reasoning.
                If None, the agent runs only deterministic checks.
        """
        self.llm_client = llm_client

    def analyze(self, profile: DatasetProfile) -> DataQualityReport:
        """Analyze a dataset profile and return a data quality report.

        Performs deterministic detection of:

        * Missing values per column
        * Duplicate rows
        * High-cardinality / possible identifier columns
        * Constant columns / near-zero variance

        Optionally merges LLM-generated findings when an LLM client is
        available.

        Args:
            profile: The DatasetProfile to analyze.

        Returns:
            A DataQualityReport containing any findings and a summary line
            reporting the total number of issues detected.
        """
        findings: list[DataQualityFinding] = []
        row_count = profile.row_count or 1

        # --- Missing value detection ---
        for column, missing_count in profile.missing_values_by_column.items():
            if missing_count <= 0:
                continue

            missing_pct = (missing_count / row_count) * 100.0

            if missing_pct <= 5.0:
                severity = "low"
            elif missing_pct <= 20.0:
                severity = "medium"
            else:
                severity = "high"

            evidence = (
                f"Column '{column}' has {missing_count} missing value(s), "
                f"which is {missing_pct:.2f}% of the {row_count} rows in the dataset."
            )
            recommendation = (
                "Investigate an appropriate missing-value handling strategy "
                "(e.g. imputation, row removal, or model-based approaches) "
                "before proceeding with modeling."
            )

            findings.append(
                DataQualityFinding(
                    issue_type="missing_values",
                    severity=severity,
                    column=column,
                    evidence=evidence,
                    recommendation=recommendation,
                )
            )

        # --- Duplicate row detection ---
        duplicate_count = profile.duplicate_row_count
        if duplicate_count and duplicate_count > 0:
            duplicate_pct = (duplicate_count / row_count) * 100.0

            if duplicate_pct <= 2.0:
                severity = "low"
            elif duplicate_pct <= 10.0:
                severity = "medium"
            else:
                severity = "high"

            evidence = (
                f"The dataset contains {duplicate_count} duplicate row(s), "
                f"which is {duplicate_pct:.2f}% of the {row_count} rows."
            )
            recommendation = (
                "Investigate whether these duplicates represent legitimate "
                "repeated observations or should be removed before modeling."
            )

            findings.append(
                DataQualityFinding(
                    issue_type="duplicate_rows",
                    severity=severity,
                    column=None,
                    evidence=evidence,
                    recommendation=recommendation,
                )
            )

        # --- High cardinality / possible identifier detection ---
        actual_row_count = profile.row_count
        if actual_row_count > 0:
            for column, unique_count in profile.unique_values_by_column.items():
                if unique_count < 1:
                    continue

                cardinality_ratio = unique_count / actual_row_count

                if cardinality_ratio >= 0.95:
                    cardinality_pct = cardinality_ratio * 100.0

                    if cardinality_pct >= 99.0:
                        severity = "high"
                    else:
                        severity = "medium"

                    evidence = (
                        f"Column '{column}' has {unique_count} unique value(s) "
                        f"out of {actual_row_count} rows, which is {cardinality_pct:.2f}% "
                        f"of the dataset."
                    )
                    recommendation = (
                        "This column may be an identifier or too high-cardinality "
                        "for direct modeling. Review its role before feature "
                        "engineering (e.g. consider grouping rare values, target "
                        "encoding, or dropping it)."
                    )

                    findings.append(
                        DataQualityFinding(
                            issue_type="high_cardinality",
                            severity=severity,
                            column=column,
                            evidence=evidence,
                            recommendation=recommendation,
                        )
                    )

        # --- Constant column / near-zero variance detection ---
        # Only flag columns with exactly one unique non-null value.
        # Do not treat an entirely missing column (0 unique non-null values)
        # as a constant column in this rule.
        for column, unique_count in profile.unique_values_by_column.items():
            if unique_count == 1:
                evidence = (
                    f"Column '{column}' contains only 1 unique non-null value "
                    f"across the {actual_row_count} rows in the dataset."
                )
                recommendation = (
                    "This column provides little or no predictive information "
                    "and should be reviewed for removal before modeling "
                    "(unless it is a deliberately constant flag or control column)."
                )

                findings.append(
                    DataQualityFinding(
                        issue_type="constant_column",
                        severity="medium",
                        column=column,
                        evidence=evidence,
                        recommendation=recommendation,
                    )
                )

        # --- LLM-based reasoning (optional) ---
        llm_findings: list[DataQualityFinding] = []
        llm_error_note: str | None = None

        if self.llm_client is not None:
            prompt = build_data_quality_prompt(profile)
            raw_response = self.llm_client.generate(prompt)

            try:
                llm_report = DataQualityReport.model_validate_json(raw_response)
                llm_findings = list(llm_report.findings)
            except Exception:
                llm_error_note = (
                    "LLM reasoning was requested but the response could not be "
                    "validated. Deterministic findings are preserved."
                )

        # Merge deterministic and LLM findings.
        combined_findings = list(findings) + llm_findings

        # --- Summary ---
        if llm_error_note:
            if combined_findings:
                summary = (
                    f"{len(combined_findings)} data-quality issue(s) detected. "
                    f"{llm_error_note}"
                )
            else:
                summary = llm_error_note
        elif combined_findings:
            summary = f"{len(combined_findings)} data-quality issue(s) detected."
        else:
            summary = "No data-quality issues detected."

        return DataQualityReport(
            findings=combined_findings,
            summary=summary,
        )
