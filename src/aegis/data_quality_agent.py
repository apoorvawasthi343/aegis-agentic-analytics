"""Data Quality Agent for AEGIS automated dataset analysis."""

from src.aegis.schemas import DataQualityFinding, DataQualityReport, DatasetProfile


class DataQualityAgent:
    """Agent that analyzes dataset profiles and produces data quality reports.

    Currently implements deterministic missing-value detection. Additional
    checks (duplicates, cardinality, invalid values, etc.) and LLM-based
    analysis will be added in later milestones.
    """

    def analyze(self, profile: DatasetProfile) -> DataQualityReport:
        """Analyze a dataset profile and return a data quality report.

        Performs deterministic missing-value detection against
        ``profile.missing_values_by_column``.

        Args:
            profile: The DatasetProfile to analyze.

        Returns:
            A DataQualityReport containing missing-value findings (if any)
            and a summary line describing how many issues were detected.
        """
        findings: list[DataQualityFinding] = []
        row_count = profile.row_count or 1  # avoid division by zero

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

        summary = (
            f"{len(findings)} missing-value issue(s) detected."
            if findings
            else "No missing-value issues detected."
        )

        return DataQualityReport(
            findings=findings,
            summary=summary,
        )
