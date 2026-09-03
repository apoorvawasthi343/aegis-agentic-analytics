"""Data Quality Agent for AEGIS automated dataset analysis."""

from src.aegis.schemas import DataQualityFinding, DataQualityReport, DatasetProfile


class DataQualityAgent:
    """Agent that analyzes dataset profiles and produces data quality reports.

    Currently implements deterministic detection of:

    * Missing values per column (``profile.missing_values_by_column``)
    * Duplicate rows (``profile.duplicate_row_count``)
    * High-cardinality / possible identifier columns
      (``profile.unique_values_by_column``)

    Additional checks (invalid values, etc.) and LLM-based analysis will be
    added in later milestones.
    """

    def analyze(self, profile: DatasetProfile) -> DataQualityReport:
        """Analyze a dataset profile and return a data quality report.

        Performs deterministic detection of:

        * Missing values per column
        * Duplicate rows
        * High-cardinality / possible identifier columns

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

        # --- Summary ---
        if findings:
            summary = f"{len(findings)} data-quality issue(s) detected."
        else:
            summary = "No data-quality issues detected."

        return DataQualityReport(
            findings=findings,
            summary=summary,
        )
