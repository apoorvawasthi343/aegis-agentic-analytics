"""Exploratory Data Analysis Agent for AEGIS.

The EDAAgent analyzes dataset profiles and produces structured
exploratory data analysis reports describing patterns, distributions,
relationships, imbalance, skewness, and potential predictive signals.
"""

from src.aegis.schemas import DatasetProfile, EDAFinding, EDAReport


class EDAAgent:
    """Agent that analyzes dataset profiles and produces EDA reports.

    Implements deterministic EDA rules for target imbalance and dominant
    categorical columns. In later milestone steps it may be extended with
    additional EDA rules and optional LLM reasoning.
    """

    def analyze(self, profile: DatasetProfile) -> EDAReport:
        """Analyze a dataset profile and return an exploratory data analysis report.

        Performs deterministic detection of:

        - Target class imbalance (when target_column and target_distribution are
          available).
        - Dominant categorical columns (when categorical_statistics and row_count
          are available).

        Args:
            profile: The DatasetProfile to analyze.

        Returns:
            An EDAReport containing any findings and a summary line.
        """
        findings: list[EDAFinding] = []

        # --- Target imbalance detection (deterministic) ---
        target_column = profile.target_column
        target_distribution = profile.target_distribution

        if target_column is not None and target_distribution:
            total_target_count = sum(target_distribution.values())

            if total_target_count > 0 and len(target_distribution) >= 2:
                largest_class = max(
                    target_distribution.items(), key=lambda item: item[1]
                )
                largest_class_name, largest_class_count = largest_class
                majority_ratio = largest_class_count / total_target_count

                if majority_ratio >= 0.60:
                    if majority_ratio < 0.75:
                        importance = "low"
                    elif majority_ratio < 0.90:
                        importance = "medium"
                    else:
                        importance = "high"

                    evidence = (
                        f"The target class '{largest_class_name}' represents "
                        f"{largest_class_count} out of {total_target_count} "
                        f"target observations, which is {majority_ratio * 100:.2f}% "
                        f"of the dataset."
                    )
                    interpretation = (
                        "One target class represents a disproportionate share of the "
                        "observations, which may indicate class imbalance."
                    )
                    modeling_implication = (
                        "Accuracy alone may be misleading for this target. Consider "
                        "class-sensitive evaluation metrics and imbalance handling "
                        "strategies (e.g. class weighting, resampling, or threshold "
                        "tuning) during modeling."
                    )

                    findings.append(
                        EDAFinding(
                            finding_type="target_imbalance",
                            importance=importance,
                            columns=[target_column],
                            evidence=evidence,
                            interpretation=interpretation,
                            modeling_implication=modeling_implication,
                        )
                    )

        # --- Dominant category detection (deterministic) ---
        row_count = profile.row_count or 1

        for column, stats in profile.categorical_statistics.items():
            if stats is None:
                continue

            most_frequent_count = stats.most_frequent_count
            if most_frequent_count <= 0:
                continue

            dominant_ratio = most_frequent_count / row_count

            if dominant_ratio >= 0.80:
                if dominant_ratio >= 0.95:
                    importance = "high"
                elif dominant_ratio >= 0.90:
                    importance = "medium"
                else:
                    importance = "low"

                most_frequent_value = stats.most_frequent_value
                evidence = (
                    f"The categorical column '{column}' is dominated by the value "
                    f"'{most_frequent_value}', which appears in "
                    f"{most_frequent_count} out of {row_count} rows "
                    f"({dominant_ratio * 100:.2f}% of the dataset)."
                )
                interpretation = (
                    f"The feature '{column}' is heavily concentrated in a single "
                    f"category, which may indicate limited variation or a nearly "
                    f"constant column."
                )
                modeling_implication = (
                    f"The feature '{column}' may contain limited variation. Evaluate "
                    f"its predictive usefulness before including it in models "
                    f"(e.g. consider grouping rare categories or dropping it if it "
                    f"does not provide meaningful signal)."
                )

                findings.append(
                    EDAFinding(
                        finding_type="dominant_category",
                        importance=importance,
                        columns=[column],
                        evidence=evidence,
                        interpretation=interpretation,
                        modeling_implication=modeling_implication,
                    )
                )

        # --- Summary ---
        if findings:
            summary = f"{len(findings)} EDA finding(s) detected."
        else:
            summary = "No EDA findings detected."

        return EDAReport(
            findings=findings,
            summary=summary,
        )