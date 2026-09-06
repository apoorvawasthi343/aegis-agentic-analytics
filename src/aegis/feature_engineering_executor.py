"""Deterministic feature engineering executor for AEGIS.

Applies a safe whitelist of feature engineering transformations to a
pandas DataFrame. The executor never executes arbitrary Python code;
it only supports pre-approved transformation types.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

from src.aegis.schemas import (
    AppliedFeature,
    FeatureEngineeringReport,
    FeatureEngineeringSpec,
    SkippedFeature,
)


class FeatureEngineeringExecutor:
    """Executor that applies safe feature engineering transformations.

    Supports the following transformation types:

    - ``log1p``: natural log + 1 transform on a non-negative numeric column
    - ``ratio``: ratio of two numeric columns
    - ``missing_indicator``: binary flag for missing values in a column
    - ``count_sum``: sum across explicitly named compatible numeric columns

    If a requested transformation cannot be applied safely (missing columns,
    wrong dtypes, negative values for log1p, etc.), it is skipped with a
    recorded reason.
    """

    SUPPORTED_TRANSFORMATIONS = frozenset({
        "log1p",
        "ratio",
        "missing_indicator",
        "count_sum",
    })

    def apply(
        self,
        df: pd.DataFrame,
        specs: list[FeatureEngineeringSpec],
    ) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
        """Apply the requested feature engineering transformations.

        Args:
            df: The original DataFrame to transform.
            specs: List of FeatureEngineeringSpec objects describing the
                transformations to attempt.

        Returns:
            A tuple of (engineered_df, report) where engineered_df is the
            transformed DataFrame and report describes what was applied
            and what was skipped.
        """
        engineered_df = df.copy()
        applied_features: list[AppliedFeature] = []
        skipped_features: list[SkippedFeature] = []

        for spec in specs:
            if spec.transformation_type not in self.SUPPORTED_TRANSFORMATIONS:
                skipped_features.append(
                    SkippedFeature(
                        feature_name=spec.feature_name,
                        transformation_type=spec.transformation_type,
                        columns=spec.columns,
                        reason=f"Unsupported transformation type: {spec.transformation_type}",
                    )
                )
                continue

            handler = getattr(self, f"_apply_{spec.transformation_type}", None)
            if handler is None:
                skipped_features.append(
                    SkippedFeature(
                        feature_name=spec.feature_name,
                        transformation_type=spec.transformation_type,
                        columns=spec.columns,
                        reason=f"No handler for transformation type: {spec.transformation_type}",
                    )
                )
                continue

            result = handler(engineered_df, spec)
            if result is None:
                continue

            if isinstance(result, SkippedFeature):
                skipped_features.append(result)
            elif isinstance(result, AppliedFeature):
                applied_features.append(result)
            else:
                pass

        report = FeatureEngineeringReport(
            original_row_count=df.shape[0],
            original_column_count=df.shape[1],
            engineered_row_count=engineered_df.shape[0],
            engineered_column_count=engineered_df.shape[1],
            applied_features=applied_features,
            skipped_features=skipped_features,
            summary=(
                f"Applied {len(applied_features)} feature(s) and skipped "
                f"{len(skipped_features)} feature(s)."
            ),
        )

        return engineered_df, report

    # ---------------------------------------------------------------------------
    # Transformation handlers
    # ---------------------------------------------------------------------------

    def _apply_log1p(
        self,
        df: pd.DataFrame,
        spec: FeatureEngineeringSpec,
    ) -> AppliedFeature | SkippedFeature | None:
        """Apply log1p to a non-negative numeric column."""
        if len(spec.columns) != 1:
            return SkippedFeature(
                feature_name=spec.feature_name,
                transformation_type="log1p",
                columns=spec.columns,
                reason="log1p requires exactly one source column.",
            )

        col = spec.columns[0]
        if col not in df.columns:
            return SkippedFeature(
                feature_name=spec.feature_name,
                transformation_type="log1p",
                columns=spec.columns,
                reason=f"Source column '{col}' does not exist.",
            )

        if not pd.api.types.is_numeric_dtype(df[col]):
            return SkippedFeature(
                feature_name=spec.feature_name,
                transformation_type="log1p",
                columns=spec.columns,
                reason=f"Source column '{col}' is not numeric.",
            )

        if (df[col].dropna() < 0).any():
            return SkippedFeature(
                feature_name=spec.feature_name,
                transformation_type="log1p",
                columns=spec.columns,
                reason=f"Source column '{col}' contains negative values; log1p requires non-negative data.",
            )

        result_col = spec.feature_name
        df[result_col] = np.log1p(df[col])
        return AppliedFeature(
            feature_name=spec.feature_name,
            transformation_type="log1p",
            columns=[col],
            result_column=result_col,
        )

    def _apply_ratio(
        self,
        df: pd.DataFrame,
        spec: FeatureEngineeringSpec,
    ) -> AppliedFeature | SkippedFeature | None:
        """Compute ratio between two numeric columns."""
        if len(spec.columns) != 2:
            return SkippedFeature(
                feature_name=spec.feature_name,
                transformation_type="ratio",
                columns=spec.columns,
                reason="ratio requires exactly two source columns (numerator, denominator).",
            )

        num_col, den_col = spec.columns
        for col, role in [(num_col, "numerator"), (den_col, "denominator")]:
            if col not in df.columns:
                return SkippedFeature(
                    feature_name=spec.feature_name,
                    transformation_type="ratio",
                    columns=spec.columns,
                    reason=f"{role} column '{col}' does not exist.",
                )
            if not pd.api.types.is_numeric_dtype(df[col]):
                return SkippedFeature(
                    feature_name=spec.feature_name,
                    transformation_type="ratio",
                    columns=spec.columns,
                    reason=f"{role} column '{col}' is not numeric.",
                )

        result_col = spec.feature_name
        # Avoid division by zero by filling with NaN where denominator is zero
        denominator = df[den_col].replace(0, np.nan)
        df[result_col] = df[num_col] / denominator
        return AppliedFeature(
            feature_name=spec.feature_name,
            transformation_type="ratio",
            columns=[num_col, den_col],
            result_column=result_col,
        )

    def _apply_missing_indicator(
        self,
        df: pd.DataFrame,
        spec: FeatureEngineeringSpec,
    ) -> AppliedFeature | SkippedFeature | None:
        """Create a binary indicator for missing values in a column."""
        if len(spec.columns) != 1:
            return SkippedFeature(
                feature_name=spec.feature_name,
                transformation_type="missing_indicator",
                columns=spec.columns,
                reason="missing_indicator requires exactly one source column.",
            )

        col = spec.columns[0]
        if col not in df.columns:
            return SkippedFeature(
                feature_name=spec.feature_name,
                transformation_type="missing_indicator",
                columns=spec.columns,
                reason=f"Source column '{col}' does not exist.",
            )

        result_col = spec.feature_name
        df[result_col] = df[col].isna().astype(int)
        return AppliedFeature(
            feature_name=spec.feature_name,
            transformation_type="missing_indicator",
            columns=[col],
            result_column=result_col,
        )

    def _apply_count_sum(
        self,
        df: pd.DataFrame,
        spec: FeatureEngineeringSpec,
    ) -> AppliedFeature | SkippedFeature | None:
        """Sum across explicitly named compatible numeric columns."""
        if len(spec.columns) < 2:
            return SkippedFeature(
                feature_name=spec.feature_name,
                transformation_type="count_sum",
                columns=spec.columns,
                reason="count_sum requires at least two source columns.",
            )

        for col in spec.columns:
            if col not in df.columns:
                return SkippedFeature(
                    feature_name=spec.feature_name,
                    transformation_type="count_sum",
                    columns=spec.columns,
                    reason=f"Source column '{col}' does not exist.",
                )
            if not pd.api.types.is_numeric_dtype(df[col]):
                return SkippedFeature(
                    feature_name=spec.feature_name,
                    transformation_type="count_sum",
                    columns=spec.columns,
                    reason=f"Source column '{col}' is not numeric.",
                )

        result_col = spec.feature_name
        df[result_col] = df[spec.columns].sum(axis=1)
        return AppliedFeature(
            feature_name=spec.feature_name,
            transformation_type="count_sum",
            columns=list(spec.columns),
            result_column=result_col,
        )