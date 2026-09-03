"""Dataset profiling utilities for AEGIS."""

from typing import Any

import pandas as pd
from pydantic import BaseModel

from src.aegis.schemas import CategoricalStat, DatasetProfile, NumericStats


def profile_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> DatasetProfile:
    """Return a structured dataset profile.

    Args:
        df: The pandas DataFrame to profile.
        target_column: Optional column name to profile as the target variable.
            If provided, target_column and target_distribution will be populated.

    Returns:
        A DatasetProfile containing row/column counts, missingness,
        uniqueness, data types, numeric/categorical statistics, and
        optionally target column info.

    Raises:
        ValueError: If target_column is provided but does not exist in the DataFrame.
    """
    if target_column is not None and target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    missing_values_by_column: dict[str, int] = {
        col: int(df[col].isna().sum()) for col in df.columns
    }
    unique_values_by_column: dict[str, int] = {
        col: int(df[col].nunique()) for col in df.columns
    }
    data_types_by_column: dict[str, str] = {
        col: str(df[col].dtype) for col in df.columns
    }

    numeric_cols = df.select_dtypes(include="number").columns
    numeric_statistics: dict[str, NumericStats] = {}
    for col in numeric_cols:
        series = df[col]
        numeric_statistics[col] = NumericStats(
            mean=float(series.mean()) if not series.isna().all() else None,
            median=float(series.median()) if not series.isna().all() else None,
            std=float(series.std()) if not series.isna().all() else None,
            min=float(series.min()) if not series.isna().all() else None,
            max=float(series.max()) if not series.isna().all() else None,
        )

    categorical_cols = df.select_dtypes(exclude="number").columns
    categorical_statistics: dict[str, CategoricalStat] = {}
    for col in categorical_cols:
        series = df[col]
        counts = series.value_counts(dropna=True)
        if counts.empty:
            categorical_statistics[col] = CategoricalStat(
                most_frequent_value=None,
                most_frequent_count=0,
            )
        else:
            top_value = counts.index[0]
            top_count = int(counts.iloc[0])
            categorical_statistics[col] = CategoricalStat(
                most_frequent_value=top_value,
                most_frequent_count=top_count,
            )

    target_distribution: dict[str, int] | None = None
    if target_column is not None:
        target_series = df[target_column]
        target_distribution = {
            str(k): int(v) for k, v in target_series.value_counts(dropna=False).items()
        }

    return DatasetProfile(
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_row_count=int(df.duplicated().sum()),
        missing_values_by_column=missing_values_by_column,
        unique_values_by_column=unique_values_by_column,
        data_types_by_column=data_types_by_column,
        numeric_statistics=numeric_statistics,
        categorical_statistics=categorical_statistics,
        target_column=target_column,
        target_distribution=target_distribution,
    )
