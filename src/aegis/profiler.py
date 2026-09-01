"""Dataset profiling utilities for AEGIS."""

import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """Return basic dataset-level profiling metrics.

    Args:
        df: The pandas DataFrame to profile.

    Returns:
        A dictionary with keys:
            - row_count: total number of rows
            - column_count: total number of columns
            - duplicate_row_count: number of fully duplicated rows
            - missing_values_by_column: number of missing values per column
            - unique_values_by_column: number of unique non-null values per column
    """
    missing_values_by_column = {
        col: int(df[col].isna().sum()) for col in df.columns
    }
    unique_values_by_column = {
        col: int(df[col].nunique()) for col in df.columns
    }
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_row_count": int(df.duplicated().sum()),
        "missing_values_by_column": missing_values_by_column,
        "unique_values_by_column": unique_values_by_column,
    }
