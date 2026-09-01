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
    """
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_row_count": int(df.duplicated().sum()),
    }
