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
            - data_types_by_column: pandas dtype as a string for each column
            - numeric_statistics: mean, median, std, min, max for each numeric column
            - categorical_statistics: most frequent value and count for each non-numeric column
    """
    missing_values_by_column = {
        col: int(df[col].isna().sum()) for col in df.columns
    }
    unique_values_by_column = {
        col: int(df[col].nunique()) for col in df.columns
    }
    data_types_by_column = {
        col: str(df[col].dtype) for col in df.columns
    }

    numeric_cols = df.select_dtypes(include="number").columns
    numeric_statistics = {}
    for col in numeric_cols:
        series = df[col]
        numeric_statistics[col] = {
            "mean": float(series.mean()) if not series.isna().all() else None,
            "median": float(series.median()) if not series.isna().all() else None,
            "std": float(series.std()) if not series.isna().all() else None,
            "min": float(series.min()) if not series.isna().all() else None,
            "max": float(series.max()) if not series.isna().all() else None,
        }

    categorical_cols = df.select_dtypes(exclude="number").columns
    categorical_statistics = {}
    for col in categorical_cols:
        series = df[col]
        counts = series.value_counts(dropna=True)
        if counts.empty:
            categorical_statistics[col] = {
                "most_frequent_value": None,
                "most_frequent_count": 0,
            }
        else:
            top_value = counts.index[0]
            top_count = int(counts.iloc[0])
            categorical_statistics[col] = {
                "most_frequent_value": top_value,
                "most_frequent_count": top_count,
            }

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_row_count": int(df.duplicated().sum()),
        "missing_values_by_column": missing_values_by_column,
        "unique_values_by_column": unique_values_by_column,
        "data_types_by_column": data_types_by_column,
        "numeric_statistics": numeric_statistics,
        "categorical_statistics": categorical_statistics,
    }
