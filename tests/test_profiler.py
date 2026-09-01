"""Tests for the AEGIS dataset profiler."""

import pandas as pd
import pytest

from src.aegis.profiler import profile_dataset


def test_profile_dataset_basic_counts():
    """A DataFrame with 3 rows and 2 columns returns correct row/column counts."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    result = profile_dataset(df)

    assert result["row_count"] == 3
    assert result["column_count"] == 2


def test_profile_dataset_duplicate_count():
    """A DataFrame with one duplicated row reports duplicate_row_count = 1."""
    df = pd.DataFrame({"x": [1, 2, 2], "y": ["a", "b", "b"]})

    result = profile_dataset(df)

    assert result["duplicate_row_count"] == 1


def test_profile_dataset_missing_values_by_column():
    """Missing value counts are reported per column correctly."""
    df = pd.DataFrame({"age": [24, None, 45], "income": [52000, 68000, 91000]})

    result = profile_dataset(df)

    assert result["missing_values_by_column"]["age"] == 1
    assert result["missing_values_by_column"]["income"] == 0
