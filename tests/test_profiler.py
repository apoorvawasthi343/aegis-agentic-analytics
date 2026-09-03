"""Tests for the AEGIS dataset profiler."""

import pandas as pd
import pytest

from src.aegis.profiler import profile_dataset
from src.aegis.schemas import DatasetProfile, NumericStats, CategoricalStat


def test_profile_dataset_basic_counts():
    """A DataFrame with 3 rows and 2 columns returns correct row/column counts."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    result = profile_dataset(df)

    assert isinstance(result, DatasetProfile)
    assert result.row_count == 3
    assert result.column_count == 2


def test_profile_dataset_duplicate_count():
    """A DataFrame with one duplicated row reports duplicate_row_count = 1."""
    df = pd.DataFrame({"x": [1, 2, 2], "y": ["a", "b", "b"]})

    result = profile_dataset(df)

    assert result.duplicate_row_count == 1


def test_profile_dataset_missing_values_by_column():
    """Missing value counts are reported per column correctly."""
    df = pd.DataFrame({"age": [24, None, 45], "income": [52000, 68000, 91000]})

    result = profile_dataset(df)

    assert result.missing_values_by_column["age"] == 1
    assert result.missing_values_by_column["income"] == 0


def test_profile_dataset_unique_values_by_column():
    """Unique value counts are reported per column and exclude missing values."""
    df = pd.DataFrame({
        "repeated": [1, 1, 2, 2],
        "all_unique": [10, 20, 30, 40],
        "with_missing": [1, 2, None, 2],
    })

    result = profile_dataset(df)

    assert result.unique_values_by_column["repeated"] == 2
    assert result.unique_values_by_column["all_unique"] == 4
    assert result.unique_values_by_column["with_missing"] == 2


def test_profile_dataset_data_types_by_column():
    """Column dtype strings are reported correctly for each type."""
    df = pd.DataFrame({
        "ints": [1, 2, 3],
        "floats": [1.5, 2.5, 3.5],
        "text": ["a", "b", "c"],
        "bools": [True, False, True],
    })

    result = profile_dataset(df)

    assert result.data_types_by_column["ints"] == "int64"
    assert result.data_types_by_column["floats"] == "float64"
    assert result.data_types_by_column["text"] == "str"
    assert result.data_types_by_column["bools"] == "bool"


def test_profile_dataset_numeric_statistics():
    """Numeric statistics are computed correctly and only include numeric columns."""
    df = pd.DataFrame({
        "ints": [10, 20, 30],
        "floats": [1.0, 2.0, 8.0],
        "text": ["a", "b", "c"],
        "with_missing": [4, None, 8],
    })

    result = profile_dataset(df)

    stats: dict[str, NumericStats] = result.numeric_statistics

    # Only numeric columns should appear
    assert set(stats.keys()) == {"ints", "floats", "with_missing"}
    assert "text" not in stats

    # Integer column
    assert stats["ints"].mean == pytest.approx(20.0)
    assert stats["ints"].median == pytest.approx(20.0)
    assert stats["ints"].std == pytest.approx(10.0)
    assert stats["ints"].min == 10.0
    assert stats["ints"].max == 30.0

    # Float column
    assert stats["floats"].mean == pytest.approx(3.6666666666666665)
    assert stats["floats"].median == pytest.approx(2.0)
    assert stats["floats"].min == 1.0
    assert stats["floats"].max == 8.0

    # Column with missing value (pandas ignores NaN)
    assert stats["with_missing"].mean == pytest.approx(6.0)
    assert stats["with_missing"].median == pytest.approx(6.0)
    assert stats["with_missing"].min == 4.0
    assert stats["with_missing"].max == 8.0
    # std with 2 values: sqrt(((4-6)^2 + (8-6)^2) / (2-1)) = sqrt(8) ≈ 2.828
    assert stats["with_missing"].std == pytest.approx(2.8284271247461903)


def test_profile_dataset_categorical_statistics():
    """Categorical statistics report most frequent value and count correctly."""
    df = pd.DataFrame({
        "repeated_cat": ["a", "a", "b", "b", "a"],
        "with_missing_cat": ["x", None, "x", "y", "x"],
        "numeric_col": [1, 2, 3, 4, 5],
        "all_missing_cat": [None, None, None, None, None],
    })

    result = profile_dataset(df)
    cat: dict[str, CategoricalStat] = result.categorical_statistics

    # Only non-numeric columns appear
    assert set(cat.keys()) == {"repeated_cat", "with_missing_cat", "all_missing_cat"}
    assert "numeric_col" not in cat

    # Repeated categorical column
    assert cat["repeated_cat"].most_frequent_value == "a"
    assert cat["repeated_cat"].most_frequent_count == 3

    # Categorical column with one missing value (missing ignored)
    assert cat["with_missing_cat"].most_frequent_value == "x"
    assert cat["with_missing_cat"].most_frequent_count == 3

    # All-missing categorical column handled safely
    assert cat["all_missing_cat"].most_frequent_value is None
    assert cat["all_missing_cat"].most_frequent_count == 0


def test_profile_dataset_returns_dataset_profile():
    """profile_dataset returns a DatasetProfile that can be serialized to JSON."""
    df = pd.DataFrame({"a": [1, 2, 3]})

    result = profile_dataset(df)

    assert isinstance(result, DatasetProfile)
    json_str = result.model_dump_json()
    assert isinstance(json_str, str)
    assert len(json_str) > 0


def test_profile_dataset_target_column_categorical():
    """A valid categorical target column reports target_column and target_distribution."""
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "label": ["Yes", "No", "Yes", "No"],
    })

    result = profile_dataset(df, target_column="label")

    assert result.target_column == "label"
    assert result.target_distribution == {"Yes": 2, "No": 2}


def test_profile_dataset_no_target_column():
    """When no target column is provided, target fields remain None."""
    df = pd.DataFrame({"x": [1, 2, 3]})

    result = profile_dataset(df)

    assert result.target_column is None
    assert result.target_distribution is None


def test_profile_dataset_missing_target_column():
    """A non-existent target column raises ValueError."""
    df = pd.DataFrame({"x": [1, 2, 3]})

    with pytest.raises(ValueError, match="not found in DataFrame"):
        profile_dataset(df, target_column="nonexistent")
