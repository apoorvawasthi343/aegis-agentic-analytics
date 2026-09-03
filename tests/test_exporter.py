"""Tests for the AEGIS export utilities."""

import json
import pytest

from pathlib import Path

from src.aegis.exporter import export_profile_to_json
from src.aegis.schemas import CategoricalStat, DatasetProfile, NumericStats


def test_export_profile_to_json_creates_file_and_preserves_fields(tmp_path):
    """Export creates the JSON file and preserves key fields on read-back."""
    profile = DatasetProfile(
        row_count=10,
        column_count=4,
        duplicate_row_count=1,
        missing_values_by_column={"a": 2, "b": 0},
        unique_values_by_column={"a": 5, "b": 3},
        data_types_by_column={"a": "int64", "b": "float64"},
        numeric_statistics={
            "a": NumericStats(mean=10.0, median=10.0, std=2.0, min=5.0, max=15.0),
        },
        categorical_statistics={
            "b": CategoricalStat(most_frequent_value="x", most_frequent_count=4),
        },
        target_column="label",
        target_distribution={"Yes": 6, "No": 4},
    )

    out_path = tmp_path / "profile" / "result.json"
    returned = export_profile_to_json(profile, out_path)

    # File is created
    assert returned.exists()
    assert returned == out_path

    # JSON can be read back
    data = json.loads(out_path.read_text())

    # Key fields preserved
    assert data["row_count"] == 10
    assert data["column_count"] == 4
    assert data["target_column"] == "label"
    assert data["target_distribution"] == {"Yes": 6, "No": 4}


def test_export_profile_to_json_nested_dirs_created(tmp_path):
    """Parent directories are created automatically when missing."""
    profile = DatasetProfile(
        row_count=1,
        column_count=1,
        duplicate_row_count=0,
        missing_values_by_column={},
        unique_values_by_column={},
        data_types_by_column={},
        numeric_statistics={},
        categorical_statistics={},
    )

    out_path = tmp_path / "a" / "b" / "c" / "profile.json"
    returned = export_profile_to_json(profile, out_path)

    assert returned.exists()
    assert returned.parent == tmp_path / "a" / "b" / "c"
