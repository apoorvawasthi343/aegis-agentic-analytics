"""Tests for the AEGIS CSV loader."""

import os

import pandas as pd
import pytest

from src.aegis.loader import load_csv


def test_load_csv_returns_dataframe(tmp_path):
    """A valid CSV file loads successfully and returns a pandas DataFrame."""
    csv_file = tmp_path / "customers.csv"
    csv_file.write_text(
        "customer_id,age,income,city,churn\n"
        "1,24,52000,Tampa,No\n"
        "2,31,68000,Miami,No\n"
    )

    result = load_csv(str(csv_file))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ["customer_id", "age", "income", "city", "churn"]


def test_load_csv_missing_file():
    """A missing CSV file raises FileNotFoundError."""
    missing_path = "/tmp/this_file_does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(missing_path)
