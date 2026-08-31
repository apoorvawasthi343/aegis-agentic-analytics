"""Dataset loading utilities for AEGIS."""

import os

import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file to load.

    Returns:
        A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    return pd.read_csv(file_path)
