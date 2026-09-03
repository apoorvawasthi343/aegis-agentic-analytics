"""Export utilities for AEGIS dataset profiling results."""

from pathlib import Path

from src.aegis.schemas import DatasetProfile


def export_profile_to_json(profile: DatasetProfile, output_path: str | Path) -> Path:
    """Export a DatasetProfile to a human-readable JSON file.

    Args:
        profile: The DatasetProfile to export.
        output_path: File path where the JSON will be written.
            Parent directories are created automatically if they don't exist.

    Returns:
        The Path to the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_text = profile.model_dump_json(indent=2)
    path.write_text(json_text, encoding="utf-8")

    return path
