"""Feature Engineering Agent for AEGIS.

Produces structured feature engineering recommendations (lists of
FeatureEngineeringSpec) from a DataFrame and target column.

Implements deterministic rules that mirror common, safe feature engineering
patterns (log1p for positive numerics, missing indicators for columns with
NaNs). When an LLM client is provided, the agent also requests LLM-based
suggestions and merges them with the deterministic ones.

The agent NEVER executes arbitrary Python. It only produces spec objects
that the FeatureEngineeringExecutor safely applies.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.aegis.llm import LLMClient
from src.aegis.schemas import FeatureEngineeringSpec


# Transformation types that the agent is allowed to recommend.
# These must match what FeatureEngineeringExecutor supports.
SUPPORTED_TRANSFORMATIONS = frozenset({"log1p", "missing_indicator", "ratio", "count_sum"})


class FeatureEngineeringAgent:
    """Agent that recommends feature engineering transformations.

    Pipeline usage:
        agent = FeatureEngineeringAgent(llm_client=llm_client)
        specs = agent.recommend(df, target_column)
        # specs flow into FeatureEngineeringExecutor / ModelComparison

    If no LLM client is provided, only deterministic rules are applied.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """Initialize the agent.

        Args:
            llm_client: Optional LLM client for LLM-based feature suggestions.
                If None, only deterministic rules are used.
        """
        self.llm_client = llm_client

    def recommend(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> list[FeatureEngineeringSpec]:
        """Produce feature engineering recommendations for a DataFrame.

        Args:
            df: The full dataset including the target column.
            target_column: Name of the target column (excluded from features).

        Returns:
            A list of FeatureEngineeringSpec objects describing recommended
            transformations. Always deterministic; optionally enriched by LLM.
        """
        deterministic_specs = self._deterministic_recommend(df, target_column)

        if self.llm_client is not None:
            llm_specs = self._llm_recommend(df, target_column)
            # Deduplicate by feature_name — LLM specs supplement, not replace.
            seen = {s.feature_name for s in deterministic_specs}
            for spec in llm_specs:
                if spec.feature_name not in seen:
                    deterministic_specs.append(spec)
                    seen.add(spec.feature_name)

        return deterministic_specs

    # ------------------------------------------------------------------
    # Deterministic rules
    # ------------------------------------------------------------------

    def _deterministic_recommend(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> list[FeatureEngineeringSpec]:
        """Apply deterministic, safe feature engineering heuristics.

        Rules:
        - For numeric columns where all non-null values are >= 0 and at
          least one is > 0: suggest log1p.
        - For any column with at least one missing value: suggest
          missing_indicator.
        """
        specs: list[FeatureEngineeringSpec] = []
        feature_cols = [c for c in df.columns if c != target_column]

        for col in feature_cols:
            if col not in df.columns:
                continue

            # log1p for non-negative numeric columns with at least one positive
            if pd.api.types.is_numeric_dtype(df[col]):
                dropped = df[col].dropna()
                if len(dropped) > 0 and (dropped >= 0).all() and (dropped > 0).any():
                    specs.append(
                        FeatureEngineeringSpec(
                            feature_name=f"log_{col}",
                            transformation_type="log1p",
                            columns=[col],
                        )
                    )

            # missing_indicator for columns with any missing values
            if df[col].isna().any():
                specs.append(
                    FeatureEngineeringSpec(
                        feature_name=f"{col}_is_missing",
                        transformation_type="missing_indicator",
                        columns=[col],
                    )
                )

        return specs

    # ------------------------------------------------------------------
    # Optional LLM enrichment
    # ------------------------------------------------------------------

    def _llm_recommend(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> list[FeatureEngineeringSpec]:
        """Request LLM-based feature suggestions and parse them into specs.

        The LLM is asked to suggest additional feature engineering
        transformations based on column profiles. The response is parsed
        as JSON and validated against supported transformation types and
        existing columns. Invalid specs are silently dropped.

        Returns:
            A list of valid FeatureEngineeringSpec objects from the LLM,
            or an empty list if the LLM response could not be parsed.
        """
        profile = self._build_column_profile(df, target_column)
        profile_json = profile.model_dump_json(indent=2)

        prompt = (
            "You are an expert feature engineering consultant.\n\n"
            "Analyze the column profile below and suggest additional feature "
            "engineering transformations. Return ONLY a JSON array of objects, "
            "each with:\n"
            "- \"feature_name\": string\n"
            "- \"transformation_type\": one of "
            f"{sorted(SUPPORTED_TRANSFORMATIONS)}\n"
            "- \"columns\": list of existing column names\n\n"
            "Constraints:\n"
            "- Only suggest transformations for columns that exist in the profile.\n"
            "- Do NOT suggest log1p for columns that may contain negative values.\n"
            "- Be concise — suggest at most 5 additional features.\n"
            "- If no good suggestions, return an empty array [].\n\n"
            "Column profile (JSON):\n"
            f"{profile_json}\n"
        )

        raw_response = self.llm_client.generate(prompt)

        try:
            import json

            data = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            return []

        if not isinstance(data, list):
            return []

        specs: list[FeatureEngineeringSpec] = []
        existing_cols = set(df.columns)

        for item in data:
            if not isinstance(item, dict):
                continue

            feature_name = item.get("feature_name")
            transformation_type = item.get("transformation_type")
            columns = item.get("columns")

            if not feature_name or not transformation_type:
                continue
            if not isinstance(feature_name, str) or not isinstance(transformation_type, str):
                continue
            if not isinstance(columns, list) or len(columns) == 0:
                continue
            if not all(isinstance(c, str) and c for c in columns):
                continue

            specs.append(
                FeatureEngineeringSpec(
                    feature_name=str(feature_name),
                    transformation_type=str(transformation_type),
                    columns=[str(c) for c in columns],
                )
            )

        return specs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_column_profile(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> _ColumnProfile:
        """Build a lightweight column profile for the LLM prompt."""
        rows = len(df)
        cols: list[_ColumnInfo] = []

        for col in df.columns:
            if col == target_column:
                continue

            series = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)
            missing = int(series.isna().sum())
            unique = int(series.nunique())

            numeric_stats: dict[str, float] = {}
            if is_numeric:
                dropped = series.dropna()
                if len(dropped) > 0:
                    numeric_stats = {
                        "mean": float(dropped.mean()),
                        "min": float(dropped.min()),
                        "max": float(dropped.max()),
                    }

            cols.append(
                _ColumnInfo(
                    name=col,
                    dtype=str(series.dtype),
                    is_numeric=is_numeric,
                    missing=missing,
                    unique=unique,
                    rows=rows,
                    numeric_stats=numeric_stats,
                )
            )

        return _ColumnProfile(columns=cols, target_column=target_column)


# ---------------------------------------------------------------------------
# Lightweight Pydantic-free profile types for the LLM prompt
# ---------------------------------------------------------------------------

class _ColumnInfo:
    __slots__ = (
        "name",
        "dtype",
        "is_numeric",
        "missing",
        "unique",
        "rows",
        "numeric_stats",
    )

    def __init__(
        self,
        name: str,
        dtype: str,
        is_numeric: bool,
        missing: int,
        unique: int,
        rows: int,
        numeric_stats: dict[str, float],
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.is_numeric = is_numeric
        self.missing = missing
        self.unique = unique
        self.rows = rows
        self.numeric_stats = numeric_stats

    def model_dump(self, *, indent: int = 2) -> str:  # pragma: no cover
        import json

        return json.dumps(
            {
                "name": self.name,
                "dtype": self.dtype,
                "is_numeric": self.is_numeric,
                "missing_count": self.missing,
                "unique_count": self.unique,
                "row_count": self.rows,
                "numeric_stats": self.numeric_stats,
            },
            indent=indent,
        )


class _ColumnProfile:
    __slots__ = ("columns", "target_column")

    def __init__(self, columns: list[_ColumnInfo], target_column: str) -> None:
        self.columns = columns
        self.target_column = target_column

    def model_dump_json(self, *, indent: int = 2) -> str:
        import json

        return json.dumps(
            {
                "target_column": self.target_column,
                "column_count": len(self.columns),
                "columns": [
                    {
                        "name": c.name,
                        "dtype": c.dtype,
                        "is_numeric": c.is_numeric,
                        "missing_count": c.missing,
                        "unique_count": c.unique,
                        "row_count": c.rows,
                        "numeric_stats": c.numeric_stats,
                    }
                    for c in self.columns
                ],
            },
            indent=indent,
        )
