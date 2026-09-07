from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.aegis.llm import LLMClient
from src.aegis.schemas import FeatureEngineeringSpec, FeatureEngineeringSuggestions

# Transformation types that the agent is allowed to recommend.
SUPPORTED_TRANSFORMATIONS = frozenset(
    [
        "log1p",
        "log",
        "sqrt",
        "count_sum",
        "binary_to_int",
        "map_values",
        "reduce_categories",
        "ordinal_encode",
        "string_length",
        "datetime_extract",
        "bool_and",
        "bool_or",
        "count_nonzero",
        "sum_squares",
        "ratio",
        "difference",
        "product",
        "missing_indicator",
    ]
)

# Regex used to reject obviously invalid LLM-produced feature names.
_VALID_FEATURE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FeatureEngineeringAgent:
    """Agent that suggests and validates feature engineering transformations.

    Parameters
    ----------
    llm_client : LLMClient | None
        Optional LLM client for AI-assisted feature suggestions.
        When None, only deterministic rules are used.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def recommend(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> list[FeatureEngineeringSpec]:
        """Produce a list of validated feature engineering specifications.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset being analyzed. The agent inspects column profiles
            from this DataFrame.
        target_column : str
            The name of the target column, in case it needs to be excluded
            from suggestions.

        Returns
        -------
        list[FeatureEngineeringSpec]
            A list of validated feature engineering specifications.
        """
        # ------------------------------------------------------------------
        # Stage 1: deterministic recommendations (always present)
        # ------------------------------------------------------------------
        deterministic_specs = self._deterministic_recommendations(df, target_column)

        # ------------------------------------------------------------------
        # Stage 2: optional LLM enrichment
        # ------------------------------------------------------------------
        if self.llm_client is not None:
            llm_specs = self._llm_recommend(df, target_column)
            # Merge: deterministic first (stable ordering), then LLM
            # suggestions. The executor applies them in list order.
            deterministic_specs.extend(llm_specs)

        return deterministic_specs

    # ------------------------------------------------------------------
    # Deterministic rules
    # ------------------------------------------------------------------

    def _deterministic_recommendations(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> list[FeatureEngineeringSpec]:
        """Return deterministic feature suggestions based on column profiles.

        Deterministic rules cover common-sense transformations that do not
        require an LLM: log1p for positive numeric columns with right-skew,
        ratio/difference/product for pairs of numeric columns, etc.

        These rules are the fallback when `self.llm_client is None` and
        the primary source of suggestions otherwise.
        """
        specs: list[FeatureEngineeringSpec] = []

        # 1. log1p for positive numeric columns (> 0 min across non-null)
        for col in df.columns:
            if col == target_column:
                continue

            series = df[col].dropna()
            if len(series) == 0:
                continue

            if not pd.api.types.is_numeric_dtype(series):
                continue

            if series.min() > 0:
                # Positive-only column: log1p safe
                specs.append(
                    FeatureEngineeringSpec(
                        feature_name=f"log1p_{col}",
                        transformation_type="log1p",
                        columns=[col],
                        parameters={},
                    )
                )
            elif series.min() >= 0 and series.max() > 0:
                # Non-negative with positive max: log1p safe (0 -> log(1)=0)
                specs.append(
                    FeatureEngineeringSpec(
                        feature_name=f"log1p_{col}",
                        transformation_type="log1p",
                        columns=[col],
                        parameters={},
                    )
                )

        # 2. ratio / difference / product for pairs of numeric columns that
        #    are not ID-like and not the target
        numeric_cols = [
            c
            for c in df.columns
            if c != target_column and pd.api.types.is_numeric_dtype(df[c])
        ]

        # ratio: for pairs where both are positive (or non-negative with
        # positive max) and neither is an obvious ID column (name is not
        # "id", "customer_id", "user_id", etc.)
        id_like_names = frozenset(
            {
                "id",
                "customer_id",
                "user_id",
                "account_id",
                "client_id",
                "customerid",
            }
        )

        for i, col_a in enumerate(numeric_cols):
            if col_a in id_like_names:
                continue
            if df[col_a].dropna().min() < 0:
                continue

            for col_b in numeric_cols[i + 1 :]:
                if col_b in id_like_names:
                    continue
                if df[col_b].dropna().min() < 0:
                    continue

                # ratio: col_a / col_b
                specs.append(
                    FeatureEngineeringSpec(
                        feature_name=f"ratio_{col_a}_to_{col_b}",
                        transformation_type="ratio",
                        columns=[col_a, col_b],
                        parameters={},
                    )
                )

                # difference: col_a - col_b
                specs.append(
                    FeatureEngineeringSpec(
                        feature_name=f"diff_{col_a}_minus_{col_b}",
                        transformation_type="difference",
                        columns=[col_a, col_b],
                        parameters={},
                    )
                )

                # product: col_a * col_b
                specs.append(
                    FeatureEngineeringSpec(
                        feature_name=f"product_{col_a}_times_{col_b}",
                        transformation_type="product",
                        columns=[col_a, col_b],
                        parameters={},
                    )
                )

        # 3. missing_indicator for columns with any missing values
        for col in df.columns:
            if col == target_column:
                continue
            if df[col].isna().sum() > 0:
                specs.append(
                    FeatureEngineeringSpec(
                        feature_name=f"{col}_is_missing",
                        transformation_type="missing_indicator",
                        columns=[col],
                        parameters={},
                    )
                )

        return specs

    # ------------------------------------------------------------------
    # LLM-based suggestions
    # ------------------------------------------------------------------

    def _llm_recommend(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> list[FeatureEngineeringSpec]:
        """Request additional feature suggestions from the LLM.

        The LLM is prompted with a column profile and asked to return JSON
        in the FeatureEngineeringSuggestions schema. The response is
        validated against the schema and filtered against supported
        transformations and existing columns.

        If the LLM is unreachable or the response is malformed, the method
        returns an empty list. It NEVER raises an exception to the caller.
        """
        profile = self._build_column_profile(df, target_column)
        profile_json = profile.model_dump_json(indent=2)

        prompt = (
            "You are an expert data scientist working on a feature "
            "engineering task for a tabular classification dataset.\n\n"
            "Below is a column profile of the dataset. Suggest a small set "
            "of additional feature engineering transformations that may "
            "improve model performance.\n\n"
            "Follow these rules STRICTLY:\n"
            "- Only suggest transformations from this exact set: "
            + ", ".join(sorted(SUPPORTED_TRANSFORMATIONS))
            + ".\n"
            "- Each suggestion must be a JSON object with keys:\n"
            "    - feature_name: a valid Python identifier (letters, digits, "
            "underscores, no spaces, no hyphens)\n"
            "    - transformation_type: one of the supported types listed above\n"
            "    - columns: a list of existing column names (strings) from the "
            "dataset\n"
            "    - parameters: an optional JSON object of additional parameters\n"
            "- Do NOT suggest a transformation with columns that do not exist "
            "in the dataset.\n"
            "- Do NOT suggest the target column itself as an input column.\n"
            "- Do NOT suggest log1p for columns that may contain negative values.\n"
            "- Be concise — suggest at most 5 additional features.\n"
            "- If no good suggestions, return an empty array [].\n\n"
            "Column profile (JSON):\n"
            f"{profile_json}\n"
        )

        raw_response = self.llm_client.generate(
            prompt,
            response_schema=FeatureEngineeringSuggestions,
        )

        try:
            suggestions = FeatureEngineeringSuggestions.model_validate_json(raw_response)
            data = suggestions.specs
        except Exception:
            return []

        if not isinstance(data, list):
            return []

        specs: list[FeatureEngineeringSpec] = []
        existing_cols = set(df.columns)

        for item in data:
            if not isinstance(item, FeatureEngineeringSpec):
                continue

            feature_name = item.feature_name
            transformation_type = item.transformation_type
            columns = item.columns

            # Validate against supported transformations
            if transformation_type not in SUPPORTED_TRANSFORMATIONS:
                continue

            # Validate columns exist in the DataFrame
            if not columns or not all(c in existing_cols for c in columns):
                continue

            specs.append(
                FeatureEngineeringSpec(
                    feature_name=feature_name,
                    transformation_type=transformation_type,
                    columns=list(columns),
                    parameters=item.parameters,
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
