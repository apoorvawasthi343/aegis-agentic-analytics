"""Prompt construction for LLM-based data quality and EDA analysis."""

from src.aegis.schemas import DatasetProfile


def build_data_quality_prompt(profile: DatasetProfile) -> str:
    """Build a prompt that asks an LLM to analyze a dataset profile.

    The prompt includes the full structured profile and instructs the model
    to return JSON with findings and a summary.
    """
    profile_json = profile.model_dump_json(indent=2)

    prompt = f"""
You are an expert data-quality analyst.

Analyze the dataset profile below and return a JSON object with:
- "findings": a list of issues, each with:
    - "issue_type": short machine-readable type
    - "severity": "low", "medium", or "high"
    - "column": column name or null if not column-specific
    - "evidence": what in the profile supports this finding
    - "recommendation": concrete next step
- "summary": a one-line overall assessment

Dataset profile (JSON):
{profile_json}

Instructions:
- Base your findings ONLY on the supplied profile. Do not invent statistics.
- Distinguish evidence from recommendations.
- Look for:
    - suspicious missingness patterns
    - possible identifier columns (high cardinality, near-unique values)
    - problematic cardinality or constant/near-constant columns
    - unusual distributions or extreme values
    - potential modeling concerns
    - possible data leakage risks
    - other meaningful data-quality concerns
- Be concise.

Return ONLY valid JSON matching this structure:
{{
  "findings": [
    {{
      "issue_type": "...",
      "severity": "low | medium | high",
      "column": "... or null",
      "evidence": "...",
      "recommendation": "..."
    }}
  ],
  "summary": "..."
}}
""".strip()

    return prompt


def build_eda_prompt(profile: DatasetProfile) -> str:
    """Build a prompt that asks an LLM to perform exploratory data analysis.

    The prompt includes the full structured profile and instructs the model
    to return JSON with EDA findings and a summary.
    """
    profile_json = profile.model_dump_json(indent=2)

    prompt = f"""
You are an expert exploratory data analyst.

Analyze the dataset profile below and return a JSON object with:
- "findings": a list of analytical observations, each with:
    - "finding_type": short machine-readable type (e.g. "distribution",
      "relationship", "imbalance", "skewness", "predictive_signal")
    - "importance": "low", "medium", or "high"
    - "columns": list of column names involved or related to this finding
    - "evidence": what in the profile supports this observation
    - "interpretation": what this means in plain language
    - "modeling_implication": suggested impact or next step for modeling, or null
- "summary": a one-line overall assessment of the most important patterns

Dataset profile (JSON):
{profile_json}

Instructions:
- Base your findings ONLY on the supplied profile. Do not invent statistics.
- Distinguish evidence from interpretation and modeling implications.
- Look for:
    - target imbalance or class distribution concerns
    - dominant categories or heavily concentrated features
    - unusual numeric distributions, skewness, or extreme values
    - potentially important relationships between features or with the target
    - suspicious feature behavior or data quirks
    - possible predictive signals worth modeling
    - potential modeling implications for downstream work
- Be concise.

Return ONLY valid JSON matching this structure:
{{
  "findings": [
    {{
      "finding_type": "...",
      "importance": "low | medium | high",
      "columns": ["..."],
      "evidence": "...",
      "interpretation": "...",
      "modeling_implication": "..." or null
    }}
  ],
  "summary": "..."
}}
""".strip()

    return prompt
