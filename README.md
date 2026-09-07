# AEGIS
## Agentic Exploratory & Generative Intelligence System

**Autonomous Multi-Agent Orchestration for Exploratory Data Analysis and Feature Engineering in Complex Datasets**

---

## Project Overview

AEGIS is a local-first multi-agent analytics framework that automates the early stages of data science workflows — from raw CSV to validated feature engineering — without requiring paid AI APIs.

The framework chains together specialist agents, each responsible for one stage of the pipeline:

- **Dataset Profiler** — inspects shape, types, missingness, cardinality
- **Data Quality Agent** — combines deterministic data-quality checks with optional local LLM reasoning; flags issues like missing values, duplicate rows, high-cardinality / possible identifier columns, and constant columns
- **EDA Agent** — combines deterministic EDA rules with optional local LLM reasoning; surfaces target imbalance, dominant categorical values, and other patterns
- **Feature Engineering Agent** — proposes transformations using deterministic rules and optional LLM reasoning
- **Feature Engineering Executor** — safely applies validated transformations; rejects invalid specs
- **Modeling Comparison** — trains baseline and engineered models, compares metrics
- **Critic Agent** — applies deterministic rules to decide whether engineered features are accepted, rejected, or flagged for review

AEGIS separates deterministic computation from LLM reasoning. The Profiler, Data Quality Agent, EDA Agent, and Executor all implement deterministic rules. The Data Quality Agent, EDA Agent, and Feature Engineering Agent each accept an optional LLMClient for LLM-based reasoning — but every LLM response is validated through a Pydantic response schema before use. The FeatureEngineeringExecutor applies only pre-approved transformation types and validates each spec at application time: supported transformation type, referenced columns exist, data type compatibility, and other per-transformation constraints. Rejected specs become `SkippedFeature` entries with reasons. No arbitrary LLM-generated Python is ever executed.

---

## Architecture

```
Dataset (CSV)
     |
     v
┌──────────────┐
│  Profiler    │  pandas / NumPy — shape, dtypes, missingness, cardinality
└──────┬───────┘
       │ DatasetProfile
       v
┌──────────────┐
│ Data Quality │  flags high-cardinality, type issues, duplicates, constant columns
│    Agent     │
└──────┬───────┘
       │ DataQualityReport
       v
┌──────────────┐
│     EDA      │  distribution patterns, imbalance, notable characteristics
│    Agent     │
└──────┬───────┘
       │ EDAReport
       v
┌────────────────────────────┐
│ Feature Engineering Agent  │  deterministic rules + optional LLM reasoning
│                            │  → structured Pydantic FeatureEngineeringSuggestions
└──────┬─────────────────────┘
       │ FeatureEngineeringReport (specs + applied + skipped)
       v
┌─────────────────────┐
│ Feature Engineering│  validates each spec at application time: supported transformation
│     Executor        │  types, referenced columns exist, data type compatibility, and
│                    │  other per-transformation constraints. Invalid specs are skipped
│                    │  with recorded reasons.
└──────┬──────────────┘
       │ transformed DataFrame
       v
┌─────────────────────┐
│  Modeling Comparison│  baseline LogisticRegression → engineered LogisticRegression
│                     │  accuracy, F1, ROC-AUC compared
└──────┬──────────────┘
       │ BaselineVsEngineeredComparison
       v
┌──────────────┐
│    Critic    │  deterministic rules: all metrics up → ACCEPT
│    Agent     │                      none up → REJECT
└──────┬───────┘                      mixed → REVIEW
       │
       v
┌─────────────────────┐
│   Final AEGIS Result│  OrchestrationResult with all stage reports + critic decision
└─────────────────────┘
```

**LLM Integration (optional):**

```
LLMClient (abstract interface)
  └── OllamaClient
        └── qwen3:1.7b (local, zero-cost)

The DataQualityAgent, EDAAgent, and FeatureEngineeringAgent each accept an
optional LLMClient and may request LLM-based reasoning when one is provided.
All LLM responses are validated through Pydantic response schemas before use.
The orchestrator itself does NOT call the LLM directly — LLM enrichment
happens inside the individual agents.
```

---

## Key Design Principles

| Principle | Description |
|-----------|-------------|
| Python computes, LLM reasons | All data transformations, profiling, validation, and metric computation use deterministic pandas/NumPy/scikit-learn. The LLM only proposes feature recommendations. |
| Structured Pydantic outputs | Every agent produces validated Pydantic models (DatasetProfile, DataQualityReport, EDAReport, FeatureEngineeringReport, OrchestrationResult). |
| Provider-independent LLM abstraction | `LLMClient` defines the interface; `OllamaClient` implements it. Other providers can be added without changing the pipeline. |
| Local Ollama inference | Default configuration uses Ollama + qwen3:1.7b locally. No paid API keys required. |
- **No arbitrary LLM-generated Python execution** — LLM suggestions must match a Pydantic schema and pass executor validation. The FeatureEngineeringExecutor checks supported transformation types, referenced column existence, data type compatibility, and other application-time constraints. Rejected specs become `SkippedFeature` entries with reasons.
| Deterministic fallback behavior | When no LLM is available or the LLM returns no valid specs, the FeatureEngineeringAgent still produces deterministic rule-based recommendations. |
| Modular specialist agents | Each agent is a single-responsibility class, injectable for testing, with no cross-dependencies beyond its inputs/outputs. |
| Centralized orchestration | `AEGISOrchestrator` / `Orchestrator` sequences the pipeline stages and assembles the final result. |

---

## Technology Stack

| Technology | Role |
|------------|------|
| Python >=3.12 | Core language |
| pandas | Data loading, profiling, transformation |
| NumPy | Numeric operations |
| Pydantic | Structured output validation for all agent reports |
| scikit-learn | LogisticRegression baseline and engineered model training, metric computation |
| pytest | Test suite (146 tests) |
| Ollama | Local LLM inference host |
| qwen3:1.7b | Default local model for optional LLM-based feature recommendations |

---

## Installation

**Clone and set up:**

```bash
git clone https://github.com/apoorv-awasthi/aegis-agentic-analytics.git
cd aegis-agentic-analytics
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell
```

**Install dependencies:**

```bash
pip install -e ".[dev]"
```

This installs pandas, NumPy, Pydantic, ollama, scikit-learn, and pytest (dev).

**Pull the local LLM model (optional, for LLM-based feature recommendations):**

```bash
ollama pull qwen3:1.7b
```

Without Ollama running, the pipeline still works — the FeatureEngineeringAgent falls back to deterministic rule-based recommendations.

---

## Usage

**Command-line interface:**

```bash
python -m aegis --data <csv_path> --target <target_column> [--model qwen3:1.7b]
```

- `--data` — path to a CSV file
- `--target` — name of the target column for classification
- `--model` — (optional) Ollama model name; defaults to `qwen3:1.7b` if omitted

**Example:**

```bash
python -m aegis --data data/raw/telco_customer_churn.csv --target Churn --model qwen3:1.7b
```

This runs the full production pipeline: Profiler → Data Quality Agent → EDA Agent → Feature Engineering Agent → Feature Engineering Executor → baseline LogisticRegression → engineered LogisticRegression → Critic Agent → final OrchestrationResult.

Output is printed to the console as a structured summary and can optionally be exported.

---

## Real-World Evaluation

AEGIS was evaluated end-to-end against the **IBM Telco Customer Churn** dataset (7,043 rows, 21 columns, target: `Churn`).

The full evaluation document is at [`docs/evaluations/telco_churn_evaluation.md`](docs/evaluations/telco_churn_evaluation.md).

**Modeling Results:**

| Metric | Baseline | Engineered | Change |
|--------|----------|------------|--------|
| Accuracy | 0.7963 | 0.8027 | +0.0064 |
| F1 | 0.7913 | 0.7948 | +0.0035 |
| ROC-AUC | 0.8403 | 0.8485 | +0.0082 |

**Critic Decision: ACCEPT**

All three evaluated metrics improved. Under AEGIS's critic policy — all metrics up → ACCEPT, none up → REJECT, mixed → REVIEW — this result is correctly classified as ACCEPT. The absolute gains are modest (accuracy +0.64%, F1 +0.35%, ROC-AUC +0.82%) but directionally positive, and three features were created with zero skipped.

**Environment:** Local Ollama + qwen3:1.7b, zero paid API usage.

---

## Testing

**Full test suite: 146 tests passing.**

```bash
pytest
```

Tests cover data quality agents, EDA agents, feature engineering (deterministic and LLM-mocked), modeling, critic rules, orchestration, and CLI integration. No real Ollama requests are made in tests — the LLM client is mocked.

---

## Repository Structure

```
aegis-agentic-analytics/
├── data/
│   └── raw/
│       └── telco_customer_churn.csv          # evaluation dataset (ignored in git)
├── docs/
│   └── evaluations/
│       └── telco_churn_evaluation.md         # full Telco evaluation report
├── src/
│   └── aegis/
│       ├── __init__.py
│       ├── __main__.py                        # CLI entry point (python -m aegis)
│       ├── main.py                            # CLI argument parsing and pipeline launch
│       ├── agents/
│       │   ├── __init__.py
│       │   └── feature_engineering_agent.py  # deterministic + LLM feature recommendations
│       ├── critic_agent.py                    # deterministic critic decision rules
│       ├── data_quality_agent.py              # data quality analysis
│       ├── eda_agent.py                       # exploratory data analysis
│       ├── exporter.py                        # result export utilities
│       ├── feature_engineering_executor.py    # safe feature application + validation
│       ├── llm.py                             # LLMClient abstract interface
│       ├── loader.py                          # CSV dataset loading
│       ├── model_comparison.py                # baseline vs engineered comparison
│       ├── modeling_agent.py                  # model training and metric computation
│       ├── ollama_client.py                   # Ollama LLM client implementation
│       ├── orchestration.py                   # AEGISOrchestrator pipeline
│       ├── orchestrator.py                    # Orchestrator pipeline
│       ├── profiler.py                        # dataset profiling
│       ├── prompts.py                         # LLM prompt templates
│       └── schemas.py                         # all Pydantic report schemas
├── tests/
│   ├── test_cli.py
│   ├── test_critic_agent.py
│   ├── test_data_quality.py
│   ├── test_eda.py
│   ├── test_feature_engineering.py
│   ├── test_feature_selection.py
│   ├── test_modeling.py
│   ├── test_orchestrator.py
│   └── test_profiler.py
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Example Workflow

1. **Load** — CSV is loaded via `loader.py` into a pandas DataFrame.
2. **Profile** — `profiler.py` produces a `DatasetProfile`: shape, column names, dtypes, missingness, cardinality.
3. **Agent reasoning** — the Data Quality Agent and EDA Agent analyze the profile and dataset, producing `DataQualityReport` and `EDAReport` with flagged findings.
4. **Feature recommendations** — the FeatureEngineeringAgent combines deterministic rules (log1p, ratio, count_sum, missing_indicator, product) with optional LLM suggestions. Output is a `FeatureEngineeringReport` containing `specs`, `applied_features`, and `skipped_features`.
5. **Safe transformations** — the FeatureEngineeringExecutor validates each spec (column existence, transformation type, no duplicate columns) and applies valid ones. Invalid specs become `SkippedFeature` entries with reasons.
6. **Modeling** — baseline and engineered LogisticRegression models are trained; accuracy, F1, and ROC-AUC are computed and compared.
7. **Critic decision** — the CriticAgent applies deterministic rules to the metric changes and returns ACCEPT, REJECT, or REVIEW.

---

## Research / Evaluation Goal

AEGIS investigates whether coordinated specialist agents can accelerate early-stage analytics and feature engineering while remaining measurable and auditable. The framework is designed so that:

- Every stage produces structured, inspectable output (Pydantic models).
- LLM involvement is optional and bounded — it proposes, never executes.
- Results are reproducible: deterministic rules produce the same output given the same input, and LLM suggestions are validated before use.
- Success is measured by concrete metrics (accuracy, F1, ROC-AUC) and a deterministic critic decision, not by subjective quality judgments.

---

## Zero-Cost Local AI

The default configuration uses **Ollama + qwen3:1.7b** running locally. No API keys, no paid usage, no cloud dependencies. The pipeline works fully offline once Ollama is installed and the model is pulled.

```bash
ollama pull qwen3:1.7b
python -m aegis --data data/raw/telco_customer_churn.csv --target Churn --model qwen3:1.7b
```

If Ollama is not available, the FeatureEngineeringAgent still operates using deterministic rules only.

---

## Limitations / Future Work

- **Additional datasets** — broader evaluation across regression, multi-class, and time-series datasets.
- **Regression support** — current modeling uses LogisticRegression (classification). Regression metrics and models can be added.
- **Additional model families** — extending the modeling comparison beyond LogisticRegression.
- **Richer feature transformations** — more transformation types, interaction features, and automated feature selection.
- **UI/dashboard** — a visual interface for inspecting pipeline results and agent findings.
- **Larger local models** — testing with larger locally-run models on stronger hardware for richer LLM-based feature recommendations.

---

## License / Author

**Author:** Apoorv Awasthi

No license is specified in this repository. All rights reserved by the author.

---

*End of README.*
