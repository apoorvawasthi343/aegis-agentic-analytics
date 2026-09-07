"""AEGIS CLI - command-line interface for the AEGIS framework.

Provides python -m aegis to run the full pipeline from the terminal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.aegis.critic_agent import CriticAgent
from src.aegis.eda_agent import EDAAgent
from src.aegis.data_quality_agent import DataQualityAgent
from src.aegis.feature_engineering_agent import FeatureEngineeringAgent
from src.aegis.feature_engineering_executor import FeatureEngineeringExecutor
from src.aegis.loader import load_csv
from src.aegis.model_comparison import ModelComparison
from src.aegis.ollama_client import OllamaClient
from src.aegis.orchestrator import AEGISOrchestrator
from src.aegis.profiler import profile_dataset
from src.aegis.schemas import OrchestrationResult


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m aegis",
        description="AEGIS — Agentic AI framework for automated EDA and feature engineering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python -m aegis --data data/raw/sample_customers.csv --target churn",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the CSV file to analyze.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of the target column for modeling.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:1.7b",
        help="Ollama model to use for LLM-based reasoning (default: qwen3:1.7b).",
    )
    return parser


def run_cli(args: argparse.Namespace) -> int:
    """Run the AEGIS pipeline from parsed CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    csv_path = Path(args.data)

    # Validate CSV exists
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        print(f"Use --help for usage information.")
        return 1

    print(f"AEGIS Pipeline")
    print(f"===============")
    print(f"Data:    {csv_path}")
    print(f"Target:  {args.target}")
    print(f"Model:   {args.model}")
    print()

    # Create components
    llm_client = OllamaClient(model=args.model)
    data_quality_agent = DataQualityAgent(llm_client=llm_client)
    eda_agent = EDAAgent(llm_client=llm_client)
    feature_engineering_agent = FeatureEngineeringAgent(llm_client=llm_client)
    feature_executor = FeatureEngineeringExecutor()
    model_comparison = ModelComparison(random_state=42)
    critic_agent = CriticAgent()

    orchestrator = AEGISOrchestrator(
        data_quality_agent=data_quality_agent,
        eda_agent=eda_agent,
        feature_executor=feature_executor,
        model_comparison=model_comparison,
        critic_agent=critic_agent,
        feature_planner=feature_engineering_agent.recommend,
        loader=load_csv,
        profiler=profile_dataset,
    )

    try:
        result: OrchestrationResult = orchestrator.run(
            str(csv_path),
            args.target,
        )
    except Exception as e:
        print(f"ERROR: Pipeline failed: {type(e).__name__}: {e}")
        return 1

    # Print concise summary
    print(f"Dataset: {result.dataset_profile.row_count} rows × {result.dataset_profile.column_count} columns")
    print(f"Data quality: {len(result.data_quality_report.findings)} finding(s)")
    print(f"EDA: {len(result.eda_report.findings)} finding(s)")
    print(f"Features: {len(result.created_features)} created, {len(result.skipped_features)} skipped")
    print()
    print(f"Baseline: accuracy={result.modeling_comparison.baseline_metrics.accuracy:.4f}, "
          f"f1={result.modeling_comparison.baseline_metrics.f1_score:.4f}, "
          f"roc_auc={result.modeling_comparison.baseline_metrics.roc_auc:.4f}")
    print(f"Engineered: accuracy={result.modeling_comparison.engineered_metrics.accuracy:.4f}, "
          f"f1={result.modeling_comparison.engineered_metrics.f1_score:.4f}, "
          f"roc_auc={result.modeling_comparison.engineered_metrics.roc_auc:.4f}")
    print()
    print(f"Critic decision: {result.critic_report.decision.upper()}")
    if result.critic_report.reasons:
        for reason in result.critic_report.reasons:
            print(f"  • {reason}")
    print()
    print(result.summary)

    return 0


def main() -> None:
    """Entry point for python -m aegis."""
    parser = build_parser()
    args = parser.parse_args()
    exit_code = run_cli(args)
    sys.exit(exit_code)
