"""Tests for the AEGIS CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.aegis.main import build_parser, run_cli
from src.aegis.schemas import (
    BaselineVsEngineeredComparison,
    CriticReport,
    DatasetProfile,
    DataQualityReport,
    EDAReport,
    FeatureEngineeringReport,
    ModelMetrics,
)


class _TestResult:
    """Concrete test result that the CLI can format without MagicMock issues."""

    def __init__(self) -> None:
        self.dataset_profile = DatasetProfile(
            row_count=100,
            column_count=5,
            duplicate_row_count=0,
            missing_values_by_column={},
            unique_values_by_column={},
            data_types_by_column={},
            numeric_statistics={},
            categorical_statistics={},
            target_column="churn",
            target_distribution={},
        )
        self.data_quality_report = DataQualityReport(findings=[], summary="No issues")
        self.eda_report = EDAReport(findings=[], summary="No findings")
        self.feature_engineering_report = FeatureEngineeringReport(
            original_row_count=100,
            original_column_count=5,
            engineered_row_count=100,
            engineered_column_count=6,
            applied_features=[],
            skipped_features=[],
            summary="No features",
        )
        self.created_features: list[str] = []
        self.skipped_features: list[str] = []

        baseline = ModelMetrics(accuracy=0.85, f1_score=0.84, roc_auc=0.90,
                                train_rows=80, test_rows=20)
        engineered = ModelMetrics(accuracy=0.87, f1_score=0.86, roc_auc=0.92,
                                  train_rows=80, test_rows=20)
        self.modeling_comparison = BaselineVsEngineeredComparison(
            baseline_metrics=baseline,
            engineered_metrics=engineered,
            features_created=["log_feature"],
            features_skipped=[],
            accuracy_change=0.02,
            f1_change=0.02,
            roc_auc_change=0.02,
            improved=True,
            summary="Improvement",
        )
        self.critic_report = CriticReport(
            decision="accept",
            accepted_features=["log_feature"],
            rejected_features=[],
            reasons=["Performance improved."],
            performance_improved=True,
            leakage_warning=False,
            summary="Accept",
        )
        self.summary = "Pipeline completed successfully."


def _mock_result() -> _TestResult:
    """Return a test result for CLI tests."""
    return _TestResult()


class TestCLIParser:
    """Tests for CLI argument parsing."""

    def test_required_args(self) -> None:
        """Required --data and --target args parse correctly."""
        parser = build_parser()
        args = parser.parse_args(["--data", "data.csv", "--target", "churn"])
        assert args.data == "data.csv"
        assert args.target == "churn"
        assert args.model == "qwen3:1.7b"

    def test_optional_model_arg(self) -> None:
        """Optional --model arg overrides default."""
        parser = build_parser()
        args = parser.parse_args([
            "--data", "data.csv",
            "--target", "churn",
            "--model", "llama3",
        ])
        assert args.data == "data.csv"
        assert args.target == "churn"
        assert args.model == "llama3"

    def test_no_args_shows_usage(self) -> None:
        """Running with no args should exit with error."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestCLIExecution:
    """Tests for CLI execution logic."""

    def test_csv_not_found(self, tmp_path: Path) -> None:
        """Missing CSV path prints error and returns 1."""
        nonexistent = tmp_path / "nonexistent.csv"
        assert not nonexistent.exists()
        result = run_cli(
            build_parser().parse_args(["--data", str(nonexistent), "--target", "y"])
        )
        assert result == 1

    def test_successful_run(self, tmp_path: Path) -> None:
        """Successful run prints summary and returns 0."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("age,score,churn\n35,70,0\n42,85,1\n", encoding="utf-8")

        with patch("src.aegis.main.load_csv", return_value=None), \
             patch("src.aegis.main.OllamaClient"), \
             patch("src.aegis.main.DataQualityAgent"), \
             patch("src.aegis.main.EDAAgent"), \
             patch("src.aegis.main.FeatureEngineeringAgent"), \
             patch("src.aegis.main.FeatureEngineeringExecutor"), \
             patch("src.aegis.main.ModelComparison"), \
             patch("src.aegis.main.CriticAgent"), \
             patch("src.aegis.main.profile_dataset",
                   return_value=DatasetProfile(
                       row_count=100, column_count=5, duplicate_row_count=0,
                       missing_values_by_column={}, unique_values_by_column={},
                       data_types_by_column={}, numeric_statistics={},
                       categorical_statistics={},
                       target_column="churn", target_distribution={},
                   )), \
             patch("src.aegis.main.AEGISOrchestrator") as mock_orch:
            mock_orch.return_value.run.return_value = _mock_result()
            exit_code = run_cli(
                build_parser().parse_args(
                    ["--data", str(csv_path), "--target", "churn"]
                )
            )
        assert exit_code == 0

    def test_orchestrator_called_with_args(self, tmp_path: Path) -> None:
        """Orchestrator.run receives correct arguments."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y\n1,0\n2,1\n", encoding="utf-8")

        with patch("src.aegis.main.load_csv", return_value=None), \
             patch("src.aegis.main.OllamaClient"), \
             patch("src.aegis.main.DataQualityAgent"), \
             patch("src.aegis.main.EDAAgent"), \
             patch("src.aegis.main.FeatureEngineeringAgent"), \
             patch("src.aegis.main.FeatureEngineeringExecutor"), \
             patch("src.aegis.main.ModelComparison"), \
             patch("src.aegis.main.CriticAgent"), \
             patch("src.aegis.main.profile_dataset",
                   return_value=DatasetProfile(
                       row_count=2, column_count=2, duplicate_row_count=0,
                       missing_values_by_column={}, unique_values_by_column={},
                       data_types_by_column={}, numeric_statistics={},
                       categorical_statistics={},
                       target_column="y", target_distribution={},
                   )), \
             patch("src.aegis.main.AEGISOrchestrator") as mock_orch:
            mock_orch.return_value.run.return_value = _mock_result()
            run_cli(
                build_parser().parse_args(
                    ["--data", str(csv_path), "--target", "churn"]
                )
            )
            mock_orch.return_value.run.assert_called_once_with(
                str(csv_path), "churn"
            )

    def test_pipeline_failure_returns_1(self, tmp_path: Path) -> None:
        """Pipeline exception returns exit code 1."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("age,score,churn\n35,70,0\n", encoding="utf-8")

        with patch("src.aegis.main.load_csv", return_value=None), \
             patch("src.aegis.main.OllamaClient"), \
             patch("src.aegis.main.DataQualityAgent"), \
             patch("src.aegis.main.EDAAgent"), \
             patch("src.aegis.main.FeatureEngineeringAgent"), \
             patch("src.aegis.main.FeatureEngineeringExecutor"), \
             patch("src.aegis.main.ModelComparison"), \
             patch("src.aegis.main.CriticAgent"), \
             patch("src.aegis.main.profile_dataset",
                   return_value=DatasetProfile(
                       row_count=1, column_count=2, duplicate_row_count=0,
                       missing_values_by_column={}, unique_values_by_column={},
                       data_types_by_column={}, numeric_statistics={},
                       categorical_statistics={},
                       target_column="y", target_distribution={},
                   )), \
             patch("src.aegis.main.AEGISOrchestrator") as mock_orch:
            mock_orch.return_value.run.side_effect = RuntimeError("boom")
            exit_code = run_cli(
                build_parser().parse_args(
                    ["--data", str(csv_path), "--target", "churn"]
                )
            )
        assert exit_code == 1


class TestCLIIntegration:
    """Integration-style tests for the CLI entry point."""

    def test_main_function_calls_run_cli(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """main() parses args and calls run_cli."""
        from src.aegis.main import main

        csv_path = tmp_path / "test.csv"
        csv_path.write_text("age,score,churn\n35,70,0\n", encoding="utf-8")

        with patch("src.aegis.main.load_csv", return_value=None), \
             patch("src.aegis.main.OllamaClient"), \
             patch("src.aegis.main.DataQualityAgent"), \
             patch("src.aegis.main.EDAAgent"), \
             patch("src.aegis.main.FeatureEngineeringAgent"), \
             patch("src.aegis.main.FeatureEngineeringExecutor"), \
             patch("src.aegis.main.ModelComparison"), \
             patch("src.aegis.main.CriticAgent"), \
             patch("src.aegis.main.profile_dataset",
                   return_value=DatasetProfile(
                       row_count=1, column_count=2, duplicate_row_count=0,
                       missing_values_by_column={}, unique_values_by_column={},
                       data_types_by_column={}, numeric_statistics={},
                       categorical_statistics={},
                       target_column="y", target_distribution={},
                   )), \
             patch("src.aegis.main.AEGISOrchestrator") as mock_orch:
            mock_orch.return_value.run.return_value = _mock_result()

            import sys
            old_argv = sys.argv
            try:
                sys.argv = [
                    "python -m aegis",
                    "--data", str(csv_path),
                    "--target", "churn",
                ]
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0
            finally:
                sys.argv = old_argv

        mock_orch.assert_called_once()
        captured = capsys.readouterr()
        assert "AEGIS Pipeline" in captured.out
