"""Tests for the AEGIS Orchestrator."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.aegis.critic_agent import CriticAgent
from src.aegis.data_quality_agent import DataQualityAgent
from src.aegis.eda_agent import EDAAgent
from src.aegis.feature_engineering_executor import FeatureEngineeringExecutor
from src.aegis.model_comparison import ModelComparison
from src.aegis.orchestrator import AEGISOrchestrator, _default_feature_planner
from src.aegis.schemas import (
    AppliedFeature,
    BaselineVsEngineeredComparison,
    CriticReport,
    DataQualityReport,
    DatasetProfile,
    EDAReport,
    FeatureEngineeringReport,
    FeatureEngineeringSpec,
    ModelMetrics,
    OrchestrationResult,
    SkippedFeature,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_profile(target_column: str = "target") -> DatasetProfile:
    """Create a minimal DatasetProfile for testing."""
    return DatasetProfile(
        row_count=100,
        column_count=3,
        duplicate_row_count=0,
        missing_values_by_column={"num_1": 0, "num_2": 0, target_column: 0},
        unique_values_by_column={"num_1": 50, "num_2": 30, target_column: 2},
        data_types_by_column={"num_1": "float64", "num_2": "float64", target_column: "int64"},
        numeric_statistics={
            "num_1": {
                "mean": 0.5,
                "median": 0.5,
                "std": 0.2,
                "min": 0.0,
                "max": 1.0,
            },
            "num_2": {
                "mean": 0.3,
                "median": 0.3,
                "std": 0.15,
                "min": 0.0,
                "max": 1.0,
            },
        },
        categorical_statistics={},
        target_column=target_column,
        target_distribution={"class_0": 60, "class_1": 40},
    )


def _make_minimal_df() -> pd.DataFrame:
    """Create a minimal DataFrame for testing."""
    return pd.DataFrame(
        {
            "num_1": [0.1 * i for i in range(100)],
            "num_2": [0.05 * i for i in range(100)],
            "target": [0 if i < 60 else 1 for i in range(100)],
        }
    )


def _make_mock_data_quality_agent(
    findings: list | None = None,
) -> MagicMock:
    """Create a mock DataQualityAgent that returns a DataQualityReport."""
    agent = MagicMock(spec=DataQualityAgent)
    agent.analyze.return_value = DataQualityReport(
        findings=findings or [],
        summary="Mock data quality report",
    )
    return agent


def _make_mock_eda_agent(
    findings: list | None = None,
) -> MagicMock:
    """Create a mock EDAAgent that returns an EDAReport."""
    agent = MagicMock(spec=EDAAgent)
    agent.analyze.return_value = EDAReport(
        findings=findings or [],
        summary="Mock EDA report",
    )
    return agent


def _make_mock_model_comparison(
    comparison: BaselineVsEngineeredComparison | None = None,
) -> MagicMock:
    """Create a mock ModelComparison that returns a BaselineVsEngineeredComparison."""
    agent = MagicMock(spec=ModelComparison)
    if comparison is None:
        baseline_metrics = ModelMetrics(
            accuracy=0.85,
            f1_score=0.8496,
            roc_auc=0.93,
            train_rows=80,
            test_rows=20,
        )
        engineered_metrics = ModelMetrics(
            accuracy=0.87,
            f1_score=0.8696,
            roc_auc=0.94,
            train_rows=80,
            test_rows=20,
        )
        comparison = BaselineVsEngineeredComparison(
            baseline_metrics=baseline_metrics,
            engineered_metrics=engineered_metrics,
            features_created=["log_num_1"],
            features_skipped=[],
            accuracy_change=0.02,
            f1_change=0.02,
            roc_auc_change=0.01,
            improved=True,
            summary="Mock comparison: engineered features improved performance.",
        )
    agent.compare.return_value = comparison
    return agent


def _make_mock_profiler() -> MagicMock:
    """Create a mock profiler that returns a DatasetProfile."""
    profiler = MagicMock()
    profiler.return_value = _make_minimal_profile()
    return profiler


def _make_mock_feature_planner() -> MagicMock:
    """Create a mock feature planner that returns feature specs."""
    planner = MagicMock()
    planner.return_value = [
        FeatureEngineeringSpec(
            feature_name="log_num_1",
            transformation_type="log1p",
            columns=["num_1"],
        ),
    ]
    return planner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_orchestrator_constructor_accepts_components() -> None:
    """Constructor should accept all required components."""
    orchestrator = AEGISOrchestrator(
        data_quality_agent=MagicMock(),
        eda_agent=MagicMock(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=MagicMock(),
        critic_agent=CriticAgent(),
    )
    assert orchestrator is not None


def test_orchestrator_run_loads_csv() -> None:
    """Orchestrator should load the CSV using the loader."""
    df = _make_minimal_df()
    loader = MagicMock(return_value=df)
    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=_make_mock_model_comparison(),
        critic_agent=CriticAgent(),
        loader=loader,
        profiler=_make_mock_profiler(),
    )

    result = orchestrator.run("dummy.csv", "target")

    loader.assert_called_once_with("dummy.csv")
    assert isinstance(result, OrchestrationResult)


def test_orchestrator_run_profiles_dataset() -> None:
    """Orchestrator should profile the dataset."""
    df = _make_minimal_df()
    profiler = MagicMock(return_value=_make_minimal_profile())
    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=_make_mock_model_comparison(),
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=df),
        profiler=profiler,
    )

    result = orchestrator.run("dummy.csv", "target")

    profiler.assert_called_once()
    call_args = profiler.call_args
    assert call_args[0][0] is df
    assert call_args[1]["target_column"] == "target"


def test_orchestrator_run_runs_data_quality_agent() -> None:
    """Orchestrator should run the data quality agent on the profile."""
    profile = _make_minimal_profile()
    data_quality_agent = _make_mock_data_quality_agent()
    orchestrator = AEGISOrchestrator(
        data_quality_agent=data_quality_agent,
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=_make_mock_model_comparison(),
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=_make_minimal_df()),
        profiler=MagicMock(return_value=profile),
    )

    result = orchestrator.run("dummy.csv", "target")

    data_quality_agent.analyze.assert_called_once_with(profile)
    assert result.data_quality_report is not None


def test_orchestrator_run_runs_eda_agent() -> None:
    """Orchestrator should run the EDA agent on the profile."""
    profile = _make_minimal_profile()
    eda_agent = _make_mock_eda_agent()
    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=eda_agent,
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=_make_mock_model_comparison(),
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=_make_minimal_df()),
        profiler=MagicMock(return_value=profile),
    )

    result = orchestrator.run("dummy.csv", "target")

    eda_agent.analyze.assert_called_once_with(profile)
    assert result.eda_report is not None


def test_orchestrator_run_runs_model_comparison() -> None:
    """Orchestrator should run the model comparison with the DataFrame and target."""
    df = _make_minimal_df()
    model_comparison = _make_mock_model_comparison()
    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=model_comparison,
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=df),
        profiler=MagicMock(return_value=_make_minimal_profile()),
        feature_planner=_make_mock_feature_planner(),
    )

    result = orchestrator.run("dummy.csv", "target")

    model_comparison.compare.assert_called_once()
    call_args = model_comparison.compare.call_args
    assert call_args[1]["target_column"] == "target"
    assert isinstance(call_args[1]["df"], pd.DataFrame)


def test_orchestrator_passes_feature_specs_to_comparison() -> None:
    """Orchestrator should pass feature specs from planner to comparison."""
    df = _make_minimal_df()
    feature_specs = [
        FeatureEngineeringSpec(
            feature_name="log_num_1",
            transformation_type="log1p",
            columns=["num_1"],
        ),
    ]
    feature_planner = MagicMock(return_value=feature_specs)
    model_comparison = _make_mock_model_comparison()
    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=model_comparison,
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=df),
        profiler=MagicMock(return_value=_make_minimal_profile()),
        feature_planner=feature_planner,
    )

    result = orchestrator.run("dummy.csv", "target")

    feature_planner.assert_called_once_with(df, "target")
    model_comparison.compare.assert_called_once()
    call_args = model_comparison.compare.call_args
    assert call_args[1]["feature_specs"] == feature_specs


def test_orchestrator_passes_target_column_to_profiler() -> None:
    """Orchestrator should pass target_column to the profiler."""
    profile = _make_minimal_profile(target_column="churn")
    profiler = MagicMock(return_value=profile)
    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=_make_mock_model_comparison(),
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=_make_minimal_df()),
        profiler=profiler,
    )

    result = orchestrator.run("dummy.csv", "churn")

    profiler.assert_called_once()
    call_args = profiler.call_args
    assert call_args[1]["target_column"] == "churn"
    assert result.dataset_profile.target_column == "churn"


def test_orchestrator_result_contains_all_stages() -> None:
    """OrchestrationResult should contain outputs from all pipeline stages."""
    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(
            findings=[
                {
                    "issue_type": "test",
                    "severity": "low",
                    "evidence": "test",
                    "recommendation": "test",
                }
            ]
        ),
        eda_agent=_make_mock_eda_agent(
            findings=[
                {
                    "finding_type": "test",
                    "importance": "low",
                    "columns": [],
                    "evidence": "test",
                    "interpretation": "test",
                }
            ]
        ),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=_make_mock_model_comparison(),
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=_make_minimal_df()),
        profiler=MagicMock(return_value=_make_minimal_profile()),
        feature_planner=_make_mock_feature_planner(),
    )

    result = orchestrator.run("dummy.csv", "target")

    assert result.dataset_profile is not None
    assert result.data_quality_report is not None
    assert result.eda_report is not None
    assert result.feature_engineering_report is not None
    assert result.modeling_comparison is not None
    assert result.critic_report is not None
    assert isinstance(result.created_features, list)
    assert isinstance(result.skipped_features, list)


def test_orchestrator_result_created_features_from_comparison() -> None:
    """created_features should match the comparison's features_created."""
    comparison = BaselineVsEngineeredComparison(
        baseline_metrics=ModelMetrics(accuracy=0.8, f1_score=0.79, roc_auc=None, train_rows=80, test_rows=20),
        engineered_metrics=ModelMetrics(accuracy=0.82, f1_score=0.81, roc_auc=None, train_rows=80, test_rows=20),
        features_created=["log_num_1", "num_1_is_missing"],
        features_skipped=[],
        accuracy_change=0.02,
        f1_change=0.02,
        roc_auc_change=None,
        improved=True,
        summary="Mock comparison",
    )
    model_comparison = MagicMock(spec=ModelComparison)
    model_comparison.compare.return_value = comparison

    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=model_comparison,
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=_make_minimal_df()),
        profiler=MagicMock(return_value=_make_minimal_profile()),
        feature_planner=_make_mock_feature_planner(),
    )

    result = orchestrator.run("dummy.csv", "target")

    assert result.created_features == ["log_num_1", "num_1_is_missing"]
    assert result.skipped_features == []


def test_orchestrator_result_skipped_features_from_comparison() -> None:
    """skipped_features should match the comparison's features_skipped."""
    comparison = BaselineVsEngineeredComparison(
        baseline_metrics=ModelMetrics(accuracy=0.8, f1_score=0.79, roc_auc=None, train_rows=80, test_rows=20),
        engineered_metrics=ModelMetrics(accuracy=0.8, f1_score=0.79, roc_auc=None, train_rows=80, test_rows=20),
        features_created=[],
        features_skipped=["bad_feature: column not found"],
        accuracy_change=0.0,
        f1_change=0.0,
        roc_auc_change=None,
        improved=False,
        summary="Mock comparison",
    )
    model_comparison = _make_mock_model_comparison(comparison=comparison)

    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=model_comparison,
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=_make_minimal_df()),
        profiler=MagicMock(return_value=_make_minimal_profile()),
        feature_planner=_make_mock_feature_planner(),
    )

    result = orchestrator.run("dummy.csv", "target")

    assert len(result.skipped_features) == 1
    assert "bad_feature: column not found" in result.skipped_features[0]


def test_orchestrator_critic_review_is_run() -> None:
    """Orchestrator should run the critic agent on the comparison."""
    comparison = BaselineVsEngineeredComparison(
        baseline_metrics=ModelMetrics(accuracy=0.8, f1_score=0.79, roc_auc=None, train_rows=80, test_rows=20),
        engineered_metrics=ModelMetrics(accuracy=0.82, f1_score=0.81, roc_auc=None, train_rows=80, test_rows=20),
        features_created=["log_num_1"],
        features_skipped=[],
        accuracy_change=0.02,
        f1_change=0.02,
        roc_auc_change=None,
        improved=True,
        summary="Mock comparison",
    )
    model_comparison = _make_mock_model_comparison(comparison=comparison)

    critic_agent = MagicMock(spec=CriticAgent)
    critic_agent.review.return_value = CriticReport(
        decision="accept",
        accepted_features=["log_num_1"],
        rejected_features=[],
        reasons=["Performance improved."],
        performance_improved=True,
        leakage_warning=False,
        summary="Critic accepted engineered features.",
    )

    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=model_comparison,
        critic_agent=critic_agent,
        loader=MagicMock(return_value=_make_minimal_df()),
        profiler=MagicMock(return_value=_make_minimal_profile()),
        feature_planner=_make_mock_feature_planner(),
    )

    result = orchestrator.run("dummy.csv", "target")

    critic_agent.review.assert_called_once_with(comparison)
    assert result.critic_report.decision == "accept"


def test_orchestrator_final_result_validates() -> None:
    """The OrchestrationResult should validate as a Pydantic model."""
    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=_make_mock_model_comparison(),
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=_make_minimal_df()),
        profiler=MagicMock(return_value=_make_minimal_profile()),
        feature_planner=_make_mock_feature_planner(),
    )

    result = orchestrator.run("dummy.csv", "target")

    # Should not raise
    result_json = result.model_dump_json()
    assert isinstance(result_json, str)
    assert len(result_json) > 0


def test_orchestrator_without_feature_planner_skips_engineering() -> None:
    """Without a feature planner, the orchestrator should still work."""
    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=_make_mock_model_comparison(),
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=_make_minimal_df()),
        profiler=MagicMock(return_value=_make_minimal_profile()),
        # No feature_planner provided
    )

    result = orchestrator.run("dummy.csv", "target")

    assert isinstance(result, OrchestrationResult)
    assert result.created_features is not None


def test_default_feature_planner_identifies_log_transformable_columns() -> None:
    """Default planner should suggest log1p for positive numeric columns."""
    df = pd.DataFrame(
        {
            "positive_col": [1.0, 2.0, 3.0, 4.0],
            "negative_col": [-1.0, -2.0, -3.0, -4.0],
            "target": [0, 0, 1, 1],
        }
    )

    specs = _default_feature_planner(df, "target")

    # Only positive_col should get a log transform
    log_specs = [s for s in specs if s.transformation_type == "log1p"]
    assert len(log_specs) == 1
    assert log_specs[0].feature_name == "log_positive_col"


def test_default_feature_planner_identifies_missing_indicator_columns() -> None:
    """Default planner should suggest missing indicators for columns with NaNs."""
    df = pd.DataFrame(
        {
            "col_with_na": [1.0, None, 3.0, 4.0],
            "col_without_na": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 0, 1, 1],
        }
    )

    specs = _default_feature_planner(df, "target")

    missing_specs = [s for s in specs if s.transformation_type == "missing_indicator"]
    assert len(missing_specs) == 1
    assert missing_specs[0].feature_name == "col_with_na_is_missing"


def test_orchestrator_no_planner_means_no_features_engineered() -> None:
    """Without a feature_planner, no features should be created.

    This verifies that the orchestrator does NOT auto-invent feature
    engineering decisions — it requires an explicit planner to be
    passed in.
    """
    model_comparison = MagicMock(spec=ModelComparison)
    comparison = BaselineVsEngineeredComparison(
        baseline_metrics=ModelMetrics(accuracy=0.8, f1_score=0.79, roc_auc=None, train_rows=80, test_rows=20),
        engineered_metrics=ModelMetrics(accuracy=0.8, f1_score=0.79, roc_auc=None, train_rows=80, test_rows=20),
        features_created=[],
        features_skipped=[],
        accuracy_change=0.0,
        f1_change=0.0,
        roc_auc_change=None,
        improved=False,
        summary="Mock comparison: no features engineered",
    )
    model_comparison.compare.return_value = comparison

    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=model_comparison,
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=_make_minimal_df()),
        profiler=MagicMock(return_value=_make_minimal_profile()),
        # No feature_planner — orchestrator should not invent features
    )

    result = orchestrator.run("dummy.csv", "target")

    # The comparison should have been called with empty feature specs
    model_comparison.compare.assert_called_once()
    call_args = model_comparison.compare.call_args
    assert call_args[1]["feature_specs"] == []

    # Result should show no features created
    assert result.created_features == []
    assert result.skipped_features == []


def test_orchestrator_default_planner_is_opt_in_not_default() -> None:
    """The _default_feature_planner must be explicitly passed to be used.

    This test verifies that constructing the orchestrator WITHOUT a
    feature_planner does NOT use _default_feature_planner automatically.
    """
    model_comparison = MagicMock(spec=ModelComparison)
    comparison = BaselineVsEngineeredComparison(
        baseline_metrics=ModelMetrics(accuracy=0.8, f1_score=0.79, roc_auc=None, train_rows=80, test_rows=20),
        engineered_metrics=ModelMetrics(accuracy=0.8, f1_score=0.79, roc_auc=None, train_rows=80, test_rows=20),
        features_created=[],
        features_skipped=[],
        accuracy_change=0.0,
        f1_change=0.0,
        roc_auc_change=None,
        improved=False,
        summary="No features engineered",
    )
    model_comparison.compare.return_value = comparison

    # Construct WITHOUT passing feature_planner
    orchestrator = AEGISOrchestrator(
        data_quality_agent=_make_mock_data_quality_agent(),
        eda_agent=_make_mock_eda_agent(),
        feature_executor=FeatureEngineeringExecutor(),
        model_comparison=model_comparison,
        critic_agent=CriticAgent(),
        loader=MagicMock(return_value=_make_minimal_df()),
        profiler=MagicMock(return_value=_make_minimal_profile()),
    )

    result = orchestrator.run("dummy.csv", "target")

    # The comparison should receive empty specs — not auto-generated ones
    model_comparison.compare.assert_called_once()
    call_args = model_comparison.compare.call_args
    assert call_args[1]["feature_specs"] == []
    assert result.created_features == []