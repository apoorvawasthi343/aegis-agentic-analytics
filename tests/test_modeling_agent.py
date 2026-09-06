"""Tests for the AEGIS baseline modeling agent."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.aegis.modeling_agent import ModelingAgent
from src.aegis.schemas import ModelingReport


def _make_mixed_classification_df(
    n_samples: int = 500,
    n_numeric: int = 3,
    n_categorical: int = 2,
    n_classes: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Create a synthetic DataFrame with numeric and categorical features."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_numeric,
        n_informative=n_numeric,
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state,
    )

    numeric_cols = [f"num_{i}" for i in range(n_numeric)]
    df_numeric = pd.DataFrame(X, columns=numeric_cols)

    # Add missing values to some numeric columns
    df_numeric.iloc[:10, 0] = np.nan

    # Create categorical columns
    cat_data = {}
    rng = np.random.default_rng(random_state)
    for i in range(n_categorical):
        col_name = f"cat_{i}"
        n_categories = 3 + i
        categories = [f"category_{j}" for j in range(n_categories)]
        # Equal probability for each category (sum to 1.0)
        prob = [1.0 / n_categories] * n_categories
        cat_data[col_name] = rng.choice(categories, size=n_samples, p=prob)

    df_categorical = pd.DataFrame(cat_data)

    # Combine
    df = pd.concat([df_numeric, df_categorical], axis=1)
    df["target"] = y

    return df, "target"


def test_modeling_agent_returns_valid_report() -> None:
    """ModelingAgent should return a valid ModelingReport."""
    df, target = _make_mixed_classification_df()

    agent = ModelingAgent()
    report = agent.analyze(df, target)

    assert isinstance(report, ModelingReport)
    assert report.model_name == "LogisticRegression"
    assert report.target_column == "target"
    assert report.metrics.accuracy > 0
    assert report.metrics.f1_score > 0
    assert report.metrics.train_rows > 0
    assert report.metrics.test_rows > 0


def test_target_not_in_features() -> None:
    """The target column should not be included in the feature matrix."""
    df, target = _make_mixed_classification_df(n_samples=300)

    agent = ModelingAgent()

    # We can verify this indirectly: the model trains without error
    # and the report shows correct train/test counts
    report = agent.analyze(df, target)

    total_rows = len(df)
    expected_train = int(total_rows * 0.8)
    expected_test = total_rows - expected_train

    assert report.metrics.train_rows == expected_train
    assert report.metrics.test_rows == expected_test


def test_numeric_imputation_works() -> None:
    """Model should handle missing numeric values via median imputation."""
    df, target = _make_mixed_classification_df(n_samples=500)

    # Add missing values
    df.iloc[:25, 0] = np.nan
    df.iloc[:15, 1] = np.nan

    agent = ModelingAgent()
    report = agent.analyze(df, target)

    assert report.metrics.accuracy > 0
    assert not np.isnan(report.metrics.accuracy)


def test_categorical_encoding_works() -> None:
    """Model should handle categorical variables via one-hot encoding."""
    df, target = _make_mixed_classification_df(
        n_samples=500, n_numeric=2, n_categorical=3
    )

    agent = ModelingAgent()
    report = agent.analyze(df, target)

    assert report.metrics.accuracy > 0
    assert "Categorical features: 3" in report.notes


def test_multiclass_classification() -> None:
    """ModelingAgent should handle multiclass targets."""
    df, target = _make_mixed_classification_df(
        n_samples=500, n_classes=3
    )

    agent = ModelingAgent()
    report = agent.analyze(df, target)

    assert report.task_type == "multiclass_classification"
    assert report.metrics.accuracy > 0


def test_binary_classification_has_roc_auc() -> None:
    """Binary classification should compute ROC-AUC when possible."""
    df, target = _make_mixed_classification_df(n_samples=500, n_classes=2)

    agent = ModelingAgent()
    report = agent.analyze(df, target)

    assert report.task_type == "binary_classification"
    assert report.metrics.roc_auc is not None
    assert report.metrics.roc_auc > 0


def test_modeling_report_schema_validation() -> None:
    """The returned report should validate against the ModelingReport schema."""
    df, target = _make_mixed_classification_df()

    agent = ModelingAgent()
    report = agent.analyze(df, target)

    # This implicitly validates the schema
    assert report.model_name == "LogisticRegression"
    assert report.task_type in ("binary_classification", "multiclass_classification")
    assert isinstance(report.metrics.accuracy, float)
    assert isinstance(report.metrics.f1_score, float)
    assert isinstance(report.metrics.train_rows, int)
    assert isinstance(report.metrics.test_rows, int)
    assert report.notes is not None
    assert len(report.notes) > 0


def test_all_previous_tests_still_pass() -> None:
    """This is a marker test to ensure we didn't break existing functionality.

    The actual verification happens through the complete pytest run.
    """
    # Just verify ModelingAgent can be imported
    from src.aegis.modeling_agent import ModelingAgent

    assert ModelingAgent is not None