"""Baseline classification modeling agent for AEGIS.

Provides a deterministic LogisticRegression baseline with safe
preprocessing using scikit-learn Pipelines.
"""

from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.aegis.schemas import ModelMetrics, ModelingReport


class ModelingAgent:
    """Agent that trains a baseline classification model on a dataset.

    Uses a scikit-learn Pipeline with separate numeric and categorical
    preprocessing, then trains LogisticRegression and returns a
    ModelingReport with accuracy, F1, and optional ROC-AUC.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        max_iter: int = 1000,
    ) -> None:
        """Initialize the modeling agent.

        Args:
            test_size: Fraction of data to hold out for testing.
            random_state: Random seed for train/test split.
            max_iter: max_iter passed to LogisticRegression.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.max_iter = max_iter

    def analyze(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> ModelingReport:
        """Train a baseline classifier and return a modeling report.

        Args:
            df: The full dataset including the target column.
            target_column: Name of the target column to predict.

        Returns:
            A ModelingReport containing model name, task type, target,
            performance metrics, and notes.
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Identify column types
        numeric_columns = X.select_dtypes(
            include=["number"]
        ).columns.tolist()
        categorical_columns = X.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        # Preprocessing for numeric features
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Preprocessing for categorical features
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False
                    ),
                ),
            ]
        )

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_columns),
                ("cat", categorical_transformer, categorical_columns),
            ]
        )

        # Full pipeline: preprocessing + classifier
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        # Stratified train/test split (fit ONLY on training data)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if len(y.unique()) > 1 else None,
        )

        # Fit the pipeline on training data only
        model.fit(X_train, y_train)

        # Predictions and probabilities
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

        # Compute metrics
        accuracy = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))

        roc_auc: Optional[float] = None
        if y_prob is not None and len(y.unique()) == 2:
            try:
                roc_auc = float(roc_auc_score(y_test, y_prob[:, 1]))
            except Exception:
                roc_auc = None

        metrics = ModelMetrics(
            accuracy=accuracy,
            f1_score=f1,
            roc_auc=roc_auc,
            train_rows=int(len(X_train)),
            test_rows=int(len(X_test)),
        )

        notes = (
            f"Baseline LogisticRegression model trained on {len(X_train)} "
            f"training rows and evaluated on {len(X_test)} test rows. "
            f"Numeric features: {len(numeric_columns)}, "
            f"Categorical features: {len(categorical_columns)}."
        )

        return ModelingReport(
            model_name="LogisticRegression",
            task_type="binary_classification"
            if len(y.unique()) == 2
            else "multiclass_classification",
            target_column=target_column,
            metrics=metrics,
            notes=notes,
        )