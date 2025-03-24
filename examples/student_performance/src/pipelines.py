from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from typing import List
import pandas as pd


class TrainingPipeline:

    def __init__(
        self,
        algo: str,
        numerical_columns: List[str],
        categorical_columns: List[str],
        target_column: str,
    ) -> None:
        """
        Initialize the TrainingPipeline class.

        :param algo: The algorithm to use.
        :param numerical_columns: List of numerical columns.
        :param categorical_columns: List of categorical columns.
        :param target_column: The target column.
        """
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.pipeline = self._get_sklearn_pipeline(algo)

    def _get_model(self, algo: str):
        """
        Get the appropriate model based on the algorithm.
        Raise ValueError: If the model is not supported.

        :param algo: The algorithm to use.
        :return model: The scikit-learn model.
        """

        allowed_algos = {
            "random_forest": RandomForestClassifier,
            "decision_tree": DecisionTreeClassifier,
        }

        model = allowed_algos.get(algo, None)

        if not model:
            raise ValueError(f"Ml model {algo} not supported")
        return model

    def _get_sklearn_pipeline(
        self,
        algo: str,
    ) -> Pipeline:
        """
        Get a scikit-learn pipeline for a classification task.
        raise ValueError: If both numerical_columns and categorical_columns are None.

        :param algo: The algorithm to use.
        :return pipeline: The scikit-learn pipeline.
        """
        model = self._get_model(algo)
        if not self.numerical_columns and not self.categorical_columns:
            raise ValueError(
                "At least one of numerical_columns or categorical_columns must be provided."
            )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", StandardScaler(), self.numerical_columns),
                ("categorical", OneHotEncoder(), self.categorical_columns),
            ]
        )

        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model())]
        )

        return pipeline

    def train(self, x_train, y_train):
        """
        Train the pipeline.

        :param x_train: The training features.
        :param y_train: The training target.
        :return pipeline: The trained pipeline.
        """
        if self.pipeline is None:
            raise ValueError("The pipeline must be initialized before training.")
        self.pipeline.fit(x_train, y_train)
        return self.pipeline

    def predict(self, x_test) -> pd.DataFrame:
        """
        Score the pipeline.
        Raise ValueError: If the pipeline is not trained.

        :param x_test: The testing features.
        :return: The classification report.
        """
        if self.pipeline is None:
            raise ValueError("The pipeline must be trained before scoring.")

        predictions = self.pipeline.predict(x_test)
        predictions_proba = self.pipeline.predict_proba(x_test)
        predictions_proba = [max(p) for p in predictions_proba]

        return pd.DataFrame(
            {"predictions": predictions, "predictions_proba": predictions_proba}
        )
    
