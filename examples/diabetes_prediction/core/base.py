import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List
from typing import Optional
from typing import Dict
import pandas as pd


class DiabetesPrediction(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow model class for diabetes prediction.
    Inherits from mlflow.pyfunc.PythonFunction to define a custom model.
    """

    def __init__(self):
        """
        Initialize the model.
        """
        self.model = None

    def fit(self, x_train, y_train):
        """
        Fit the model using the training data.
        This method is called when training the model.

        :param x_train: The input features for training.
        :param y_train: The target variable for training.
        """
        numerical_features = self._get_numerical_features()
        categorical_features = self._get_categorical_features()
        pipeline = self.get_sklearn_pipeline(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        # train the model
        self.model = pipeline.fit(x_train, y_train)

    def predict(self, context, model_input, params: Optional[Dict[str, str]] = {}):
        """
        Predict method for the custom model.
        This method is called when making predictions with the model.

        :param context: The context object containing information about the model and input data.
        :param model_input: The input data for prediction.
        :return: The predicted output.
        """
        if not self.model:
            print("Model not loaded")
            return None

        if params.get("probabilities", None):
            predictions_df = self._predict_with_probabilities(model_input)
            return predictions_df

        predictions = self.model.predict(model_input)
        # Convert predictions to DataFrame for better readability
        predictions_df = pd.DataFrame({"predictions": predictions})

        return predictions_df

    def _predict_with_probabilities(self, model_input):
        """
        Predict method for the custom model with probabilities.
        This method is called when making predictions with the model.

        :param model_input: The input data for prediction.
        :return: The predicted output with probabilities.
        """
        if not self.model:
            print("Model not loaded")
            return None

        # Get the probabilities of each class
        probabilities = self.model.predict_proba(model_input)

        # Convert predictions to DataFrame for better readability
        predictions_df = pd.DataFrame(
            {
                "predictions": self.model.predict(model_input),
                "prob_0": probabilities[:, 0],
                "prob_1": probabilities[:, 1],
            }
        )

        return predictions_df

    def _get_numerical_features(self):
        """
        Get the numerical features for the model.
        :return: A list of numerical feature names.
        """
        return ["age", "bmi", "HbA1c_level", "blood_glucose_level"]

    def _get_categorical_features(self):
        """
        Get the categorical features for the model.
        :return: A list of categorical feature names.
        """
        return [
            "heart_disease",
            "smoking_history",
            "hypertension",
            "gender",
        ]

    def get_sklearn_pipeline(
        self, numerical_features: List[str], categorical_features: List[str]
    ) -> Pipeline:
        """
        Get the sklearn pipeline for the diabetes prediction model.

        :param numerical_features: List of numerical feature names.
        :param categorical_features: List of categorical feature names.
        :return: A sklearn pipeline object.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(), numerical_features),
                ("cat", OneHotEncoder(), categorical_features),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier()),
            ]
        )
        return pipeline
