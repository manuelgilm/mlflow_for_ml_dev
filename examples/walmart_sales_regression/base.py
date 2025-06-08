import mlflow
from mlflow.types import Schema
from mlflow.types.schema import ColSpec
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ParamSpec
from mlflow.types.schema import ParamSchema
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import List
from typing import Optional
from mlflow.models import infer_signature
from pathlib import Path
import platform


class WalmartSalesRegressor(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow model for sales regression.
    """

    def __init__(self, config):
        """
        Initialize the WalmartSalesRegressor.
        """
        self.numerical_features = config["numerical_features"]
        self.categorical_features = config["categorical_features"]
        self.target = config["target"]
        self.artifact_uris = {}

    def load_context(self, context):
        """
        Load the context for the model, which includes the artifact URIs.

        :param context: The context object containing the model.
        :return: None
        """
        if platform.system() == "Linux":
            # Convert Windows-style paths to POSIX paths for Linux compatibility
            print(
                "Converting Windows-style paths to POSIX paths for Linux compatibility..."
            )
            context_artifacts = {
                key: value.replace("\\", "/")
                for key, value in context.artifacts.items()
            }

        else:
            # Use the context artifacts as is for non-Linux systems
            print("Using context artifacts as is for non-Linux systems...")
            context_artifacts = context.artifacts

        print("Loading model artifacts from context...")
        self.models = {
            store_id: mlflow.sklearn.load_model(uri)
            for store_id, uri in context_artifacts.items()
        }
        print(f"Model artifact URIs loaded: {context_artifacts}")

    def fit_model(self, x_train, y_train, store_id: int, run_id: str):
        """
        Fits a single model to the training data for a specific store.

        :param x_train: Training features.
        :param y_train: Training target variable.
        :param store_id: The store ID for which to fit the model.
        :param run_id: The ID of the parent MLflow run.
        :return: None
        """
        store_data = x_train[x_train["Store"] == store_id]
        store_target = y_train[y_train["Store"] == store_id]

        pipeline = self._get_sklearn_pipeline(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
        )

        pipeline.fit(store_data, store_target[self.target])
        model_signature = infer_signature(
            store_data[self.categorical_features + self.numerical_features],
            store_target[self.target],
        )

        with mlflow.start_run(
            run_name=f"run_store_{store_id}", parent_run_id=run_id, nested=True
        ) as run:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=f"model_store_{store_id}",
                signature=model_signature,
                input_example=store_data[
                    self.categorical_features + self.numerical_features
                ].iloc[0:1],
            )
            mlflow.log_params({"store_id": store_id})
            self.artifact_uris[str(store_id)] = (
                f"runs:/{run.info.run_id}/model_store_{store_id}"
            )

    def predict(self, context, model_input, params=None):
        """
        Perform prediction using the model.

        :param context: The context object containing the model.
        :param model_input: Input data for prediction.
        :param params: Additional parameters for prediction.
        :return: Predicted values.
        """
        if params is not None:
            store_id = params.get("store_id", "1")
            if store_id not in self.artifact_uris.keys():
                raise ValueError(f"Model for store ID {store_id} not found.")
            return self._predict(store_id, model_input)
        else:
            return self._predict(None, model_input)

    def _get_model_signature(self) -> ModelSignature:
        """
        Get the model signature for the MLflow model.

        :return: Model signature object.
        """
        feature_specification = [
            ColSpec(type="long", name="Holiday_Flag"),
            ColSpec(type="double", name="Temperature"),
            ColSpec(type="double", name="Fuel_Price"),
            ColSpec(type="double", name="CPI"),
            ColSpec(type="double", name="Unemployment"),
        ]

        param_specification = [
            ParamSpec(dtype="string", name="store_id", default="1"),
        ]
        param_schema = ParamSchema(
            params=param_specification,
        )
        input_schema = Schema(inputs=feature_specification)
        output_schema = Schema(
            inputs=[ColSpec(type="float", name=self.target)],
        )
        signature = ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
            params=param_schema,
        )

        return signature

    def _get_sklearn_pipeline(
        self, numerical_features: List[str], categorical_features: List[str]
    ) -> Pipeline:
        """
        Get a scikit-learn pipeline for preprocessing and model training.

        :param numerical_features: List of numerical feature names.
        :param categorical_features: List of categorical feature names.
        :return: A scikit-learn pipeline object.
        """

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="mean"), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )

        return pipeline

    def _predict(self, store_id: Optional[str], x):
        """
        Predicts the target variable using the fitted model.

        :param store_id: The store ID for which to make predictions.
        :param x: The input data for prediction.
        :return: The predicted values.
        """
        if store_id is None:
            # If no store_id is provided, use the first available model
            store_id = list(self.artifact_uris.keys())[0]
            print(f"No store_id provided, using default store_id: {store_id}")

        model = self.models.get(store_id)
        predictions = model.predict(x)

        return predictions
