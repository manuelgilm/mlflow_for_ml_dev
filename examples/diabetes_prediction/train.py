from examples.diabetes_prediction.core.data import get_train_test_data
from examples.diabetes_prediction.core.base import DiabetesPrediction
from examples.diabetes_prediction.core.pipeline import get_model_signature
from examples.diabetes_prediction.core.data import get_feature_spec
from examples.utils.decorators import mlflow_tracking_uri
from examples.utils.decorators import mlflow_client
from examples.utils.decorators import mlflow_experiment

import mlflow


@mlflow_tracking_uri
@mlflow_client
@mlflow_experiment(name="diabetes_prediction")
def main(**kwargs) -> None:
    """
    Trains the diabetes prediction model and logs it to MLflow.
    This function loads the training data, fits the model, and logs it to MLflow.
    """
    x_train, x_test, y_train, y_test = get_train_test_data()
    diabetes_model = DiabetesPrediction()
    diabetes_model.fit(x_train, y_train)
    feature_spec = get_feature_spec()
    signature = get_model_signature(feature_spec=feature_spec)
    with mlflow.start_run() as run:
        # Log the model
        registered_model_name = "Diabetes_Prediction_Model"
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=diabetes_model,
            registered_model_name=registered_model_name,
            signature=signature,
            infer_code_paths=True,
        )

        # Set model version alias to "production"
        model_version = mlflow.search_model_versions(
            filter_string=f"name='{registered_model_name}'", max_results=1
        )[0]
        client = kwargs["mlflow_client"]
        client.set_registered_model_alias(
            name=registered_model_name,
            version=model_version.version,
            alias="production",
        )
        eval_data = x_test.copy()
        eval_data["diabetes"] = y_test

        pred_df = diabetes_model.predict(context=None, model_input=x_test)
        eval_data["predictions"] = pred_df["predictions"].values
        mlflow.evaluate(
            data=eval_data,
            targets="diabetes",
            predictions="predictions",
            model_type="classifier",
        )
