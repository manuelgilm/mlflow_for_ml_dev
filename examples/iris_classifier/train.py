from sklearn.ensemble import RandomForestClassifier
from examples.iris_classifier.data import get_train_test_data
from examples.utils.decorators import mlflow_tracking_uri
from examples.utils.decorators import mlflow_client
from examples.utils.decorators import mlflow_experiment
from examples.utils.mlflow_utils import set_alias_to_latest_version

from mlflow.models import infer_signature
import mlflow


@mlflow_tracking_uri
@mlflow_experiment(name="iris_classifier")
@mlflow_client
def main(**kwargs) -> None:
    """
    Main function to train a Random Forest Classifier on the Iris dataset.
    This function retrieves training and test data, trains the model, logs parameters,
    """

    # Get training and test data
    x_train, x_test, y_train, y_test = get_train_test_data()

    # define parameters and create the model
    params = {"n_estimators": 1, "max_depth": 10}
    model = RandomForestClassifier(**params)

    # Train the model
    model.fit(x_train, y_train)

    # infer model signature
    model_signature = infer_signature(x_train, y_train)

    # create evaluationn data
    eval_data = x_test.copy()
    eval_data["target"] = y_test

    # Get the MLflow client
    client = kwargs["mlflow_client"]
    registered_model_name = "Iris_Classifier_Model"
    with mlflow.start_run(run_name="training-rfc-model") as run:
        # log parameters.
        mlflow.log_params(model.get_params())

        # log model.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=model_signature,
            input_example=x_train.iloc[0:3],
            registered_model_name=registered_model_name,
        )

        # set model version alias to "production"
        set_alias_to_latest_version(
            registered_model_name=registered_model_name,
            alias="production",
            client=client,
        )

        # log evaluation metrics
        mlflow.evaluate(
            model=f"runs:/{run.info.run_id}/model",
            data=eval_data,
            model_type="classifier",
            targets="target",
        )
