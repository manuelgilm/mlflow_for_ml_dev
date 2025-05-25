from sklearn.ensemble import RandomForestClassifier
from examples.iris_classifier.data import get_train_test_data
from examples.utils.decorators import mlflow_tracking_uri
from examples.utils.decorators import mlflow_client
from examples.utils.decorators import mlflow_experiment
from examples.utils.mlflow_utils import set_alias_to_latest_version
from typing import Optional
from typing import Dict

from mlflow.models import infer_signature
import mlflow


def train(x_train, y_train, params: Optional[Dict[str, str]]) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier on the provided training data.
    The function returns the trained model.

    :param x_train: The training features (input data).
    :param y_train: The training labels (target data).
    :return: The trained Random Forest Classifier model.
    """
    clf = RandomForestClassifier(**params)
    clf.fit(x_train, y_train)
    return clf


@mlflow_tracking_uri
@mlflow_experiment(name="iris_classifier")
@mlflow_client
def main(**kwargs) -> None:
    # Example usage of the train function
    x_train, x_test, y_train, y_test = get_train_test_data()
    params = {"n_estimators": 1, "max_depth": 10}
    model = train(x_train, y_train, params)
    model_signature = infer_signature(x_train, y_train)

    eval_data = x_test.copy()
    eval_data["target"] = y_test
    client = kwargs["mlflow_client"]
    registered_model_name = "Iris_Classifier_Model"
    with mlflow.start_run(run_name="training-rfc-model") as run:
        # log parameters.
        mlflow.log_params(model.get_params())

        # log model
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

        # model uri
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.evaluate(
            model=model_uri,
            data=eval_data,
            model_type="classifier",
            targets="target",
        )
