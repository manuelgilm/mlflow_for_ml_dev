import mlflow
import random
from sklearn.ensemble import RandomForestClassifier


def create_registered_models_batch(n_models: int, n_versions: int):
    """
    Create a batch of registered models with random tags.

    :param experiment: Experiment object
    """
    client = mlflow.MlflowClient()
    registered_model_names = [f"model_{i}" for i in range(1, n_models)]

    for registered_model_name in registered_model_names:
        model_tags = {
            "model_type": random.choice(["random_forest", "logistic_regression"]),
            "model_owner": random.choice(["Alice", "Bob"]),
            "organization": random.choice(["Acme", "Umbrella"]),
        }

        # create n versions of the model
        for i in range(1, n_versions):
            # log a dummy model
            with mlflow.start_run(run_name=f"classifier_{i}") as run:
                rfc = RandomForestClassifier()
                mlflow.sklearn.log_model(
                    rfc, "model", registered_model_name=registered_model_name
                )

                # set tags for the registered model
                for key, value in model_tags.items():
                    client.set_registered_model_tag(
                        name=registered_model_name, key=key, value=value
                    )

            model_version_tags = {
                "validation_status": random.choice(
                    ["pending", "in progress", "completed", "failed"]
                ),
                "task_type": random.choice(
                    ["classification", "regression", "clustering"]
                ),
            }

            # set tags for the model version
            for tag_key, tag_value in model_version_tags.items():
                client.set_model_version_tag(
                    name=registered_model_name,
                    version=str(i),
                    key=tag_key,
                    value=tag_value,
                )
