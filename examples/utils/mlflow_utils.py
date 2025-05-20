import mlflow
from typing import Dict
from typing import Optional
from pathlib import Path


def get_or_create_experiment(
    name: str, tags: Optional[Dict[str, str]] = None
) -> mlflow.entities.Experiment:
    """
    Get or create an experiment in MLflow.

    :param name: Name of the experiment.
    :param tags: Tags to set for the experiment.
    :return experiment: The experiment object.
    """

    experiment = mlflow.get_experiment_by_name(name=name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name, tags=tags)
        print(f"Experiment created with ID: {experiment_id}")

    experiment = mlflow.set_experiment(experiment_name=name)

    return experiment


def set_mlflow_tracking_uri(path: str) -> None:
    """
    Set the MLflow tracking URI.

    :param path: Path to the MLflow tracking server.
    """

    mlflow.set_tracking_uri(path)


def set_alias_to_latest_version(
    registered_model_name: str, alias: str, client: mlflow.MlflowClient
) -> None:
    """
    Set the alias to the latest version of the model.

    :param registered_model_name: Name of the model.
    :param alias: Alias to set for the model version.
    :param client: MLflow client.
    """
    # Set the model version alias to "production"
    model_version = mlflow.search_model_versions(
        filter_string=f"name='{registered_model_name}'",
        max_results=1,
    )[0]
    print(f"Model version: {model_version.version}")
    print(f"Model name: {model_version.name}")
    client.set_registered_model_alias(
        name=registered_model_name,
        version=model_version.version,
        alias=alias,
    )
    print(f"Alias '{alias}' set to model version {model_version.version}.")
