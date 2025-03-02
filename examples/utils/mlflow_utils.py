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

    experiment = mlflow.get_experiment_by_name(name=name, tags=tags)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name)
        print(f"Experiment created with ID: {experiment_id}")

    experiment = mlflow.set_experiment(experiment_name=name)

    return experiment

def set_mlflow_tracking_uri(path: str) -> None:
    """
    Set the MLflow tracking URI.

    :param path: Path to the MLflow tracking server.
    """

    mlflow.set_tracking_uri(path)