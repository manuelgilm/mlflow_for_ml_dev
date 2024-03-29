from mlflow_for_ml_dev.utils.utils import get_root_project
from mlflow.entities.experiment import Experiment

from typing import Optional 
from typing import Dict

import mlflow

def get_or_create_experiment(
    experiment_name: str, tags: Optional[Dict[str, str]] = None
) -> Experiment:
    """
    Get or create an experiment.

    :param experiment_name: The name of the experiment.
    :param tags: A dictionary of tags to add to the experiment.
    :return: The experiment_id.
    """

    # Get the root project directory
    project_dir = get_root_project()

    # set the tracking uri
    tracking_uri = (project_dir / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # If the experiment does not exist, create it
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name, tags=tags)
        experiment = mlflow.get_experiment(experiment_id)

    mlflow.set_experiment(experiment_name=experiment_name)

    return experiment


def print_experiment_info(experiment: Experiment):
    """
    Print experiment information.

    :param experiment: An instance of mlflow.entities.experiment.Experiment.
    """
    print("\n Experiment Information \n")
    print(f"Name: {experiment.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print("\n \n")

