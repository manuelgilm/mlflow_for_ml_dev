import mlflow
from typing import Any

import pandas as pd

from sklearn.datasets import make_classification


def create_mlflow_experiment(
    experiment_name: str, artifact_location: str, tags: dict[str, Any]
) -> str:
    """
    Create a new mlflow experiment with the given name and artifact location.

    Parameters:
    ----------
    experiment_name: str
        The name of the experiment to create.
    artifact_location: str
        The artifact location of the experiment to create.
    tags: dict[str,Any]
        The tags of the experiment to create.

    Returns:
    -------
    experiment_id: str
        The id of the created experiment.
    """
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except:
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id


def get_mlflow_experiment(
    experiment_id: str = None, experiment_name: str = None
) -> mlflow.entities.Experiment:
    """
    Retrieve the mlflow experiment with the given id or name.

    Parameters:
    ----------
    experiment_id: str
        The id of the experiment to retrieve.
    experiment_name: str
        The name of the experiment to retrieve.

    Returns:
    -------
    experiment: mlflow.entities.Experiment
        The mlflow experiment with the given id or name.
    """
    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided.")
    return experiment


def delete_mlflow_experiment(
    experiment_id: str = None, experiment_name: str = None
) -> None:
    """
    Delete the mlflow experiment with the given id or name.

    Parameters:
    ----------
    experiment_id: str
        The id of the experiment to delete.
    experiment_name: str
        The name of the experiment to delete.
    """
    if experiment_id is not None:
        mlflow.delete_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        mlflow.delete_experiment(experiment_id)
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided.")


def create_dataset(
    n_samples: int = 10000,
    n_features: int = 50,
    n_informative: int = 10,
    class_sep: float = 1.0,
) -> pd.DataFrame:
    """
    Create a dataset for testing purposes.

    :param n_samples: The number of samples.
    :param n_features: The number of features.
    :param n_informative: The number of informative features.
    :return: pd.DataFrame
    """

    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        class_sep=class_sep,
        random_state=42,
    )

    df = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y

    return df
