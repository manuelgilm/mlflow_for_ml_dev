from examples.utils.file_utils import get_root_dir
from examples.utils.mlflow_utils import get_or_create_experiment
import mlflow

from typing import Optional
from typing import Dict


def mlflow_experiment(name: str, tags: Optional[Dict[str, str]] = None):
    """
    Decorator to set the MLflow experiment name and tags.
    This decorator creates a new experiment if it doesn't exist.
    The decorator modifies the function to accept an `experiment` argument,
    which is the MLflow experiment object.

    :param name: The name of the MLflow experiment.
    :param tags: A dictionary of tags to associate with the experiment.
    :return: A decorator function that sets the MLflow experiment.
    """

    def decorator(func, name=name, tags=tags):
        """
        Decorator function to set the MLflow experiment name and tags.
        It creates a new experiment if it doesn't exist.
        The decorator modifies the function to accept an `experiment` argument,
        which is the MLflow experiment object.

        :param func: The function to be decorated.
        :param name: The name of the MLflow experiment.
        :param tags: A dictionary of tags to associate with the experiment.
        :return: The decorated function with the MLflow experiment set.
        """

        def wrapper(*args, **kwargs):
            print(f"Setting MLflow experiment: {name}")
            experiment = get_or_create_experiment(name=name, tags=tags)
            kwargs["mlflow_experiment"] = experiment
            return func(*args, **kwargs)

        return wrapper

    return decorator


def mlflow_client(func):
    """
    Decorator to pass the MLflow client as argument to the function.
    """

    def wrapper(*args, **kwargs):
        print("Setting MLflow client...")
        mlflow_client = mlflow.MlflowClient()
        kwargs["mlflow_client"] = mlflow_client
        return func(*args, **kwargs)

    return wrapper


def mlflow_tracking_uri(func):
    """
    Set the MLflow tracking URI to the local file system.
    """
    print(func.__name__)

    def wrapper(*args, **kwargs):
        print("Setting MLflow tracking URI...")
        mlflow.set_tracking_uri(uri=(get_root_dir() / "mlruns").as_uri())
        return func(*args, **kwargs)

    return wrapper
