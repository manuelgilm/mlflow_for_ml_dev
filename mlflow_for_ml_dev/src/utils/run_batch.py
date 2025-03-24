import mlflow
import random


def create_runs_batch(experiment_id: str, n_runs: int):
    """
    Create a batch of runs in a given experiment.

    :param experiment_id: Experiment ID
    :param n_runs: Number of runs to create
    """
    random_tags = {
        "random_run": "yes",
        "mlflow.note.content": "This is a note",
        "project_type": random.choice(
            ["experiment", "production", "research", "development"]
        ),
        "algorithm_type": random.choice(
            ["linear", "tree", "neural network", "ensemble"]
        ),
    }
    for i in range(n_runs):
        with mlflow.start_run(experiment_id=experiment_id, tags=random_tags) as run:
            # do nothing
            mlflow.log_metric("metric_1", random.random())
            mlflow.log_metric("metric_2", random.random())
            mlflow.log_param("param_1", random.choice(["a", "b", "c"]))
            mlflow.log_param("param_2", random.choice(["x", "y", "z"]))
