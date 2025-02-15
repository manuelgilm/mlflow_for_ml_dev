import random 
from typing import List
import mlflow 

def create_experiments_batch(n_exp:int)->List[str]:
    """ 
    Create a batch of experiments with random tags.

    :param n_exp: Number of experiments to create
    """
    artificial_experiments = n_exp
    experiment_ids = []

    for i in range(artificial_experiments):
        exp_name = f"experiment_{i}"
        tags = {
            "random_experiment": "yes",
            "mlflow.note.content": "This is a note",
            "project_type": random.choice(["experiment", "production", "research", "development"]),
            "algorithm_type": random.choice(["linear", "tree", "neural network", "ensemble"]),
            "data_type": random.choice(["structured", "unstructured", "semi-structured"]),
            "data_source": random.choice(["csv", "database", "api", "cloud", "streaming"]),
            "data_quality": random.choice(["high", "medium", "low"]),
            "data_size": random.choice(["small", "medium", "large"]),
            "inference_type": random.choice(["batch", "online"]),
        }
        experiment_id = mlflow.create_experiment(name=exp_name, tags=tags)
        experiment_ids.append(experiment_id)

    return experiment_ids

def delete_experiments_batch(experiment_ids:List[str]):
    """
    Delete the experiments created in the batch.

    :param experiment_ids: List of experiment ids to delete

    """

    for exp_id in experiment_ids:
        mlflow.delete_experiment(exp_id)