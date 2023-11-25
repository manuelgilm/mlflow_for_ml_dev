import mlflow
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
import pandas as pd
from typing import Dict

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


import matplotlib.pyplot as plt


def set_or_create_experiment(experiment_name: str) -> str:
    """
    Get or create an experiment.

    :param experiment_name: Name of the experiment.
    :return: Experiment ID.
    """

    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    finally:
        mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id


def get_performance_plots(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, prefix: str
) -> Dict[str, any]:
    """
    Get performance plots.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param prefix: Prefix for the plot names.
    :return: Performance plots.
    """
    roc_figure = plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    cm_figure = plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    pr_figure = plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    return {
        f"{prefix}_roc_curve": roc_figure,
        f"{prefix}_confusion_matrix": cm_figure,
        f"{prefix}_precision_recall_curve": pr_figure,
    }


def get_classification_metrics(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, prefix: str
) -> Dict[str, float]:
    """
    Log classification metrics.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param prefix: Prefix for the metric names.
    :return: Classification metrics.
    """
    metrics = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred),
        f"{prefix}_recall": recall_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_pred),
    }

    return metrics

def register_model_with_client(model_name: str, run_id: str, artifact_path: str):
    """
    Register a model.

    :param model_name: Name of the model.
    :param run_id: Run ID.  
    :param artifact_path: Artifact path.

    :return: None.
    """
    client = mlflow.tracking.MlflowClient()
    client.create_registered_model(model_name)
    client.create_model_version(name=model_name, source=f"runs:/{run_id}/{artifact_path}")
        