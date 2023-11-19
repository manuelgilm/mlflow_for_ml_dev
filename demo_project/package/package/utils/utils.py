import mlflow 
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
import pandas as pd 
from typing import Dict

import matplotlib.pyplot as plt
def set_or_create_experiment(experiment_name:str)->str:
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
    
def get_performance_plots(y_true:pd.DataFrame, y_pred:pd.DataFrame, prefix:str)->Dict[str, any]:
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
    cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    pr_figure = plt.figure()
    pr_curve = PrecisionRecallDisplay.from_predictions(y_true, y_pred, ax=plt.gca())

    return {
        f"{prefix}_roc_curve": roc_figure,
        f"{prefix}_confusion_matrix": cm_figure,
        f"{prefix}_precision_recall_curve": pr_figure
    }


