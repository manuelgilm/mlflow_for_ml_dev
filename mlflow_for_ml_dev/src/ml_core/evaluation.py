from typing import Union
from typing import Optional
from typing import Dict

import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt


def get_confusion_matrix(y_pred: pd.Series, y_true: pd.Series) -> pd.DataFrame:
    """
    Get confusion matrix from predictions and true labels.

    :param y_pred: Predictions
    :param y_true: True labels
    :return: Confusion matrix
    """
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    return cm_display.figure_
    
