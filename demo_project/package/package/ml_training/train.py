import pandas as pd
from sklearn.pipeline import Pipeline
import mlflow
from mlflow.models.signature import infer_signature

from typing import Tuple


def train_model(
    pipeline: Pipeline, run_name: str,model_name:str, artifact_path: str, x: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[str, Pipeline]:
    """
    Train a model and log it to MLflow.

    :param pipeline: Pipeline to train.
    :param run_name: Name of the run.
    :param x: Input features.
    :param y: Target variable.
    :return: Run ID.
    """

    signature = infer_signature(x, y)
    with mlflow.start_run(run_name=run_name) as run:
        pipeline = pipeline.fit(x, y)
        mlflow.sklearn.log_model(sk_model=pipeline, artifact_path=artifact_path,signature=signature, registered_model_name=model_name)
    return run.info.run_id, pipeline
