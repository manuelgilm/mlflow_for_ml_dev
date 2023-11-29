from mlflow_utils import create_dataset
from mlflow_utils import create_mlflow_experiment

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import hp

from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
import mlflow
from functools import partial


def get_classification_metrics(
    y_true: pd.Series, y_pred: pd.Series, prefix: str
) -> Dict[str, float]:
    """
    Get the classification metrics.

    :param y_true: The true target values.
    :param y_pred: The predicted target values.
    :param prefix: The prefix of the metric names.
    :return: The classification metrics.
    """

    return {
        f"{prefix}_accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
        f"{prefix}_precision": precision_score(y_true=y_true, y_pred=y_pred),
        f"{prefix}_recall": recall_score(y_true=y_true, y_pred=y_pred),
        f"{prefix}_f1": f1_score(y_true=y_true, y_pred=y_pred),
    }


def get_sklearn_pipeline(
    numerical_features: List[str], categorical_features: Optional[List[str]] = []
) -> Pipeline:
    """
    Get the sklearn pipeline.

    :param numerical_features: The numerical features.
    :param categorical_features: The categorical features.
    :return: The sklearn pipeline.
    """

    preprocessing = ColumnTransformer(
        transformers=[
            ("numerical", SimpleImputer(strategy="median"), numerical_features),
            (
                "categorical",
                OneHotEncoder(),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", RandomForestClassifier()),
        ]
    )

    return pipeline


def objective_function(
    params: Dict,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
) -> float:
    """
    Objective function to minimize.

    :param params: The hyperparameter values to evaluate.
    :param x_train: The training data.
    :param x_test: The test data.
    :param y_train: The training target.
    :param y_test: The test target.
    :param numerical_features: The numerical features.
    :param categorical_features: The categorical features.
    :return: The score of the model.
    """

    pipeline = get_sklearn_pipeline(numerical_features=numerical_features)
    params.update({"model__max_depth": int(params["model__max_depth"])})
    params.update({"model__n_estimators": int(params["model__n_estimators"])})
    pipeline.set_params(**params)
    with mlflow.start_run(nested=True) as run:
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        metrics = get_classification_metrics(
            y_true=y_test, y_pred=y_pred, prefix="test"
        )

        mlflow.log_params(pipeline["model"].get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, f"{run.info.run_id}-model")
    return -metrics["test_f1"]


if __name__ == "__main__":
    df = create_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop("target", axis=1),
        df["target"],
        test_size=0.2,
        random_state=42,
    )

    numerical_features = [f for f in x_train.columns if f.startswith("feature")]
    print(numerical_features)

    space = {
        "model__n_estimators": hp.quniform("model__n_estimators", 20, 200, 10),
        "model__max_depth": hp.quniform("model__max_depth", 10, 100, 10),
    }

    experiment_id = create_mlflow_experiment(
        "hyperopt_experiment",
        artifact_location="hyperopt_mlflow_artifacts",
        tags={"mlflow.note.content": "hyperopt experiment"},
    )
    with mlflow.start_run(run_name="hyperparameter_opmization") as run:
        best_params = fmin(
            fn=partial(
                objective_function,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                numerical_features=numerical_features,
                categorical_features=None,
            ),
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=Trials(),
        )

        pipeline = get_sklearn_pipeline(numerical_features=numerical_features)

        best_params.update({"model__max_depth": int(best_params["model__max_depth"])})
        best_params.update(
            {"model__n_estimators": int(best_params["model__n_estimators"])}
        )

        pipeline.set_params(**best_params)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        metrics = get_classification_metrics(
            y_true=y_test, y_pred=y_pred, prefix="best_model_test"
        )

        mlflow.log_params(pipeline["model"].get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, f"{run.info.run_id}-best-model")
