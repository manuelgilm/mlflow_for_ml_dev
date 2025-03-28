from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple
from typing import Dict
import mlflow

from mlflow.models import ModelSignature
from mlflow.types.schema import Schema
from mlflow.types.schema import ColSpec
from mlflow.types.schema import ParamSchema
from mlflow.types.schema import ParamSpec


def get_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get the training and testing data
    """
    x, y = make_classification(n_samples=1000, n_features=20, n_classes=2)

    feature_columns = [f"feature_{i}" for i in range(x.shape[1])]
    target_column = "target"

    x = pd.DataFrame(x, columns=feature_columns)
    y = pd.Series(y, name=target_column)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return x_train, x_test, y_train, y_test


def get_model_signature(
    input: pd.DataFrame, output: pd.Series, params: Dict[str, str]
) -> ModelSignature:
    """
    Get the model signature.

    :param input: The input data
    :param output: The output data
    :param params: The model parameters
    :return: The model signature
    """
    inputs = [ColSpec(name=col, type="double") for col in input.columns]
    output = [ColSpec(name=output.name, type="double")]
    input_schema = Schema(inputs)
    output_schema = Schema(output)
    input_params = [
        ParamSpec(name=param, dtype="string", default=value)
        for param, value in params.items()
    ]
    param_schema = ParamSchema(input_params)
    return ModelSignature(
        inputs=input_schema, outputs=output_schema, params=param_schema
    )


def set_alias(model_name: str):
    """
    Set the alias for the model
    """
    # search versions
    client = mlflow.MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")

    # set alias to the latest version
    latest_version = model_versions[0].version
    client.set_registered_model_alias(
        name=model_name,
        version=latest_version,
        alias="champion",
    )

    # set alias to the second latest version
    second_latest_version = model_versions[1].version
    client.set_registered_model_alias(
        name=model_name,
        version=second_latest_version,
        alias="challenger",
    )
