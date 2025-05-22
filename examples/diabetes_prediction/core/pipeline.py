from typing import List
from mlflow.types import Schema
from mlflow.types import ColSpec
from mlflow.types import ParamSchema
from mlflow.types import ParamSpec
from typing import Dict
from mlflow.models import ModelSignature


def get_model_signature(feature_spec: List[Dict[str, str]]) -> ModelSignature:
    """
    Return the model signature for the diabetes prediction model.

    :param feature_spec: List of feature specifications.
    :return: A dictionary containing the input and output schema for the model.
    """

    input_spec = [
        ColSpec(type=feature["type"], name=feature["name"]) for feature in feature_spec
    ]
    output_spec = [
        ColSpec(type="integer", name="predictions", required=True),
        ColSpec(type="float", name="prob_0", required=False),
        ColSpec(type="float", name="prob_1", required=False),
    ]

    parameter_spec = [
        ParamSpec(dtype="boolean", name="probabilities", default=False),
    ]

    input_schema = Schema(input_spec)
    output_schema = Schema(output_spec)
    param_schema = ParamSchema(parameter_spec)
    signature = ModelSignature(
        inputs=input_schema, outputs=output_schema, params=param_schema
    )
    return signature
