from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from typing import List
from sklearn.ensemble import RandomForestClassifier


def get_sk_pipeline(
    categorical_columns: List[str], numerical_columns: List[str]
) -> Pipeline:
    """
    Get a sklearn pipeline with a column transformer for preprocessing.

    :param categorical_columns: List of categorical columns
    :param numerical_columns: List of numerical columns
    :return: sklearn pipeline
    """

    preprocessing = ColumnTransformer(
        transformers=[
            ("numerical_imputer", SimpleImputer(strategy="median"), numerical_columns),
            ("encoder", OneHotEncoder(), categorical_columns),
        ]
    )

    pipeline = Pipeline(
        steps=[("preprocessing", preprocessing), ("model", RandomForestClassifier())]
    )

    return pipeline
