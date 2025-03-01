from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from typing import List
from typing import Optional


def get_sklearn_pipeline(
    numerical_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
) -> Pipeline:
    """
    Get a scikit-learn pipeline for a classification task.
    raise ValueError: If both numerical_columns and categorical_columns are None.

    :param numerical_columns: List of numerical columns.
    :param categorical_columns: List of categorical columns.
    :return pipeline: The scikit-learn pipeline.
    """

    if not numerical_columns and not categorical_columns:
        raise ValueError(
            "At least one of numerical_columns or categorical_columns must be provided."
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), numerical_columns),
            ("categorical", OneHotEncoder(), categorical_columns),
        ]
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
    )

    return pipeline

def create_features():
    pass 

def get_features():
    pass
