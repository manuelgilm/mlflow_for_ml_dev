import pandas as pd
from typing import Tuple
from typing import Dict
from typing import List
from sklearn.model_selection import train_test_split


def get_categorical_features() -> List[str]:
    """
    Get the categorical features for the diabetes prediction model.
    :return: A list of categorical feature names.
    """
    return [
        "heart_disease",
        "smoking_history",
        "hypertension",
        "gender",
    ]


def get_numerical_features() -> List[str]:
    """
    Get the numerical features for the diabetes prediction model.
    :return: A list of numerical feature names.
    """
    return ["age", "bmi", "HbA1c_level", "blood_glucose_level"]


def get_feature_spec() -> List[Dict[str, str]]:
    """
    Get the feature specification for the diabetes prediction model.
    :return: A dictionary mapping feature names to their types.
    """
    data = pd.read_csv(
        "/Users/manue/Downloads/diabetes-prediction-dataset/diabetes_prediction_dataset.csv"
    )
    feature_spec = []

    for column in data.columns:
        if column == "diabetes":
            continue
        dtype = data[column].dtype
        if dtype == "object":
            feature_spec.append({"name": column, "type": "string"})
        elif dtype == "int64":
            feature_spec.append({"name": column, "type": "long"})
        elif dtype == "float64":
            feature_spec.append({"name": column, "type": "double"})
        else:
            feature_spec.append({"name": column, "type": "unknown"})
    return feature_spec


def get_train_test_data(
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the train and test data for the diabetes prediction model.
    This function loads the data from a CSV file and splits it into training and testing sets.

    :return: Tuple of (X_train, X_test, y_train, y_test)
    """
    # Load the dataset
    data = pd.read_csv(
        "/Users/manue/Downloads/diabetes-prediction-dataset/diabetes_prediction_dataset.csv"
    )
    target_column = "diabetes"
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test
