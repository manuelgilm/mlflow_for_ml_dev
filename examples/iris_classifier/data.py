from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Optional
from typing import Tuple
import pandas as pd


def get_train_test_data(
    test_size: Optional[float] = 0.2, random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the iris dataset and split it into training and testing sets.
    The function returns the training and testing data as pandas DataFrames.

    :param test_size: The proportion of the dataset to include in the test split (default is 0.2).
    :param random_state: Controls the shuffling applied to the data before applying the split (default is 42).
    :return: A tuple containing the training features (X_train), testing features (X_test),
             training labels (y_train), and testing labels (y_test).
    """
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
