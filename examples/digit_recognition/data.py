import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from examples.utils.file_utils import get_root_dir


def get_train_val_test_data(
    test_size: Optional[float] = 0.1, random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get the training and testing data for the handwritten digit recognition dataset.

    :param test_size: The proportion of the dataset to include in the test split (default is 0.2).
    :param random_state: Controls the shuffling applied to the data before applying the split (default is 42).
    :return: A tuple containing the training features (X_train), testing features (X_test),
                training labels (y_train), and testing labels (y_test).
    """

    root = get_root_dir()
    train_path = root.parent / "handwritten_digit_recognition_dataset" / "train.csv"
    df = pd.read_csv(train_path)
    # Split the dataset into training and testing sets
    X = df.drop(columns=["label"])
    y = df["label"]

    # First split into train and validation+test
    X_train, X_val_, y_train, y_val_ = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Then split the validation+test set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_, y_val_, test_size=test_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def transform_to_image(x: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input DataFrame into a 3D array representing images.

    :param x: The input DataFrame containing pixel values.
    :return: A DataFrame with the transformed image data.
    """
    x = x.values.reshape(-1, 28, 28, 1)
    return x.astype("float32")
