from examples.utils.file_utils import get_root_dir
from examples.utils.file_utils import read_file
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple
import pandas as pd


def create_training_and_testing_dataset(
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create training and testing datasets.

    :param test_size: The proportion of the dataset to include in the test split.
    :return dataset: The dataset.
    """

    # Get the root directory.
    root_dir = get_root_dir()

    # data path
    data_path = Path(
        root_dir, "student_performance", "data", "raw_data", "student_performance.csv"
    )
    dataset = read_file(path=data_path)
    # exclude StudentID column
    dataset = dataset.drop(columns=["StudentID"])
    # split the dataset into training and testing datasets
    target = "GradeClass"
    x = dataset.drop(columns=[target])
    y = dataset[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )
    return x_train, x_test, y_train, y_test


def get_numerical_features():
    """
    Get numerical features from a DataFrame.
    """
    return ["Age", "StudyTimeWeekly", "Absences", "GPA"]


def get_categorical_features():
    """
    Get categorical features from a DataFrame.
    """
    return [
        "Gender",
        "Ethnicity",
        "ParentalEducation",
        "Tutoring",
        "ParentalSupport",
        "Extracurricular",
        "Sports",
        "Music",
        "Volunteering",
    ]
