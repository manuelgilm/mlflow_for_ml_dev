from pathlib import Path
from typing import Union
import pandas as pd
from typing import Tuple
from typing import Dict
from typing import Any
from sklearn.model_selection import train_test_split


class SalesDataProcessor:
    def __init__(self, path, configs: Dict[str, Any]):
        """
        Initialize the SalesDataProcessor with the path to the data and the data itself.

        :param path: Path to the data file.
        """
        self.load_data(path)
        self.configs = configs

    def load_data(self, path: Union[str, Path]) -> None:
        """
        Load the data from the specified path.

        :return: DataFrame containing the loaded data.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        self.df = pd.read_csv(path)
        self.data = self.df.copy()

    def create_train_test_split(self, test_size: float = 0.2) -> Tuple:
        """
        Splits the DataFrame into a training and testing set.

        :param test_size: The proportion of the dataset to include in the test split.
        :return: Tuple containing the training and testing sets.
        """
        df = self.data.copy()
        numerical_features = self.configs["numerical_features"]
        categorical_features = self.configs["categorical_features"]
        target = self.configs["target"]

        # drop date
        df = df.drop(columns=["Date"])

        X = df[["Store"] + numerical_features + categorical_features]
        y = df[["Store"] + [target]]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test
