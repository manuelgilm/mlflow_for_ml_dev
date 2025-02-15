from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from typing import Union
import pandas as pd
import numpy as np
from typing import Union
from typing import Tuple
from typing import List
from typing import Dict
from typing import Any

from scipy.spatial import distance
import mlflow


class CustomClassifier(mlflow.pyfunc.PythonModel):
    def __init__(self, x_ref_path: str = None, run_id: str = None):
        self.model = RandomForestClassifier()
        self.x_ref_path = x_ref_path
        self.run_id = run_id
        self.threshold = 0.1
        self.result = {
            "method": "js_distance",
            "features": {},
            "threshold": self.threshold,
        }

    def fit_estimator(
        self, x: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> None:
        """
        Fit the model to the data

        :param x: The input data
        :param y: The target data

        """
        self.model.fit(x, y)

    def log_drift_detection(self, drift_results):
        """
        Log the drift detection results

        :param drift_results: The drift detection results
        """
        client = mlflow.MlflowClient()
        with mlflow.start_run(run_id=self.run_id):
            for feature in drift_results["features"]:
                metric_name = feature + "_js_distance"
                step = len(
                    client.get_metric_history(run_id=self.run_id, key=metric_name)
                )
                mlflow.log_metric(
                    metric_name, drift_results["features"][feature][0], step=step
                )

    def score(
        self, x_ref: pd.DataFrame, x_new: pd.DataFrame, feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Score the distance between two datasets.

        :param x_ref: reference dataset
        :param x_new: new dataset
        :param feature_names: list of feature names
        :return: dictionary containing the drift result
        """

        for feature in feature_names:
            hist1, hist2 = self.calculate_histogram(x_ref[feature], x_new[feature])
            js_distance = self.calculate_js_distance(hist1, hist2)
            is_drift = js_distance > self.threshold
            self.result["features"][feature] = (js_distance, is_drift)

        return self.result

    def calculate_histogram(
        self,
        s1: Union[np.ndarray, pd.Series],
        s2: Union[np.ndarray, pd.Series],
        bins: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the histogram of the data.

        :param s1: reference raw data
        :param s2: new raw data
        :return: histogram of the data
        """

        global_min = min(min(s1), min(s2))
        global_max = max(max(s1), max(s2))

        hist1, _ = np.histogram(s1, bins=bins, range=(global_min, global_max))
        hist2, _ = np.histogram(s2, bins=bins, range=(global_min, global_max))

        return hist1, hist2

    def calculate_js_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate the Jensen-Shannon distance between two probability distributions.

        :param p: probability distribution
        :param q: probability distribution
        :return: Jensen-Shannon distance
        """

        js_stat = distance.jensenshannon(p, q, base=2)
        js_stat = np.round(js_stat, 4)
        return js_stat

    def predict(
        self, context, model_input: Union[pd.DataFrame, np.ndarray], params={}
    ) -> np.ndarray:
        """
        Predict the target values

        :param context: The context object
        :param model_input: The input data
        :param params: The model parameters
        :return: The predicted target values
        """
        df = pd.read_csv(context.artifacts["x_ref_path"])
        feature_names = model_input.columns.tolist()
        drift_results = self.score(
            x_ref=df[feature_names], x_new=model_input, feature_names=feature_names
        )
        self.log_drift_detection(drift_results)
        return self.model.predict(model_input)

