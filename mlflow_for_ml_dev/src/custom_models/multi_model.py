
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
class MultiModel(mlflow.pyfunc.PythonModel):
    
    def __init__(self, models: Dict[str, Any] = {}):
        self.models = models

    def fit_estimators(self, x: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Fit the models to the data

        :param x: The input data
        :param y: The target data
        """
        for model_id, model in self.models.items():
            print("Fitting model: ", model_id)
            model.fit(x, y)

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
        model_id = params["model_id"]

        if model_id not in self.models.keys():
            raise ValueError(f"Model with id {model_id} not found")
        
        model = self.models[model_id]
        print("Predicting with model: ", model_id)
        prediction = model.predict(model_input)

        return {model_id: prediction}