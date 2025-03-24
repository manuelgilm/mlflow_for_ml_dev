import mlflow 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class MultiClassifier(mlflow.pyfunc.PythonModel):
    

    def __init__(self):
        algos = ["random_forest", "decision_tree"]
        self.models = {algo: None for algo in algos}

    def train_estimators(self, x, y):
        """
        Train the models

        :param x: The input data
        :param y: The target data

        """
        for algo in self.models.keys():
            model = self._get_model(algo)
            model.fit(x, y)
            self.models[algo] = model

    def _get_model(self, algo: str):
        """
        Get the model object
        """
        if algo == "random_forest":
            return RandomForestClassifier()
        elif algo == "decision_tree":
            return DecisionTreeClassifier()
        else:
            raise ValueError(f"Model {algo} not found")
        
    def predict(self, context, model_input, params={}):
        """
        Predict the target values

        :param context: The context object
        :param model_input: The input data
        :param params: The model parameters
        :return: The predicted target values
        """
        if params:
            algo = params.get("algo", "random_forest")
        else:
            algo = "random_forest"
        print("Predicting with model: ", algo)
        model = self.models[algo]
        return model.predict(model_input)

