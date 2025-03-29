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

    def predict(self, context, model_input, params):
        """
        Predict the target values

        :param context: The context object
        :param model_input: The input data
        :param params: The model parameters
        :return: The predicted target values
        """       
        if self.algo is None:
            print("Algo not found in context")
            self.algo = params.get("algo", None)
        if self.algo is None:
            print("Algo not found in params. Using default algo")
            # default value
            self.algo = "random_forest"

        print("Predicting with model: ", self.algo)
        model = self.models[self.algo]
        return model.predict(model_input)
    
    def load_context(self, context):
        """
        Load the context
        """
        print(context)
        self.algo = context.model_config.get("algo", None)
        print(self.algo)
        if self.algo is None:
            print("Algo not found in context")
    
    
