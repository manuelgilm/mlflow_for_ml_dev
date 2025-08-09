from examples.utils.decorators import mlflow_tracking_uri
import mlflow


class ModelLoader:

    # To validate mlflow aliases
    __ALLOWED_ENVIRONMENTS = ["Production", "Staging", "Development"]
    __ML_MODELS = {}

    @mlflow_tracking_uri
    def load_model(self, model_name: str, environment: str = "Production") -> None:
        """
        Load a model from MLflow. Use this function to load a model
        at the before starting to serve the endpoints.

        :param model_name: Name of the model to load.
        :param environment: Environment from which to load the model.
        """
        if environment not in self.__ALLOWED_ENVIRONMENTS:
            raise ValueError(
                f"Invalid environment: {environment}. Allowed values are: {self.__ALLOWED_ENVIRONMENTS}"
            )

        registered_models = mlflow.MlflowClient().get_model_version_by_alias(
            name=model_name, alias=environment
        )

        if not registered_models:
            raise ValueError(
                f"No registered model found for name: {model_name} and environment: {environment}"
            )

        model_uri = f"models:/{model_name}@{environment}"
        model = mlflow.sklearn.load_model(model_uri=model_uri)
        self.__ML_MODELS[model_name] = model

    @classmethod
    def get_model(cls, model_name: str):
        """
        Get a loaded model by name.
        :param model_name: Name of the model to retrieve.
        :return: The loaded model if found, None otherwise.
        """
        return cls.__ML_MODELS.get(model_name)

    @classmethod
    def clear_models(cls):
        """
        Clear all loaded models.
        """
        cls.__ML_MODELS.clear()
