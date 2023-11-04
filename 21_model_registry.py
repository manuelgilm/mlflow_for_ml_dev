import mlflow
from sklearn.ensemble import RandomForestRegressor
from mlflow_utils import create_mlflow_experiment

class CustomModel(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input):
        return model_input


if __name__ == "__main__":
    experiment_id = create_mlflow_experiment(
        experiment_name="model_registry",
        artifact_location="model_registry_artifacts",
        tags={"purpose": "learning"},
    )

    with mlflow.start_run(run_name="model_registry") as run:
        model = CustomModel()
        mlflow.pyfunc.log_model(artifact_path="custom_model", python_model=model, registered_model_name="CustomModel")
        mlflow.sklearn.log_model(artifact_path="rfr_model", sk_model=RandomForestRegressor(), registered_model_name="RandomForestRegressor")
        mlflow.sklearn.log_model(artifact_path="rft_model2", sk_model=RandomForestRegressor())
            
