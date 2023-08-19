import mlflow
from mlflow_utils import create_mlflow_experiment

experiment_id = create_mlflow_experiment(
    experiment_name= "Nested Runs",
    artifact_location= "nested_run_artifacts",
    tags={"purpose":"learning"}
)

with mlflow.start_run(run_name="parent") as run:
    mlflow.set_tag("level","first")

    with mlflow.start_run(run_name="child_1", nested=True) as child1:
    
        mlflow.set_tag("level","second")
