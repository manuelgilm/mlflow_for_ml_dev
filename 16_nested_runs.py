import mlflow
from mlflow_utils import create_mlflow_experiment

experiment_id = create_mlflow_experiment(
    experiment_name= "Nested Runs",
    artifact_location= "nested_run_artifacts",
    tags={"purpose":"learning"}
)


with mlflow.start_run(run_name="parent") as parent:
    print("RUN ID parent:", parent.info.run_id)

    mlflow.log_param("parent_param", "parent_value")

    with mlflow.start_run(run_name="child1",nested=True) as child1:
        print("RUN ID child1:", child1.info.run_id)
        mlflow.log_param("child1_param", "child1_value")

        with mlflow.start_run(run_name="child_11", nested=True) as child_11:
            print("RUN ID child_11:", child_11.info.run_id )
            mlflow.log_param("child_11_param", "child_11_value")

        with mlflow.start_run(run_name="child_12", nested=True) as child_12:
            print("RUN ID child_12:", child_12.info.run_id)
            mlflow.log_param("child_12_param", "child_12_value")

    with mlflow.start_run(run_name="child2", nested=True) as child2:
        print("RUN ID child2:", child2.info.run_id)
        mlflow.log_param("child2_param", "child2_value")