import mlflow
from mlflow_utils import get_mlflow_experiment

if __name__=="__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")

    print("Name: {}".format(experiment.name))

    with mlflow.start_run(run_name="logging_artifacts", experiment_id=experiment.experiment_id) as run:

        # your machine learning code goes here

        mlflow.log_artifacts(local_dir="./run_artifacts",artifact_path="run_artifacts")

        # print run info
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))   