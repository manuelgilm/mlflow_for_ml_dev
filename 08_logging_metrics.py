import mlflow 
from mlflow_utils import get_mlflow_experiment

if __name__=="__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print("Name: {}".format(experiment.name))

    with mlflow.start_run(run_name="logging_metrics", experiment_id = experiment.experiment_id) as run:
        # Your machine learning code goes here

        mlflow.log_metric("random_metric", 0.01)

        metrics = {
            "mse": 0.01,
            "mae": 0.01,
            "rmse": 0.01,
            "r2": 0.01
        }

        mlflow.log_metrics(metrics)

        # print run info
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))


