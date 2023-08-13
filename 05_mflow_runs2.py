import mlflow 

if __name__=="__main__":

    with mlflow.start_run(run_name="mlflow_runs") as run:

        # Your machine learning code goes here
        mlflow.log_param("learning_rate",0.01)
        print("RUN ID")
        print(run.info.run_id)

        print(run.info)