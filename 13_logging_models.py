import mlflow
from mlflow.models import infer_signature
from mlflow_utils import get_mlflow_experiment


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd 

if __name__=="__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print("Name: {}".format(experiment.name))

    with mlflow.start_run(run_name="logging_models", experiment_id=experiment.experiment_id) as run:

        
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

        X_train = pd.DataFrame(X_train, columns=["feature_{}".format(i) for i in range(10)])
        X_test = pd.DataFrame(X_test, columns=["feature_{}".format(i) for i in range(10)])
        y_train = pd.DataFrame(y_train, columns=["target"])
        y_test = pd.DataFrame(y_test, columns=["target"])

        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)
        y_pred = pd.DataFrame(y_pred, columns=["prediction"])

        # infer signature
        model_signature = infer_signature(model_input=X_train, model_output=y_pred)


        # log model 
        mlflow.sklearn.log_model(sk_model=rfc, artifact_path="random_forest_classifier", signature=model_signature)
        
        
        # print info about the run
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))




