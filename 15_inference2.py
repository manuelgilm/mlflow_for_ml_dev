import mlflow
from mlflow.models import infer_signature
from mlflow_utils import get_mlflow_experiment


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd 

if __name__=="__main__":

    run_id = "aa04555aa84442668c9cd49dbbb59cc3"
        
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
    X = pd.DataFrame(X, columns=["feature_{}".format(i) for i in range(10)])
    y = pd.DataFrame(y, columns=["target"])

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
    print(X_test["feature_1"].dtype)
    print(X_test["feature_1"].head())
    X_test["feature_1"] = X_test["feature_1"].astype("int64")
    print(X_test["feature_1"].dtype)
    print(X_test["feature_1"].head())

    # load model
    model_uri = f'runs:/{run_id}/random_forest_classifier'
    rfc = mlflow.sklearn.load_model(model_uri=model_uri,)

    y_pred = rfc.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=["prediction"])

    print(y_pred.head())


       


