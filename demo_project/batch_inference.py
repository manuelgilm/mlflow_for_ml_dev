from package.feature.data_processing import get_feature_dataframe
from package.ml_training.retrieval import get_train_test_score_set
from sklearn.metrics import classification_report
import pandas as pd

import mlflow 


if __name__=="__main__":


    df = get_feature_dataframe()
    x_train, x_test, x_score, y_train, y_test, y_score = get_train_test_score_set(df)
    features = [f for f in x_train.columns if f not in ["id", "target", "MedHouseVal"]]

    model_uri = "models:/registered_model/latest"
    mlflow_model = mlflow.sklearn.load_model(model_uri=model_uri)

    predictions = mlflow_model.predict(x_score[features])

    scored_data = pd.DataFrame({"prediction": predictions, "target": y_score})

    classification_report = classification_report(y_score, predictions)
    print(classification_report)
    print(scored_data.head(10))