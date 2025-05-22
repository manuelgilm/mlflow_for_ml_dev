from examples.diabetes_prediction.core.data import get_train_test_data
import pandas as pd
import mlflow


def main(**kwargs) -> None:

    _, x_test, _, y_test = get_train_test_data()
    # Load the model from MLflow
    registered_model_name = "Diabetes_Prediction_Model"
    model_uri = f"models:/{registered_model_name}@production"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    predictions_with_probs = model.predict(x_test, params={"probabilities": True})
    print(predictions_with_probs.head())
