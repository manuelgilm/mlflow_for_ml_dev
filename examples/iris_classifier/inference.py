# import necessary libraries
import mlflow
from examples.utils.decorators import mlflow_tracking_uri
from examples.iris_classifier.data import get_train_test_data


@mlflow_tracking_uri
def main():
    """
    Main function to run the batch inference process.
    """
    # Load the test data
    _, x_test, _, _ = get_train_test_data()

    # Load the model from the specified path
    registered_model_name = "Iris_Classifier_Model"
    model_path = f"models:/{registered_model_name}@production"
    model = mlflow.sklearn.load_model(model_path)

    # Perform inference on the test data
    predictions = model.predict(x_test)
    x_test["predictions"] = predictions
    print(x_test.head())
