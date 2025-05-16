import mlflow
import numpy as np
from utils.decorators import mlflow_tracking_uri
from examples.digit_recognition.data import get_train_test_data
from examples.digit_recognition.data import transform_to_image
import pandas as pd


# command to run the model server
# poetry run mlflow models serve -m models:/Digit_Recognition_Model@production --no-conda
@mlflow_tracking_uri
def main():
    """
    Main function to run the batch inference process.
    """
    # Load the model from the specified path
    _, x_test, _, y_test = get_train_test_data()
    x_test = transform_to_image(x_test)
    registered_model_name = "Digit_Recognition_Model"
    model_path = f"models:/{registered_model_name}@production"
    model = mlflow.pyfunc.load_model(model_path)
    pred = model.predict({"image_input": x_test})
    # Perform inference on the test data
    predictions = np.argmax(pred, axis=-1)
    print(predictions)

    predictions_df = pd.DataFrame({"predictions": predictions, "y_test": y_test})
    print(predictions_df.head())
