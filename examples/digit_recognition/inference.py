import os

os.environ["KERAS_BACKEND"] = "torch"

import mlflow
import numpy as np
from examples.utils.decorators import mlflow_tracking_uri
from examples.digit_recognition.data import get_train_val_test_data
from examples.digit_recognition.data import transform_to_image
from examples.utils.file_utils import get_root_dir
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


@mlflow_tracking_uri
def main():
    """
    Main function to run the batch inference process.

    To deploy the model in the local server, run the following command:
    `poetry run mlflow models serve -m models:/Digit_Recognition_Model@production --env-manager local`
    """
    # Load the model from the specified path
    _, _, x_test, _, _, y_test = get_train_val_test_data()
    x_test = transform_to_image(x_test)

    registered_model_name = "Digit_Recognition_Model"
    model_path = f"models:/{registered_model_name}@production"
    model = mlflow.keras.load_model(model_path)
    pred = model.predict(x_test)
    # Perform inference on the test data
    predictions = np.argmax(pred, axis=-1)
    print(predictions)
    evaluation_report = classification_report(y_test, predictions)

    # Print evaluation report
    print("Evaluation Report:\n", evaluation_report)

    # Display confusion matrix
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    cm_display.ax_.set_title("Confusion Matrix")
    cm_display.figure_.suptitle("Confusion Matrix for Digit Recognition Model")
    cm_display.figure_.savefig(
        get_root_dir() / "examples" / "digit_recognition" / "confusion_matrix.png"
    )
