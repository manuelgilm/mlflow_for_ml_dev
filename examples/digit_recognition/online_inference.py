from examples.digit_recognition.data import get_train_val_test_data
from examples.digit_recognition.data import transform_to_image
import httpx
import pandas as pd
import json
import numpy as np


def main() -> None:
    """
    Perform online inference using a REST API.
    To deploy the model in the local server, run the following command:

    `poetry run mlflow models serve -m models:/Digit_Recognition_Model@production --env-manager local`
    """
    _, _, x_test, _, _, y_test = get_train_val_test_data()
    x_test = transform_to_image(x_test)

    url = "http://127.0.0.1:5001/invocations"
    n_samples = 5
    samples = x_test[0:n_samples]

    payload = {
        "instances": {"image_input": samples.tolist()},
    }
    # print(payload)
    headers = {"Content-Type": "application/json"}
    response = httpx.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        predictions = response.json().get("predictions")
        pred = np.argmax(predictions, axis=-1)
        print(pd.DataFrame({"predictions": pred, "y_test": y_test[0:n_samples]}))
        return
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")
