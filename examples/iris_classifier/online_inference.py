# import necessary libraries
from examples.iris_classifier.data import get_train_test_data
import httpx
import json
import pandas as pd


def main() -> None:
    """
    Perform online inference using a REST API.

    To deploy the model in the local server, run the following command:
    `poetry run mlflow models serve -m models:/Iris_Classifier_Model@production --env-manager local`
    """
    # Load the test data
    _, x_test, _, y_test = get_train_test_data()
    samples = 3  # Number of samples to include in the payload

    # Uncomment the following line to make the api call fail
    # x_test["sepal length (cm)"] = ["" for _ in range(len(x_test))]

    # define the payload for online inference
    payload = {
        "dataframe_split": x_test.iloc[0:samples].to_dict(orient="split"),
    }

    # Define the endpoint URL and headers for the REST API call
    url = "http://127.0.0.1:5000/invocations"
    headers = {"Content-Type": "application/json"}

    # Make the REST API call to perform online inference
    response = httpx.post(url, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        json_response = response.json()
        predictions = json_response.get("predictions")
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

    # Print the predictions and corresponding labels
    print(
        pd.DataFrame(
            {
                "predictions": predictions,
                "labels": y_test.iloc[0:samples].values,
            }
        )
    )
