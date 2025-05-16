from usecases.iris_classifier.data import get_train_test_data
import httpx
import json
import pandas as pd


def get_predictions_from_response(response):
    """
    Process the response from the REST API.

    :param response: The response object from the HTTP request.
    :return: The JSON content of the response.
    """
    if response.status_code == 200:
        json_response = response.json()
        predictions = json_response.get("predictions")
        if predictions is not None:
            return predictions
        else:
            raise Exception("No predictions found in the response.")
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


def get_payload(samples: int) -> dict:
    """
    Get the payload for online inference.

    :param samples: Number of samples to include in the payload.
    :return: Dictionary containing the payload for online inference.
    """
    _, x_test, _, y_test = get_train_test_data()
    payload = {
        "dataframe_split": x_test.iloc[0:samples].to_dict(orient="split"),
    }
    return payload, y_test.iloc[0:samples]


def main() -> None:
    """
    Perform online inference using a REST API.

    To deploy the model in the local server, run the following command:
    `poetry run mlflow models serve -m models:/Iris_Classifier_Model@production --no-conda`

    """
    payload, labels = get_payload(10)
    url = "http://127.0.0.1:5000/invocations"
    response = httpx.get(url)
    print(response.status_code)
    print(response.text)

    print(payload)
    headers = {"Content-Type": "application/json"}
    response = httpx.post(url, data=json.dumps(payload), headers=headers)
    predictions = get_predictions_from_response(response)
    print(
        pd.DataFrame(
            {
                "predictions": predictions,
                "labels": labels,
            }
        )
    )
