from examples.diabetes_prediction.core.data import get_train_test_data
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


# poetry run mlflow models build-docker --model-uri models:/Diabetes_Prediction_Model@production -n diabetes_prediction_model
def main() -> None:
    """
    Perform online inference using a REST API.
    """
    # payload, labels = get_payload(10)
    url = "http://127.0.0.1:5000/invocations"

    # print(payload)
    payload = {
        "dataframe_split": {
            "columns": [
                "gender",
                "age",
                "hypertension",
                "heart_disease",
                "smoking_history",
                "bmi",
                "HbA1c_level",
                "blood_glucose_level",
            ],
            "data": [
                ["Female", 13.0, 0, 0, "No Info", 20.82, 5.8, 126],
                ["Female", 3.0, 0, 0, "No Info", 21.0, 5.0, 145],
                ["Male", 63.0, 0, 0, "former", 25.32, 3.5, 200],
                ["Female", 2.0, 0, 0, "never", 17.43, 6.1, 126],
                ["Female", 33.0, 0, 0, "not current", 40.08, 6.2, 200],
                ["Female", 70.0, 0, 0, "never", 23.89, 6.5, 200],
                ["Female", 51.0, 0, 0, "current", 27.32, 5.0, 158],
                ["Female", 12.0, 0, 0, "No Info", 27.32, 4.8, 158],
                ["Female", 45.0, 0, 0, "No Info", 27.32, 6.2, 145],
                ["Female", 19.0, 0, 0, "former", 27.32, 6.2, 90],
            ],
        }
    }
    headers = {"Content-Type": "application/json"}
    response = httpx.post(url, data=json.dumps(payload), headers=headers)
    predictions = get_predictions_from_response(response)
    print(
        pd.DataFrame(
            {
                "predictions": predictions,
                # "labels": labels,
            }
        )
    )
