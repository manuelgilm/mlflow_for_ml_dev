# import necessary libraries
from examples.iris_classifier.data import get_train_test_data
import httpx
import json
import pandas as pd
import sys 

def mlflow_endpoints():
    """
    Showcase MLflow model serving endpoints.

    The inference server provides 4 endpoints:

    /invocations: An inference endpoint that accepts POST requests with input data and returns predictions.

    /ping: Used for health checks.

    /health: Same as /ping

    /version: Returns the MLflow version.
    """

    base_url = "http://127.0.0.1:5000"

    endpoints = {
        "invocations": f"{base_url}/invocations", # post request
        "ping": f"{base_url}/ping", # get request
        "health": f"{base_url}/health", # get request
        "version": f"{base_url}/version", # get request
    }

    # get parameter from cli
    endpoint = sys.argv[1] if len(sys.argv) > 1 else "version"
    if endpoint not in endpoints:
        print(f"Invalid endpoint: {endpoint}")
        sys.exit(1)
    if endpoint == "invocations":
        print("Use the main function in `online_inference.py` script to call the /invocations endpoint")
        sys.exit(0)
        
    print(f"Calling MLflow endpoint: {endpoint}")
    response = httpx.get(endpoints[endpoint])
    print(f"Response status code: {response.status_code}")
    print(response.text)

def invocation_csv():
    """
    This function uses the CSV format for online inference.
    """
    # payload, headers, labels = get_request_body("text/csv")
    # Make the REST API call to perform online inference
    _, x_test, _, y_test = get_train_test_data()
    samples = 3  # Number of samples to include in the payload

    payload = x_test.iloc[0:samples].to_csv(index=False)

    # show payload example
    print(payload)
    # specify content-type as "text/csv"
    headers = {"Content-Type": "text/csv"}

    url = "http://127.0.0.1:5000/invocations"

    response = httpx.post(url, data=payload, headers=headers)
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

def invocation_json():
    """
    This function uses the JSON format for online inference.
    """
    _, x_test, _, y_test = get_train_test_data()
    samples = 3  # Number of samples to include in the payload

    payload = json.dumps({
        "dataframe_split": x_test.iloc[0:samples].to_dict(orient="split"),
    })
    headers = {"Content-Type": "application/json"}

    url = "http://127.0.0.1:5000/invocations"

    response = httpx.post(url, data=payload, headers=headers)
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


