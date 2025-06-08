from examples.walmart_sales_regression.data import SalesDataProcessor
from examples.utils.file_utils import get_root_dir
from examples.utils.file_utils import read_file
import pandas as pd
import httpx


def main():
    """
    Perform online inference using a REST API.

    To deploy the model using the local server, run the following command:
    `poetry run mlflow models serve -m models:/walmart-store-sales-regressor-code@production -p 5000 --env-manager local`
    """

    url = "http://localhost:5000/invocations"
    root_dir = get_root_dir()
    configs = read_file(
        root_dir / "examples" / "walmart_sales_regression" / "configs.yaml"
    )

    data_path = (
        root_dir.parents[1] / "Downloads" / "sales-walmart" / "Walmart_Sales.csv"
    )  # change this to your data path

    data_processor = SalesDataProcessor(path=data_path, configs=configs)
    _, x_test, _, y_test = data_processor.create_train_test_split()

    # predicting for the store store_id
    store_id = 1
    x_test = x_test[x_test["Store"] == store_id]
    y_test = y_test[y_test["Store"] == store_id]

    payload = {
        "dataframe_split": x_test.to_dict(orient="split"),
        "params": {"store_id": str(store_id)},
    }
    headers = {"Content-Type": "application/json"}
    response = httpx.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        predictions = response.json().get("predictions")
        weekly_sales_pred = predictions
        weekly_sales = y_test[configs["target"]].values

        print(
            pd.DataFrame(
                {
                    "predictions": weekly_sales_pred,
                    "y_test": weekly_sales,
                    "difference": abs(weekly_sales_pred - weekly_sales),
                }
            ).head()
        )
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")
