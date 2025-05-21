import mlflow
import pandas as pd
from examples.utils.file_utils import get_root_dir
from examples.walmart_sales_regression.data import SalesDataProcessor


def main():
    """
    Perform inference using the Walmart sales regression model.
    """
    root_dir = get_root_dir()
    data_path = (
        root_dir.parents[1] / "Downloads" / "sales-walmart" / "Walmart_Sales.csv"
    )  # change this to your data path

    data_processor = SalesDataProcessor(path=data_path)
    _, x_test, _, y_test = data_processor.create_train_test_split()

    # load model
    model_uri = "models:/walmart-store-sales-regressor@production"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    print("Model loaded.")

    # predicting for the store store_id
    store_id = 3
    x_test = x_test[x_test["Store"] == store_id]
    y_test = y_test[y_test["Store"] == store_id]
    x_test = x_test.drop(columns=["Store"])
    y_test = y_test.drop(columns=["Store"])

    # make predictions
    predictions = model.predict(x_test, params={"store_id": store_id})

    weekly_sales = y_test["Weekly_Sales"].values
    print(
        pd.DataFrame(
            {
                "predictions": predictions,
                "y_test": weekly_sales,
                "difference": abs(predictions - weekly_sales),
            }
        ).head()
    )
