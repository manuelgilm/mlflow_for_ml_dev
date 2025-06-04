import mlflow
import pandas as pd
from examples.utils.file_utils import get_root_dir
from examples.utils.file_utils import read_file
from examples.walmart_sales_regression.data import SalesDataProcessor


def main():
    """
    Perform inference using the Walmart sales regression model.
    """
    root_dir = get_root_dir()

    configs = read_file(
        root_dir / "examples" / "walmart_sales_regression" / "configs.yaml"
    )
    registered_model_name = configs["registered_model_name"]
    data_path = (
        root_dir.parents[1] / "Downloads" / "sales-walmart" / "Walmart_Sales.csv"
    )  # change this to your data path

    data_processor = SalesDataProcessor(path=data_path, configs=configs)
    _, x_test, _, y_test = data_processor.create_train_test_split()

    # load model
    model_uri = f"models:/{registered_model_name}@production"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    print("Model loaded.")

    # # predicting for the store store_id
    store_id = "2"
    x_test = x_test[x_test["Store"] == int(store_id)]
    y_test = y_test[y_test["Store"] == int(store_id)]
    x_test = x_test.drop(columns=["Store"])
    y_test = y_test.drop(columns=["Store"])

    # make predictions
    predictions = model.predict(x_test, params={"store_id": store_id})

    weekly_sales = y_test[configs["target"]].values
    print(
        pd.DataFrame(
            {
                "predictions": predictions,
                "y_test": weekly_sales,
                "difference": abs(predictions - weekly_sales),
            }
        ).head()
    )
