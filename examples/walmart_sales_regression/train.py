from examples.walmart_sales_regression.data import SalesDataProcessor
from examples.walmart_sales_regression.base import WalmartSalesRegressor
from examples.utils.file_utils import get_root_dir
from examples.utils.decorators import mlflow_tracking_uri
from examples.utils.decorators import mlflow_client
from examples.utils.decorators import mlflow_experiment

import mlflow


@mlflow_tracking_uri
@mlflow_experiment(name="walmart_sales_regression")
@mlflow_client
def main(**kwargs):
    """
    Train the Walmart sales regression model.
    """
    registered_model_name = "walmart-store-sales-regressor"
    root_dir = get_root_dir()
    data_path = (
        root_dir.parents[1] / "Downloads" / "sales-walmart" / "Walmart_Sales.csv"
    )  # change this to your data path

    # data_path = "../../Downloads/sales-walmart/Walmart_Sales.csv"
    data_processor = SalesDataProcessor(path=data_path)
    x_train, x_test, y_train, y_test = data_processor.create_train_test_split()
    print("Data loaded and split into training and testing sets.")

    # train model for three stores
    x_train = x_train[x_train["Store"].isin([1, 2, 3])]
    y_train = y_train[y_train["Store"].isin([1, 2, 3])]
    x_test = x_test[x_test["Store"].isin([1, 2, 3])]
    y_test = y_test[y_test["Store"].isin([1, 2, 3])]

    store_sales_regressor = WalmartSalesRegressor()

    with mlflow.start_run(run_name="walmart-sales-regressors") as run:

        for store_id in x_train["Store"].unique():
            store_sales_regressor.fit_model(
                x_train=x_train,
                y_train=y_train,
                store_id=store_id,
                run_id=run.info.run_id,
            )

        # Log the entire class as a model
        signature = store_sales_regressor._get_model_signature()
        mlflow.pyfunc.log_model(
            artifact_path="store-sales-regressor",
            python_model=store_sales_regressor,
            registered_model_name=registered_model_name,
            signature=signature,
        )

        print("Models fitted successfully.")

        # Set the model version alias to "production"
        model_version = mlflow.search_model_versions(
            filter_string=f"name='{registered_model_name}'",
            max_results=1,
        )[0]
        client = kwargs["mlflow_client"]
        print(f"Model version: {model_version.version}")
        print(f"Model name: {model_version.name}")
        client.set_registered_model_alias(
            name=registered_model_name,
            version=model_version.version,
            alias="production",
        )
