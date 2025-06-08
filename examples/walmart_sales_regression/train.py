from examples.walmart_sales_regression.data import SalesDataProcessor
from examples.walmart_sales_regression.base import WalmartSalesRegressor
from examples.utils.file_utils import get_root_dir
from examples.utils.file_utils import read_file
from examples.utils.decorators import mlflow_tracking_uri
from examples.utils.decorators import mlflow_client
from examples.utils.decorators import mlflow_experiment
from examples.utils.mlflow_utils import set_alias_to_latest_version

import mlflow


@mlflow_tracking_uri
@mlflow_experiment(name="walmart_sales_regression")
@mlflow_client
def main(**kwargs):
    """
    Train the Walmart sales regression model.
    """
    root = get_root_dir()
    configs = read_file(root / "examples" / "walmart_sales_regression" / "configs.yaml")

    registered_model_name = configs["registered_model_name"]
    root_dir = get_root_dir()
    data_path = (
        root_dir.parents[1] / "Downloads" / "sales-walmart" / "Walmart_Sales.csv"
    )  # change this to your data path

    data_processor = SalesDataProcessor(path=data_path, configs=configs)
    x_train, x_test, y_train, y_test = data_processor.create_train_test_split()
    print("Data loaded and split into training and testing sets.")

    # train model for three stores
    x_train = x_train[x_train["Store"].isin([1, 2, 3])]
    y_train = y_train[y_train["Store"].isin([1, 2, 3])]
    x_test = x_test[x_test["Store"].isin([1, 2, 3])]
    y_test = y_test[y_test["Store"].isin([1, 2, 3])]

    store_sales_regressor = WalmartSalesRegressor(config=configs)

    with mlflow.start_run(run_name=configs["run_name"]) as run:

        for store_id in x_train["Store"].unique():
            store_sales_regressor.fit_model(
                x_train=x_train,
                y_train=y_train,
                store_id=store_id,
                run_id=run.info.run_id,
            )

        # Log the entire class as a model
        signature = store_sales_regressor._get_model_signature()
        # log model without code
        mlflow.pyfunc.log_model(
            artifact_path=configs["artifact_path"],
            python_model=store_sales_regressor,
            registered_model_name=registered_model_name,
            input_example=x_test.sample(5),
            signature=signature,
            artifacts=store_sales_regressor.artifact_uris,
        )

        set_alias_to_latest_version(
            registered_model_name=registered_model_name,
            alias="production",
            client=kwargs["mlflow_client"],
        )
