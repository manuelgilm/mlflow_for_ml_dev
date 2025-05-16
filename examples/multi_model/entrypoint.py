from examples.multi_model.pipeline import MultiClassifier
from examples.multi_model.utils import get_model_signature
from examples.multi_model.utils import get_train_test_data
from examples.multi_model.utils import set_alias
from examples.utils.mlflow_utils import get_or_create_experiment
from mlflow_for_ml_dev.src.utils.folder_operations import get_project_root
import mlflow
import sys


def train_multi_model():
    """
    Train the multi model
    """
    try:
        algo = sys.argv[1]
    except IndexError:
        # default value
        algo = "random_forest"

    x_train, x_test, y_train, y_test = get_train_test_data()

    multi_classifier = MultiClassifier()
    multi_classifier.train_estimators(x_train, y_train)

    # set tracking uri
    mlflow.set_tracking_uri(get_project_root() / "mlruns")

    # set experiment
    experiment_name = "multi_model"
    experiment = get_or_create_experiment(
        name=experiment_name,
        tags={"project_name": "multi_model"},
    )

    # set the experiment description
    mlflow.set_experiment_tag(
        key="mlflow.note.content",
        value="Experiment to test the multi model implementation with sklearn",
    )

    signature = get_model_signature(x_train, y_train, {"algo": algo})

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.set_tag("model_type", "multi_model")
        mlflow.set_tag("mlflow.note.content", "Custom model trained with random data")

        registered_model_name = "multi_model"
        mlflow.pyfunc.log_model(
            artifact_path="multi_model",
            python_model=multi_classifier,
            signature=signature,
            input_example=x_train.sample(5),
            registered_model_name=registered_model_name,
            model_config={"algo": algo},
        )

        model_uri = f"runs:/{run.info.run_id}/multi_model"
        eval_data = x_test.copy()
        eval_data["target"] = y_test

        eval_results = mlflow.evaluate(
            model=model_uri,
            data=eval_data,
            targets="target",
            model_type="classifier",
            evaluator_config={"metric_prefix": "eval_"},
            model_config={"algo": algo},
        )

    set_alias(model_name=registered_model_name)


def inference_multimodel():
    """
    Inference the multi model.
    """
    try:
        algo = sys.argv[1]
    except IndexError:
        # default value
        algo = "random_forest"

    _, x_test, _, _ = get_train_test_data()

    registered_model_name = "multi_model"

    # get the champion model
    model_uri = f"models:/{registered_model_name}@champion"
    model = mlflow.pyfunc.load_model(model_uri, model_config={"algo": None})

    predictions = model.predict(x_test)
    print(predictions)
