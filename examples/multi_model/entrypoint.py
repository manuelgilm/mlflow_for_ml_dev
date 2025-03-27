from examples.multi_model.pipeline import MultiClassifier
from examples.multi_model.utils import get_model_signature
from examples.multi_model.utils import get_train_test_data
from examples.multi_model.utils import set_alias
from examples.utils.mlflow_utils import get_or_create_experiment
from mlflow_for_ml_dev.src.utils.folder_operations import get_project_root
from sklearn.metrics import classification_report
import mlflow

def train_multi_model():
    """
    Train the multi model
    """
    x_train, x_test, y_train, y_test = get_train_test_data()

    multi_classifier = MultiClassifier()
    multi_classifier.train_estimators(x_train, y_train)

    # get predictions
    predictions = multi_classifier.predict(None, x_test, params={"algo": "decision_tree"})
    

    #set tracking uri
    mlflow.set_tracking_uri(get_project_root() / "mlruns")

    # set experiment
    experiment_name = "multi_model"
    experiment = get_or_create_experiment(
        name=experiment_name,
        tags={"project_name": "student_performance"},
    )

    # set the experiment description
    mlflow.set_experiment_tag(
        key="mlflow.note.content",
        value="Experiment to test the random forest classifier on the student performance dataset.",
    )
    
    signature = get_model_signature(x_train, y_train, {"algo": "random_forest"})

    metrics_repot = classification_report(y_test, predictions, output_dict=True)

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.set_tag("model_type", "multi_model")
        mlflow.set_tag("mlflow.note.content", "Custom model trained with random data")
        mlflow.log_metrics(metrics_repot["weighted avg"])

        registered_model_name = "multi_model"
        mlflow.pyfunc.log_model(
            artifact_path="multi_model",
            python_model=multi_classifier,
            signature=signature,
            input_example=x_train.sample(5),
            registered_model_name=registered_model_name,
        )

    set_alias(model_name = registered_model_name)

def inference_multimodel():
    """
    Inference the multi model.
    """
    _, x_test, _, _ = get_train_test_data()

    registered_model_name = "multi_model"

    #get the champion model
    model_uri = f"models:/{registered_model_name}@champion"
    model = mlflow.pyfunc.load_model(model_uri)

    predictions = model.predict(x_test, params={"algo": "decision_tree"})
    print(predictions)