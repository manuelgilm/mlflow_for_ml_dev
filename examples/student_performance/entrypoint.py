from examples.student_performance.src.data_processing import get_categorical_features
from examples.student_performance.src.data_processing import get_numerical_features
from examples.student_performance.src.pipelines import TrainingPipeline
from examples.student_performance.src.data_processing import (
    create_training_and_testing_dataset,
)
from mlflow_for_ml_dev.src.utils.folder_operations import get_project_root
from examples.utils.mlflow_utils import get_or_create_experiment
from sklearn.metrics import classification_report
import mlflow
from mlflow.models.signature import infer_signature


def log_sklearn_pipeline():
    """
    Login the sklearn pipeline.
    """
    x_train, x_test, y_train, y_test = create_training_and_testing_dataset()
    numerical_columns = get_numerical_features()
    categorical_columns = get_categorical_features()
    target_column = "GradeClass"

    pipeline = TrainingPipeline(
        algo="random_forest",
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
    )

    # train the pipeline
    pipeline.train(x_train, y_train)

    predictions = pipeline.predict(x_test)
    print(predictions.head())
    metric_report = classification_report(
        y_test, predictions["predictions"], output_dict=True
    )

    # set tracking uri
    mlflow.set_tracking_uri(get_project_root() / "mlruns")
    # set experiment
    experiment = get_or_create_experiment(
        name="rf_classifier_sp",
        tags={"algo": "random_forest", "project_name": "student_performance"},
    )
    # set the experiment description
    mlflow.set_experiment_tag(
        key="mlflow.note.content",
        value="Experiment to test the random forest classifier on the student performance dataset.",
    )

    model_signature = infer_signature(x_test, y_test)
    with mlflow.start_run(
        run_name="rf_classifier_sp", experiment_id=experiment.experiment_id
    ):
        # Set description for the run
        mlflow.set_tag(
            "mlflow.note.content", "Run to test the random forest classifier."
        )
        mlflow.log_param("algo", "random_forest")
        mlflow.log_params(pipeline.pipeline.get_params())
        mlflow.log_dict(
            dictionary=metric_report, artifact_file="classification_report.json"
        )
        mlflow.log_dict(
            dictionary=metric_report, artifact_file="classification_report.yaml"
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline.pipeline,
            artifact_path="pipeline_model",
            input_example=x_test.sample(5),
            signature=model_signature,
        )

