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
    algo = "random_forest"
    experiment_name = "rf_classifier_sp"
    pipeline = TrainingPipeline(
        algo=algo,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        target_column=target_column,
    )

    # train the pipeline
    pipeline.train(x_train, y_train)

    predictions = pipeline.predict(x_test)
    print(predictions.head())

    # set tracking uri
    mlflow.set_tracking_uri(get_project_root() / "mlruns")
    # set experiment
    experiment = get_or_create_experiment(
        name=experiment_name,
        tags={"algo": algo, "project_name": "student_performance"},
    )
    # set the experiment description
    mlflow.set_experiment_tag(
        key="mlflow.note.content",
        value="Experiment to test the random forest classifier on the student performance dataset.",
    )

    model_signature = infer_signature(x_test, y_test)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id
    ) as run:
        # Set description for the run
        mlflow.set_tag(
            "mlflow.note.content", "Run to test the random forest classifier."
        )
        mlflow.set_tag("algo", algo)
        mlflow.set_tag("project_name", "student_performance")
        mlflow.log_params(pipeline.pipeline.get_params())
         
        mlflow.sklearn.log_model(
            sk_model=pipeline.pipeline,
            artifact_path="pipeline_model",
            input_example=x_test.sample(5),
            signature=model_signature,
        )

        model_uri = f"runs:/{run.info.run_id}/pipeline_model"
        eval_data = x_test.copy()
        eval_data["target"] = y_test

        eval_result = mlflow.evaluate(
            model = model_uri,
            data = eval_data,
            targets = "target",
            model_type = "classifier",
            evaluator_config = {
                "metric_prefix": "eval_"
            }
        )


def student_performance_inference():
    """
    Function to perform inference on the student performance dataset.
    """
    experiment_name = "rf_classifier_sp"
    algo = "random_forest"
    # set tracking uri
    mlflow.set_tracking_uri(get_project_root() / "mlruns")

    # search for the run 
    runs = mlflow.search_runs(
        experiment_names = [experiment_name],
        filter_string=f"tags.algo = '{algo}'",
        order_by=["metrics.eval_f1_score desc"],

    )
    # show the results
    print(runs[["run_id","metrics.eval_recall_score","metrics.eval_f1_score"]].head())

    # get the best run
    best_run = runs.iloc[0]
    run_id = best_run["run_id"]
    print(f"Best run id: {run_id}")

    # load the model
    model_uri = f"runs:/{run_id}/pipeline_model"
    model = mlflow.sklearn.load_model(model_uri)
    print(type(model))

    _, x_test, _, _ = create_training_and_testing_dataset()

    predictions = model.predict(x_test)
    print(predictions)