from package.feature.data_processing import get_feature_dataframe
from package.ml_training.retrieval import get_train_test_score_set

from package.ml_training.train import train_model
from package.ml_training.preprocessing_pipeline import get_pipeline
from package.utils.utils import set_or_create_experiment
from package.utils.utils import get_performance_plots
import mlflow 

if __name__ == '__main__':
    experiment_name = "house_pricing_classifier"
    run_name = "training_classifier"
    df = get_feature_dataframe()
    print(df.head())

    x_train, x_test, x_score, y_train, y_test, y_score = get_train_test_score_set(df)

    features = [f for f in x_train.columns if f not in ["id", "target", "MedHouseVal"]]

    pipeline = get_pipeline(numerical_features =features, categorical_features= [])

    experiment_id = set_or_create_experiment(experiment_name=experiment_name)

    run_id, model = train_model(pipeline = pipeline, run_name = run_name, x = x_train, y = y_train)

    y_pred = model.predict(x_test)

    performance_plots = get_performance_plots(y_true=y_test, y_pred=y_pred, prefix="test")
    # log performance metrics
    with mlflow.start_run(run_id=run_id):
        for plot_name, fig in performance_plots.items():
            mlflow.log_figure(fig, plot_name+".png")
