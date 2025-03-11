from examples.student_performance.src.data_processing import get_categorical_features
from examples.student_performance.src.data_processing import get_numerical_features
from examples.student_performance.src.pipelines import training_pipeline
from examples.student_performance.src.data_processing import (
    create_training_and_testing_dataset,
)


def main():

    pipeline = training_pipeline()
