# MLflow for Machine Learning Development

This repository is part of the playlist on Youtube named: MLflow for Machine Learning Development (yet to be created)

# Notebooks 


A series of notebooks is provided to follow along the playlist, there are two folders with notebooks that addresses two main topics. 

## Experiment Tracking Fundamentals

The fundamentals of experiment tracking and machine learning development with MLflow are explored in the notebooks under this topic. It addresses the concepts of Experiment and Run, and the metadata involved during ML model development, such as metrics, artifacts, parameters, and tags. Additionally, it covers tracking ML models using MLflow flavors, including how to provide signatures, input examples, and other metadata with the model.

Custom model creation is also addressed in this section, demonstrating how to use the Pyfunc Flavor to define your custom model using a Python function or Python class. Finally, it covers concepts associated with the Model Registry, including Model objects, Model versions, aliases, and Model Registry operations.

## Traditional ML Model Evaluation.

This section focuses on evaluating traditional machine learning models using the `mlflow.evaluate` method. It provides an overview of model evaluation fundamentals, including how to assess model performance. Additionally, it demonstrates how to define custom metrics to tailor evaluations to specific use cases and create custom artifacts to enhance the evaluation process. These concepts are essential for gaining deeper insights into model behavior and improving the overall quality of machine learning workflows.


# Package: `mlflow_for_ml_dev`

To run the notebooks is important to create the virtual environment with the required libraries, such a jupyter an the respective kernel. In addition, you need packages such as `sklearn`, `mlflow` and `pandas`


# How to use with Poetry.

* Installing poetry

To install poetry follow the steps provided in the [Poetry Documentation.](https://python-poetry.org/docs/#installing-with-the-official-installer)

* Creating the virtual environment. 

After installing Poetry, to create the virtual environment (.venv) you can run:

`poetry install`

* Activate the virtual environment.

    * With poetry:

        `poetry shell`

    * Without Poetry:

        Depending on the terminal you are using:

        * Command Prompt (cmd):

            `.venv\Scripts\activate`

        * Git Bash

            `source .venv/Scripts/activate`

# How to use without Poetry.

* Create Virtual Environment. 

    `python -m venv .venv`

* Activate the environment

    Depending on the terminal you are using:

    * Command Prompt (cmd):

        `.venv\Scripts\activate`

    * Git Bash

        `source .venv/Scripts/activate`
        
* Install requirements.

    `pip install -r requirements.txt`

> **Note:** If you are not using Poetry to manage the project and instead created a virtual environment, the functions defined in the entry point modules cannot be executed directly through the command line. To address this, you need to refactor the code to allow execution of specific functions within the Python script. For example:

```python
def train_multi_model():
    # Your training logic here
    ...

if __name__ == "__main__":
    train_multi_model()
```

Then, run the script with the desired function as an argument:

```cmd
(.venv) python path/to/script.py train_multi_model
```

## Content: 

- `mlflow_for_ml_dev` package
    - notebooks 
        - experiment Tracking Fundamentals.
            * Starting with MLflow Experiments. Notebook: [1_1_experiments_create_experiments](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/1_1_experiments_create_experiments.ipynb):
                - video: 
                - video: 
            * Retrieve Experiments. Notebook: [1_2_experiments_retrieve_experiment](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/1_2_experiments_retrieve_experiments.ipynb).
                - video:
                - video:
            
            * Updating Experiments. Notebook:[1_3_experiments_update_experiments](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/1_3_experiments_update_experiments.ipynb).
                - video:
                - video:
            * Deleting experiments. Notebook: [1_4_experiments_delete_experiments](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/1_4_experiments_delete_experiments.ipynb)
                - video
                - video
            * Starting with MLflow runs. Notebook: [2_1_runs_create_run](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/2_1_runs_create_run.ipynb)
                - video
                - video
            * Retrieving runs. Notebook: [2_2_runs_retrieve_run](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/2_2_runs_retrieve_run.ipynb)
                - video
                - video
            * Updating runs. Notebook: [2_3_runs_update_run](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/2_3_runs_update_run.ipynb)
                - video
                - video
            * Deleting runs. Notebook: [2_4_runs_delete_run](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/2_4_runs_delete_run.ipynb)
                - video
                - video
            * Starting with login functions. Notebook: [3_1_login_functions](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/3_1_login_functions.ipynb)
                - video
                - video
            * Login ML models. Notebook: [3_2_logging_models](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/3_2_logging_models.ipynb)
                - video
                - video
            * Model Signature. Notebook: [3_3_model_signature](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/3_3_model_signature.ipynb)
                - video
                - video
            * Signature Enforcement. Notebook: [3_4_signture_enforcement](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/3_4_signature_enforcement.ipynb)
                - video
                - video
            * Custom functions. Notebook: [4_1_log_custom_functions](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/4_1_log_custom_functions.ipynb)
                - video
                - video
            * Custom functions context. Notebook: [4_2_custom_functions_context](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/4_2_custom_functions_context.ipynb)
                - video
                - video
            * Registering a model. Notebook: [5_1_registering_a_model](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/5_1_registering_a_model.ipynb)
                - video
                - video
            * Update Registered Model. Notebook: [5_2_update_registered_model](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/5_2_update_registered_model.ipynb)
                - video
                - video
            * Update model version. Notebook: [5_3_update_model_version](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/5_3_update_model_version.ipynb)
                - video
                - video
            * Retrieving Model Information. Notebook: [5_4_retrieve_model_info](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/5_4_retrieve_model_info.ipynb)
                - video
                - video
            * Loading a registered model. Notebook: [5_5_loading_registered_models](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/5_5_loading_registered_models.ipynb)
                - video
                - video
            * Deleting registered models. Notebook: [5_6_delete_registered_model_info](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/5_6_delete_registered_model_info.ipynb)
                - video
                - video

        - Traditional ML Evaluation with MLflow
            * Model Evaluation with Mlflow. Notebook: [1_1_model_evaluation_with_mlflow](/mlflow_for_ml_dev/notebooks/traditional_ml_evaluation/1_1_model_evaluation_with_mlflow.ipynb)

            * Defining custom metrics. Notebook: [1_2_custom_metrics](/mlflow_for_ml_dev/notebooks/traditional_ml_evaluation/1_2_custom_metrics.ipynb)

            * Custom artifacts. Notebook: [1_3_custom_artifacts](/mlflow_for_ml_dev/notebooks/traditional_ml_evaluation/1_3_custom_artifacts.ipynb)

            * More about models and evaluation. Notebook: [1_4_evaluation_with_functions](/mlflow_for_ml_dev/notebooks/traditional_ml_evaluation/1_4_evaluation_with_functions.ipynb)

            
## Contributing

If you want to contribute to this project and make it better, your help is very welcome. Contributing is also a great way to learn more and improve your skills. You can contribute in different ways:

- Reporting a bug
- Coming up with a feature request
- Writing code
- Writing tests
- Writing documentation
- Reviewing code
- Giving feedback on the project
- Spreading the word
- Sharing the project
  
## Contact

If you need to contact me, you can reach me at:

- [manuelgilsitio@gmail.com](manuelgilsitio@gmail.com)
- [linkedin](www.linkedin.com/in/manuelgilmatheus)