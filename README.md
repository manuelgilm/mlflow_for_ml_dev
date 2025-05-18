# MLflow for Machine Learning Development

This repository is part of the playlist on Youtube named: [MLflow for Machine Learning Development (new)](https://www.youtube.com/playlist?list=PLQqR_3C2fhUUOmaeowgv4WquvH515zVmo)

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
### Experiment Tracking with MLflow:
- [Introduction to the playlist](https://youtu.be/5pPflDSdFLg)
- [Experiments & Runs in MLflow](https://youtu.be/c3OaZjm-n8g)

- `mlflow_for_ml_dev` package
    - notebooks 
        - experiment Tracking Fundamentals.
            * Starting with MLflow Experiments. Notebook: [1_1_experiments_create_experiments](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/1_1_experiments_create_experiments.ipynb):
                - video: [2. Starting with MLflow Experiments](https://youtu.be/xzXWoqX6A9o) 
                
            * Retrieve Experiments. Notebook: [1_2_experiments_retrieve_experiment](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/1_2_experiments_retrieve_experiments.ipynb).
                - video: [3. Retrieving MLflow Experiments.](https://youtu.be/M4FI-_qdlrI)
                
            
            * Updating Experiments. Notebook:[1_3_experiments_update_experiments](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/1_3_experiments_update_experiments.ipynb).
                - video: [4. Updating MLflow experiments](https://youtu.be/hs5645Z3W94)
            * Deleting experiments. Notebook: [1_4_experiments_delete_experiments](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/1_4_experiments_delete_experiments.ipynb)
                - video: [5. Deleting MLflow Experiments.](https://youtu.be/W6Lex3leBFI)
                
            * Starting with MLflow runs. Notebook: [2_1_runs_create_run](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/2_1_runs_create_run.ipynb)
                - video: [6. Runs, creating a MLflow Run](https://youtu.be/dbIcyaShlM8)
                - video: [7. Runs, Using runs as context manager](https://youtu.be/Z6_BtG_sxAc)
                - video: [8. Runs, Using the MLflow client to manage runs.](https://youtu.be/AkPUuNO4_WY)

            * Retrieving runs. Notebook: [2_2_runs_retrieve_run](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/2_2_runs_retrieve_run.ipynb)
                - video: [9. Runs, retrieving MLflow runs.](https://youtu.be/vHX33prW_cU)
            * Updating runs. Notebook: [2_3_runs_update_run](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/2_3_runs_update_run.ipynb)
                - video: [10. Runs, Updating run tags.](https://youtu.be/T0c61XRyUCQ)
            * Nested runs. Notebook: [2_4_runs_nested_run](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/2_4_runs_nested_runs.ipynb)
                - video: [11. Runs, nested runs.](https://youtu.be/qPg6-Jrzksw)
            * Delete a run. Notebook: [2_5_runs_delete_run](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/2_5_runs_delete_run.ipynb)
                - video: [12. Runs, Delete MLflow runs.](https://youtu.be/ycnMkjmbtxc)
            * Starting with login functions. Notebook: [3_1_login_functions](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/3_1_login_functions.ipynb)
                - video: [13. Login functions. Metrics.](https://youtu.be/You0arp42MA)
                - video: [14. Login Functions, artifacts](https://youtu.be/U__zJudlKNg)
                - video: [15. Login Functions. Dictionaries and figures.](https://youtu.be/1vBmdSNGY7A)
            * Login ML models. Notebook: [3_2_logging_models](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/3_2_logging_models.ipynb)
                - video: [16. MLflow Models. what's a flavor?](https://youtu.be/CCHgaAioBAg)
                - video: [17. MLflow models. The sklearn flavor.](https://youtu.be/-4zffRcHbks)
            * Model Signature. Notebook: [3_3_model_signature](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/3_3_model_signature.ipynb)
                - video: [18. MLflow models. Model Signature.](https://youtu.be/A7gTCrQV7to)
                
            * Signature Enforcement. Notebook: [3_4_signture_enforcement](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/3_4_signature_enforcement.ipynb)
                - video: [19. MLflow models, signature enforcement.](https://youtu.be/pUggO6OCHzY)
                - video: [20. MLflow Models, signature with optional columns.](https://youtu.be/xHjN_iS71TU)
                - video: [21. MLflow Models. Infer model signature.](https://youtu.be/dBP1-H1wKfc)
            * Custom functions. Notebook: [4_1_log_custom_functions](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/4_1_log_custom_functions.ipynb)
                - video: [22. MLflow Models. Pyfunc Flavor.](https://youtu.be/KQnSgnj_QRY)
                - video: [23. MLflow Models, Class based models.](https://youtu.be/fLvCKse0Qto)
                - video: [24. MLflow Models, Custom Models with Signature.](https://youtu.be/UuvJH4Z3CYI)

            * Custom functions context. Notebook: [4_2_custom_functions_context](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/4_2_custom_functions_context.ipynb)
                - video: [25. MLflow Models. Python Model Context](https://youtu.be/FUlw9p7nx1U)
                - video: [26. MLflow Models. Python Model Context Part 2](https://youtu.be/JB3nLYnWpoc)
                
            * Registering a model. Notebook: [5_1_registering_a_model](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/5_1_registering_a_model.ipynb)
                - video: [27. Model Registry.](https://youtu.be/qxO872AhKNU)
                - video: [28. Model Registry. Registering a Model using the Python SDK.](https://youtu.be/3D_dEeU_OR8)
                
            * Update Registered Model. Notebook: [5_2_update_registered_model](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/5_2_update_registered_model.ipynb)
                - video: [29. Model Registry. Updating model metadata](https://youtu.be/hIqZwnPXYi0)
                - video: [30. Model Registry. Updating model metadata Part 2](https://youtu.be/71OiVI7PJM8)

            * Update model version. Notebook: [5_3_update_model_version](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/5_3_update_model_version.ipynb)
                - video: [31. Model Registry. Updating model metadata Part 3.](https://youtu.be/8_RGIB8itX4)
            * Retrieving Model Information. Notebook: [5_4_retrieve_model_info](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/5_4_retrieve_model_info.ipynb)
                - video: [32. Model Registry. Retrieving models.](https://youtu.be/P-zk1vMN0-0)
            * Loading a registered model. Notebook: [5_5_loading_registered_models](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/5_5_loading_registered_models.ipynb)
                - video: [33. Model Registry. Retrieving a registered  model.](https://youtu.be/jJfDNq2a80Y)
                
            * Deleting registered models. Notebook: [5_6_delete_registered_model_info](/mlflow_for_ml_dev/notebooks/experiment_tracking_fundamentals/5_6_delete_registered_model_info.ipynb)
                - video: [34. Model Registry. Removing model metadata.](https://youtu.be/pe5vMdjJttc)

        - Traditional ML Evaluation with MLflow
            * Model Evaluation with Mlflow. Notebook: [1_1_model_evaluation_with_mlflow](/mlflow_for_ml_dev/notebooks/traditional_ml_evaluation/1_1_model_evaluation_with_mlflow.ipynb)
                - video: [35. Model Evaluation.](https://youtu.be/Fxv5INhlrkk)
                - video: [36. Model Evaluation. Part 2.](https://youtu.be/tnn_LNqj-i8)
                - video: [37. Model Evaluation. Part 3](https://youtu.be/IsY9ye169Ao)

            * Defining custom metrics. Notebook: [1_2_custom_metrics](/mlflow_for_ml_dev/notebooks/traditional_ml_evaluation/1_2_custom_metrics.ipynb)
                - video: [38. Model Evaluation. Custom Metrics.](https://youtu.be/j2NDuKyf3GI)
                - video: [39. Model Evaluation. Custom Metrics Part 2.](https://youtu.be/rPmrBMwaAgc)
                - video: [40. Model Evaluation. Custom Metrics Part 3](https://youtu.be/os8dkwE5mpA)

            * Custom artifacts. Notebook: [1_3_custom_artifacts](/mlflow_for_ml_dev/notebooks/traditional_ml_evaluation/1_3_custom_artifacts.ipynb)
                - video: [41. Model Evaluation. Custom Artifacts.](https://youtu.be/_LE9Go4h6-c)

            * More about models and evaluation. Notebook: [1_4_evaluation_with_functions](/mlflow_for_ml_dev/notebooks/traditional_ml_evaluation/1_4_evaluation_with_functions.ipynb)
                - video: [42. Model Evaluation. More about evaluate method.](https://youtu.be/ttm--W1OBVU)
            
            - Example:
                * Custom model (multimodel) Code: [multimodel](/examples/multi_model/)
                    - video: [43. Example 1. Custom model (multimodel)](https://youtu.be/ttm--W1OBVU)
                    - video: [44. Example 1. Custom model (multimodel) Part 2](https://youtu.be/yqvkXNADsYU)

### Local Model Serving Use cases.

In this section, we explore how MLflow enables local model serving. You'll learn how to launch a local MLflow server to deploy and serve ML models, making them accessible for real-time predictions via REST API endpoints. This approach is useful for testing, prototyping, and integrating models into local applications before moving to production environments.

#### Usecases
* Iris classifier 

* Digit Recognition




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