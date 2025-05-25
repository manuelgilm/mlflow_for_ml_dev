# EXAMPLES

In this folder, you will find use cases referenced in the playlist to illustrate specific topics. These examples are based on public datasets and, while they may not always reflect real-world scenarios, they are designed to demonstrate particular aspects of the ML lifecycle.

**Notebooks**

Notebooks under `mlflow_for_ml_dev/notebooks/local_model_serving/` are used to demonstrate how to make the API requests in an interactive way.

## Iris Classifier

This example uses the classic Iris dataset, which is included in the scikit-learn package. The dataset provides a simple, well-known classification problem, making it ideal for demonstrating machine learning workflows.

* package: `examples/iris_classifier`

* dataset source: [sklearn load_iris method](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)


* Associated inference notebook:[deploying_local_iris_model.ipynb](/mlflow_for_ml_dev/notebooks/local_model_serving/deploying_local_iris_model.ipynb)

### Available Entrypoints: 

* `iris_clf_train`: Trains the model and register it in the Model registry with name **Iris_Classifier_Model**, The model is set with alias **production** 

* `iris_clf_inference`: Performs batch inference.

* `iris_clf_online_inference`: Performs online inference. Before running this script, ensure that the model has been trained and registered (for example, using the `iris_clf_train` script). Next, deploy the model locally by executing:

```
poetry run mlflow models serve -m models:/Iris_Classifier_Model@production --env-manager local
```

## Digit Recognition

This examples uses a public dataset from kaggle, for more details about the dataset, click [here](https://www.kaggle.com/datasets/bhavikjikadara/handwritten-digit-recognition) 

* package: `examples/digit_recognition`

* dataset source: [kaggle dataset](https://www.kaggle.com/datasets/bhavikjikadara/handwritten-digit-recognition)

* Associated inference notebook: [deploying_local_digit_recognizer.ipynb](/mlflow_for_ml_dev/notebooks/local_model_serving/deploying_local_digit_recognizer.ipynb)

### Available Entrypoints: 

* `digit_recog_train`: Trains the model and register it in the Model registry with name **Digit_Recognition_Model**, The model is set with alias **production** 

* `digit_recog_inference`: Performs batch inference.

* `digit_recog_online_inference`: Performs online inference. Before running this script, ensure that the model has been trained and registered (for example, using the `digit_recog_train` script). Next, deploy the model locally by executing:

```
poetry run mlflow models serve -m models:/Digit_Recognition_Model@production --env-manager local
```

## Diabetes Prediction

This example utilizes a publicly available dataset from Kaggle. The project highlights the importance of including additional code dependencies within the model to ensure reproducibility and seamless deployment.

* package: `examples/diabetes_prediction`

* Dataset Source: [Kaggle Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

* Associated Inference Notebook: [deploying_local_diabetes_prediction.ipynb](/mlflow_for_ml_dev/notebooks//local_model_serving/deploying_local_diabetes_prediction.ipynb)

### Available Entrypoints: 

* `diabetes_pred_train`: Trains the model and register it in the Model registry with name **Diabetes_Prediction_Model**, The model is set with alias **production**
* `diabetes_pred_inference`: Performs batch inference.
* `diabetes_pred_online_inference`:  Performs online inference. Before running this script, ensure that the model has been trained and registered (for example, using the `diabetes_pred_train` script). Next, deploy the model locally by executing:

`poetry run mlflow models serve -m models:/Digit_Recognition_Model@production --env-manager local`

## Walmart Sales
This example uses a public dataset on Walmart sales over a specific period. The primary goal of this use case is to demonstrate how to deploy multiple models using a single serving endpoint. Additionally, it highlights some limitations of MLflow's ability to automatically generate the appropriate Dockerfile for model deployment.

* package: `examples/walmart_sales_regression`

* Dataset Source: [Kaggle Dataset](https://www.kaggle.com/datasets/mikhail1681/walmart-sales)

* Associated Inference Notebook: [deploying_local_sales_regressor.ipynb](/mlflow_for_ml_dev/notebooks/local_model_serving/deploying_local_sales_regressor.ipynb)

### Available Entrypoints:

* `walmart_reg_train`: Trains the model and register it in the Model registry with name **walmart-store-sales-regressor**, The model is set with alias **production**
* `walmart_reg_inference`: Perform batch inference.
* `walmart_reg_online_inference`:  Performs online inference. Before running this script, ensure that the model has been trained and registered (for example, using the `walmart_reg_train` script). Next, deploy the model locally by executing:

`poetry run mlflow models serve -m models:/walmart-store-sales-regressor@production -p 5000 --env-manager local`


