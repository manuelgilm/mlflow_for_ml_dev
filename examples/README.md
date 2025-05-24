# EXAMPLES

In this folder, you will find use cases referenced in the playlist to illustrate specific topics. These examples are based on public datasets and, while they may not always reflect real-world scenarios, they are designed to demonstrate particular aspects of the ML lifecycle.

**Notebooks**

Notebooks under `mlflow_for_ml_dev/notebooks/local_model_serving/` are used to demonstrate how to make the API requests in an interactive way.

## Iris Classifier

This example uses the classic Iris dataset, which is included in the scikit-learn package. The dataset provides a simple, well-known classification problem, making it ideal for demonstrating machine learning workflows.

* package: `examples/iris_classifier`

* dataset source: [sklearn load_iris method](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)


* Associated inference notebook:
`mlflow_for_ml_dev/noteboks/local_model_serving/deploying_local_iris_model.ipynb`

### Entrypoints available: 

* `iris_clf_train ="examples.iris_classifier.train:main"`
Trains the model and register it in the Model registry with name **Iris_Classifier_Model**, The model is set with alias **production** 

* `iris_clf_inference = "examples.iris_classifier.inference:main"`
Performs batch inference.

* `iris_clf_online_inference="examples.iris_classifier.online_inference:main"`
Performs online inference. Before running this script, ensure that the model has been trained and registered (for example, using the `iris_clf_train` script). Next, deploy the model locally by executing:

```
poetry run mlflow models serve -m models:/Iris_Classifier_Model@production --env-manager local
```

## Digit Recognition

This examples uses a public dataset from kaggle, for more details about the dataset, click [here](https://www.kaggle.com/datasets/bhavikjikadara/handwritten-digit-recognition) 

* package: `examples/digit_recognition`

* dataset source: [kaggle dataset](https://www.kaggle.com/datasets/bhavikjikadara/handwritten-digit-recognition)




