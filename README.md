# MLflow for Machine Learning Development

This repository is part of the playlist on Youtube named: MLflow for Machine Learning Development (yet to be created)

# Notebooks 

## Experiment Tracking Fundamentals

A series of notebooks is provided to follow along the playlist, there are two folders with notebooks that addresses two main topics. 

The fundamentals of experiment tracking and machine learning development with MLflow are explored in the notebooks under this topic. It addresses the concepts of Experiment and Run, and the metadata involved during ML model development, such as metrics, artifacts, parameters, and tags. Additionally, it covers tracking ML models using MLflow flavors, including how to provide signatures, input examples, and other metadata with the model.

Custom model creation is also addressed in this section, demonstrating how to use the Pyfunc Flavor to define your custom model using a Python function or Python class. Finally, it covers concepts associated with the Model Registry, including Model objects, Model versions, aliases, and Model Registry operations.

## Traditional ML Model Evaluation.


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


## Content: 

- `mlflow_for_ml_dev` package
    - notebooks 
        - experiments.
            * Main Concepts.
            * Retrieve Experiments.
            * Restore Experiments.
            
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