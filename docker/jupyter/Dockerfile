# FROM python:3.11-slim
# RUN mkdir /mlflow_for_ml_dev
# COPY . /mlflow_for_ml_dev
# WORKDIR /mlflow_for_ml_dev
# RUN pip3 install poetry 
# RUN poetry install
# WORKDIR /mlflow_for_ml_dev/mlflow_for_ml_dev/notebooks
# CMD ["poetry","run", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Pick your favorite docker-stacks image
FROM quay.io/jupyter/minimal-notebook


# Install mlflow 

# # Moving relevant files to the container
RUN mkdir -p mlflow_for_ml_dev/mlflow_for_ml_dev
COPY /mlflow_for_ml_dev/ mlflow_for_ml_dev/mlflow_for_ml_dev
COPY pyproject.toml pyproject.toml poetry.toml README.md /home/jovyan/mlflow_for_ml_dev/
WORKDIR /home/jovyan/mlflow_for_ml_dev
RUN pip install poetry
RUN poetry install --without dev





