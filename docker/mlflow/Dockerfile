FROM python:3.11-slim
RUN mkdir /mlflow_for_ml_dev
WORKDIR /mlflow_for_ml_dev
COPY . /mlflow_for_ml_dev
RUN pip3 install poetry 
RUN poetry install
WORKDIR /mlflow_for_ml_dev/notebooks
CMD ["poetry", "run", "mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]