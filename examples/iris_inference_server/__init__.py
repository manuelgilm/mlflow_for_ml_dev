from examples.utils.decorators import mlflow_tracking_uri
from fastapi import FastAPI
from fastapi import Request
from typing import List
from contextlib import asynccontextmanager

ml_models = {}


@mlflow_tracking_uri
@asynccontextmanager
async def load_ml_model(app: FastAPI):
    """
    Context manager to load the ML model.
    This is a placeholder for actual model loading logic.
    """
    try:
        import mlflow

        # Load your ML model here
        print("Loading ML model...")
        registered_model_name = "Iris_Classifier_Model"
        model_uri = f"models:/{registered_model_name}@Production"
        model = mlflow.sklearn.load_model(model_uri=model_uri)
        ml_models[registered_model_name] = model
        yield  # This is where the model would be used
        ml_models.clear()  # Clear the model after use
    finally:
        print("Model loaded successfully.")


app = FastAPI(title="Inference Server", lifespan=load_ml_model)


@app.post("/predict")
async def root(request: Request):
    """
    Root endpoint for the inference server.
    This endpoint accepts a POST request with a JSON body containing
    the features for prediction.
    It returns the prediction made by the ML model.
    """

    body = await request.json()
    print("Body received:", body)
    features = body.get("features", None)
    if not features or not isinstance(features, List):
        return {"error": "Invalid input. 'features' must be a list."}
    model = ml_models.get("Iris_Classifier_Model", None)
    if model:
        # Assuming the model has a predict method
        prediction = model.predict([features])
        return {"prediction": prediction.tolist()}
    else:
        return {"error": "Model not found."}
