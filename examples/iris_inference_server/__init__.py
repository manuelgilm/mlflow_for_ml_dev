from examples.iris_inference_server.routers import inference
from examples.iris_inference_server.load_model import ModelLoader
from fastapi import FastAPI
from contextlib import asynccontextmanager

API_VERSION = "v1"


@asynccontextmanager
async def load_ml_model(app: FastAPI):
    """
    Context manager to load the ML model.
    This is a placeholder for actual model loading logic.
    """
    try:
        # Load your ML model here
        ml_loader = ModelLoader()
        ml_loader.load_model("Iris_Classifier_Model", "Production")
        yield  # This is where the model would be used
        ml_loader.clear_models()  # Clear the model after use
    finally:
        print("Model loaded successfully.")


app = FastAPI(title="Inference Server", lifespan=load_ml_model)

app.include_router(inference, prefix=f"/{API_VERSION}", tags=["inference"])
