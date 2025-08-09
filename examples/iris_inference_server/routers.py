from fastapi import APIRouter
from fastapi.responses import JSONResponse
import mlflow
from examples.iris_inference_server.schemas import IrisRequest
from examples.iris_inference_server.schemas import IrisResponse
from examples.iris_inference_server.load_model import ModelLoader

inference = APIRouter()


@inference.get("/health")
def health_check():
    return {"status": "healthy"}


@inference.get("/ping")
def ping():
    return {"status": "pong"}


@inference.get("/version")
def version():
    return {"version": mlflow.__version__}


@inference.post("/invocations")
def invocations(iris_request: IrisRequest) -> IrisResponse:
    features = iris_request.model_dump()
    model = ModelLoader.get_model("Iris_Classifier_Model")
    if model:
        # Assuming the model has a predict method
        features = iris_request.get_feature_values()
        prediction = model.predict([features])
        proba = model.predict_proba([features])
        print("prediction:", prediction)
        print("probability:", proba)
        return IrisResponse(species=prediction[0], confidence=proba[0].max())
    else:
        return JSONResponse(content={"error": "Model not found."}, status_code=404)
