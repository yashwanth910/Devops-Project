"""
app.py
------
FastAPI prediction service.

Endpoints:
    GET  /health   - liveness check
    GET  /features - list of feature names the model expects
    POST /predict  - run inference on a single sample

Run locally:
    uvicorn src.app:app --host 0.0.0.0 --port 8000
"""

from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import predict, get_feature_names

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Serves predictions from the best model registered via the MLOps pipeline.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Mapping of feature name to value. Use GET /features to see required keys.",
        json_schema_extra={
            "example": {
                "mean radius": 14.0,
                "mean texture": 20.0,
                "mean perimeter": 90.0,
                "mean area": 600.0,
                "mean smoothness": 0.1,
            }
        },
    )


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: Dict[str, float] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    try:
        return {"required_features": get_feature_names()}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    try:
        result = predict(request.features)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
