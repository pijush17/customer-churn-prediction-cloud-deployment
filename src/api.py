"""FastAPI service for customer churn predictions."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.pkl"
MODEL_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "internet_service",
    "tech_support",
]

app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLflow MLOps Assignment Deployment",
    version="1.0.0",
)


class ChurnRequest(BaseModel):
    tenure: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)
    contract_type: str
    internet_service: str
    tech_support: str


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/")
def home() -> dict[str, str]:
    return {"project": "Customer Churn Prediction", "status": "running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/model-info")
def model_info() -> dict[str, str | list[str]]:
    """Describe the deployed model and its required input fields."""
    return {
        "model": "RandomForestClassifier",
        "experiment": "Customer Churn Prediction",
        "features": MODEL_FEATURES,
    }


@app.post("/predict")
def predict(request: ChurnRequest) -> dict[str, int | str]:
    try:
        prediction = model.predict(pd.DataFrame([request.model_dump()]))[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    churn_prediction = int(prediction)
    return {
        "churn_prediction": churn_prediction,
        "result": "Customer likely to churn" if churn_prediction == 1 else "Customer likely to stay",
        "recommendation": (
            "Offer a retention plan or customer-support follow-up."
            if churn_prediction == 1
            else "Continue with regular customer engagement."
        ),
    }
