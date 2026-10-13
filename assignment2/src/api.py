from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLflow MLOps Assignment Deployment",
    version="1.0"
)

model = joblib.load("models/churn_model.pkl")


@app.get("/")
def home():
    return {
        "project": "Customer Churn Prediction",
        "status": "running"
    }


@app.post("/predict")
def predict(
    tenure: int,
    monthly_charges: float,
    total_charges: float,
    contract_type: str,
    internet_service: str,
    tech_support: str
):

    data = pd.DataFrame([{
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "tech_support": tech_support
    }])

    prediction = model.predict(data)[0]

    return {
        "churn_prediction": int(prediction),
        "result": "Customer likely to churn"
        if prediction == 1
        else "Customer likely to stay"
    }