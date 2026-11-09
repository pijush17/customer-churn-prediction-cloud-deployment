from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app
from src.train import REQUIRED_COLUMNS, build_pipeline, load_dataset, train_and_evaluate


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_dataset_has_required_schema():
    dataset = load_dataset(ROOT_DIR / "data" / "churn.csv")
    assert not dataset.empty
    assert set(REQUIRED_COLUMNS).issubset(dataset.columns)
    assert set(dataset["churn"].unique()).issubset({0, 1})


def test_pipeline_can_train_and_predict():
    dataset = load_dataset()
    pipeline, metrics = train_and_evaluate(dataset)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1_score"}
    assert 0 <= metrics["f1_score"] <= 1
    assert pipeline.predict(dataset.drop(columns="churn").head(1))[0] in {0, 1}


def test_pipeline_handles_unknown_category():
    dataset = load_dataset()
    pipeline = build_pipeline().fit(dataset.drop(columns="churn"), dataset["churn"])
    sample = dataset.drop(columns="churn").head(1).copy()
    sample.loc[:, "contract_type"] = "Unknown"
    assert pipeline.predict(sample)[0] in {0, 1}


def test_api_health_and_prediction():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "healthy"}
    assert client.get("/model-info").json()["features"] == [
        "tenure",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "internet_service",
        "tech_support",
    ]
    response = client.post(
        "/predict",
        json={
            "tenure": 12,
            "monthly_charges": 75.5,
            "total_charges": 906.0,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "tech_support": "No",
        },
    )
    assert response.status_code == 200
    assert response.json()["churn_prediction"] in {0, 1}
    assert "recommendation" in response.json()
