"""Train, validate, and track the customer-churn classifier."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "churn.csv"
MODEL_PATH = ROOT_DIR / "models" / "churn_model.pkl"
MLFLOW_DB_PATH = ROOT_DIR / "mlflow.db"
EXPERIMENT_NAME = "Customer Churn Prediction"
F1_THRESHOLD = 0.80

CATEGORICAL_FEATURES = ["contract_type", "internet_service", "tech_support"]
REQUIRED_COLUMNS = [
    "tenure", "monthly_charges", "total_charges", *CATEGORICAL_FEATURES, "churn"
]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the training dataset."""
    dataset = pd.read_csv(path)
    missing_columns = set(REQUIRED_COLUMNS) - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    if dataset.empty:
        raise ValueError("Dataset must contain at least one row.")
    if not set(dataset["churn"].dropna().unique()).issubset({0, 1}):
        raise ValueError("The churn target must contain only 0 and 1 values.")
    return dataset


def build_pipeline() -> Pipeline:
    """Create the preprocessing and classifier pipeline."""
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    return Pipeline([("preprocessing", preprocessor), ("model", classifier)])


def train_and_evaluate(dataset: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    """Fit the pipeline and return it with holdout-set metrics."""
    features = dataset.drop(columns="churn")
    target = dataset["churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    return pipeline, {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    dataset = load_dataset()
    print(f"Dataset loaded successfully: {dataset.shape[0]} rows, {dataset.shape[1]} columns")

    with mlflow.start_run():
        pipeline, metrics = train_and_evaluate(dataset)
        mlflow.log_params(
            {"model": "Random Forest", "n_estimators": 100, "max_depth": 5, "test_size": 0.2}
        )
        mlflow.log_metrics(metrics)
        print("\nModel Training Results")
        for name, value in metrics.items():
            print(f"{name.replace('_', ' ').title()}: {value:.4f}")
        print(f"Required F1 Score: {F1_THRESHOLD:.4f}")

        if metrics["f1_score"] < F1_THRESHOLD:
            mlflow.set_tag("validation_status", "FAILED")
            raise RuntimeError(
                f"F1 score {metrics['f1_score']:.4f} is below the required threshold {F1_THRESHOLD:.4f}."
            )

        mlflow.set_tag("validation_status", "PASSED")
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        mlflow.sklearn.log_model(pipeline, name="churn_model")
        print(f"\nVALIDATION PASSED\nModel saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
