
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# 1. Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer Churn Prediction")

# 2. Load dataset
df = pd.read_csv("data/churn.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# 3. Separate features and target
X = df.drop("churn", axis=1)
y = df["churn"]

# 4. Identify categorical features
categorical_features = [
    "contract_type",
    "internet_service",
    "tech_support",
]

# 5. Create preprocessing component
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        )
    ],
    remainder="passthrough",
)

# 6. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
)

# 7. Combine preprocessing and model
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model),
    ]
)

# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

F1_THRESHOLD = 0.80

# 9. Train, evaluate, and track the model
with mlflow.start_run():

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    # Log parameters
    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("test_size", 0.2)

    # Log evaluation metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Display evaluation results
    print("\nModel Training Results")
    print("----------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Required F1 Score: {F1_THRESHOLD:.4f}")

    # 10. Validate the model
    if f1 >= F1_THRESHOLD:

        print("\nVALIDATION PASSED")

        mlflow.set_tag("validation_status", "PASSED")

        # Save the model locally for the API
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/churn_model.pkl")

        print("Local model saved: models/churn_model.pkl")

        # Log the model as an MLflow artifact
        mlflow.sklearn.log_model(
    pipeline,
    name="churn_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)
        print("Model logged to MLflow successfully.")

    else:
        mlflow.set_tag("validation_status", "FAILED")

        print("\nVALIDATION FAILED")
        print("Model did not meet the required F1 threshold.")

        raise RuntimeError(
            f"F1 score {f1:.4f} is below "
            f"the required threshold {F1_THRESHOLD:.4f}."
        )

print("\nTraining and validation completed successfully.")
