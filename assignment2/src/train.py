import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
# 1. Load dataset
df = pd.read_csv("data/churn.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# 2. Separate features and target
X = df.drop("churn", axis=1)
y = df["churn"]


# 3. Identify categorical and numerical columns
categorical_features = [
    "contract_type",
    "internet_service",
    "tech_support"
]

numerical_features = [
    "tenure",
    "monthly_charges",
    "total_charges"
]


# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# 5. Create ML model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)


# 6. Create complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 8. Start MLflow experiment
mlflow.set_experiment("Customer Churn Prediction")


with mlflow.start_run():

    # 9. Train model
    pipeline.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/churn_model.pkl")
    print("Model saved to models/churn_model.pkl")

    # 10. Make predictions
    predictions = pipeline.predict(X_test)

    # 11. Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    # 12. Log parameters
    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("test_size", 0.2)

    # 13. Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # 14. Save model to MLflow
   # 14. Model validation
F1_THRESHOLD = 0.80

print("\nModel Validation")
print("-----------------")
print(f"F1 Score: {f1:.4f}")
print(f"Required F1 Score: {F1_THRESHOLD:.4f}")

if f1 >= F1_THRESHOLD:

    print("VALIDATION PASSED")
    print("Model is approved for registration.")

    mlflow.set_tag("validation_status", "PASSED")

    mlflow.sklearn.log_model(
        pipeline,
        name="churn_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

else:

    print("VALIDATION FAILED")
    print("Model is rejected.")

    mlflow.set_tag("validation_status", "FAILED")

    # 15. Display results
    print("\nModel Training Completed!")
    print("-------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nMLflow run completed successfully.")
