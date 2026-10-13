import pandas as pd


def test_dataset_exists():
    df = pd.read_csv("data/churn.csv")

    assert len(df) > 0
    assert "churn" in df.columns


def test_target_values():
    df = pd.read_csv("data/churn.csv")

    assert set(df["churn"].unique()).issubset({0, 1})


def test_required_features():
    df = pd.read_csv("data/churn.csv")

    required_columns = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "internet_service",
        "tech_support",
        "churn"
    ]

    for column in required_columns:
        assert column in df.columns