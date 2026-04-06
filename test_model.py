from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_FILE = BASE_DIR / "kmeans_artifacts.joblib"


@lru_cache(maxsize=1)
def load_artifacts():
    artifacts = joblib.load(ARTIFACTS_FILE)
    return (
        artifacts["model"],
        artifacts["scaler"],
        artifacts["feature_columns"],
        artifacts["train_features"],
        artifacts["train_labels"],
    )


def build_predict_row(data, feature_columns):
    row = pd.DataFrame([data], columns=feature_columns)
    row["Gender"] = row["Gender"].map({"Male": 1, "Female": 0})
    return row


def predict_cluster(data):
    model, scaler, feature_columns, _, _ = load_artifacts()
    row = build_predict_row(data, feature_columns)
    row_scaled = scaler.transform(row)
    cluster = int(model.predict(row_scaled)[0])
    return cluster, row


def visualize_with_new_customer(new_row, cluster):
    model, scaler, _, train_features, train_labels = load_artifacts()
    centroids = scaler.inverse_transform(model.cluster_centers_)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        train_features["Annual Income (k$)"],
        train_features["Spending Score (1-100)"],
        c=train_labels,
        cmap="viridis",
        alpha=0.55,
        edgecolors="none",
        label="Training data",
    )

    plt.scatter(
        centroids[:, 2],
        centroids[:, 3],
        s=230,
        c="red",
        marker="X",
        edgecolors="black",
        label="Centroids",
    )

    plt.scatter(
        [new_row["Annual Income (k$)"].iloc[0]],
        [new_row["Spending Score (1-100)"].iloc[0]],
        s=220,
        c="orange",
        edgecolors="black",
        marker="*",
        label=f"New customer -> Cluster {cluster}",
    )

    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("K-Means Test: New Customer Cluster Assignment")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "kmeans_test_result.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    try:
        load_artifacts()
    except Exception as error:
        print(f"Failed to load artifacts: {error}")
        print("Run train_model.py first.")
        return

    gender = input("Enter Gender (Male/Female): ").strip().title()
    age = int(input("Enter Age: ").strip())
    annual_income = float(input("Enter Annual Income (k$): ").strip())
    spending_score = float(input("Enter Spending Score (1-100): ").strip())

    if gender not in {"Male", "Female"}:
        print("Gender must be 'Male' or 'Female'.")
        return

    payload = {
        "Gender": gender,
        "Age": age,
        "Annual Income (k$)": annual_income,
        "Spending Score (1-100)": spending_score,
    }

    cluster, new_row = predict_cluster(payload)
    print(f"Predicted cluster: {cluster}")

    try:
        visualize_with_new_customer(new_row, cluster)
    except Exception as error:
        print(f"Visualization skipped: {error}")
        print("Cluster prediction is still valid.")


if __name__ == "__main__":
    main()
