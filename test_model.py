from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

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
    train_scaled = scaler.transform(train_features)
    new_row_scaled = scaler.transform(new_row)

    pca = PCA(n_components=2)
    train_projected = pca.fit_transform(train_scaled)
    centroids_projected = pca.transform(model.cluster_centers_)
    new_customer_projected = pca.transform(new_row_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        train_projected[:, 0],
        train_projected[:, 1],
        c=train_labels,
        cmap="tab10",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        label="Training data",
    )

    plt.scatter(
        centroids_projected[:, 0],
        centroids_projected[:, 1],
        s=280,
        c="#e53935",
        marker="X",
        edgecolors="black",
        linewidths=1.1,
        label="Centroids",
    )

    plt.scatter(
        [new_customer_projected[0, 0]],
        [new_customer_projected[0, 1]],
        s=260,
        c="orange",
        edgecolors="black",
        marker="*",
        label=f"New customer -> Cluster {cluster}",
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("K-Means Test: New Customer Assignment (PCA Projection)")
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
