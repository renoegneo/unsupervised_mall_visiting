from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
DATAFILE = BASE_DIR / "Mall_Customers.csv"
ARTIFACTS_FILE = BASE_DIR / "kmeans_artifacts.joblib"
PLOT_FILE = BASE_DIR / "kmeans_clusters.png"

DATASET_NAME = "Mall Visiting Customer Data"
DEFAULT_MODEL_COLUMNS = [
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

app = Flask(__name__)


def load_dataset():
    return pd.read_csv(DATAFILE)


def load_metrics():
    if not ARTIFACTS_FILE.exists():
        return {"silhouette_score": None, "inertia": None}

    artifacts = joblib.load(ARTIFACTS_FILE)
    return {
        "silhouette_score": artifacts.get("silhouette_score"),
        "inertia": artifacts.get("inertia"),
    }


def get_used_columns():
    if not ARTIFACTS_FILE.exists():
        return DEFAULT_MODEL_COLUMNS

    artifacts = joblib.load(ARTIFACTS_FILE)
    columns = artifacts.get("feature_columns", DEFAULT_MODEL_COLUMNS)
    return [column for column in columns if column in DEFAULT_MODEL_COLUMNS]


DATASET = load_dataset()
DATASET_COLUMNS = DATASET.columns.tolist()
DATASET_HEAD = DATASET.head(10).to_dict(orient="records")
MODEL_USED_COLUMNS = get_used_columns()
UNUSED_COLUMNS = [col for col in DATASET_COLUMNS if col not in MODEL_USED_COLUMNS]
METRICS = load_metrics()

UNSUPERVISED_EXPLANATION = (
    "Unsupervised learning discovers structure in data without target labels. "
    "The algorithm groups objects by similarity and helps us identify natural "
    "segments in customer behavior."
)

PROJECT_DESCRIPTION = (
    "This project trains a K-Means model on customer profile features "
    "(Gender, Age, Annual Income, and Spending Score) to segment mall visitors. "
    "The dashboard below shows the clustering figure, dataset columns, and a "
    "preview of the first 10 rows."
)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        dataset_name=DATASET_NAME,
        dataset_columns=DATASET_COLUMNS,
        dataset_head=DATASET_HEAD,
        model_used_columns=MODEL_USED_COLUMNS,
        unused_columns=UNUSED_COLUMNS,
        silhouette_score=METRICS["silhouette_score"],
        inertia=METRICS["inertia"],
        unsupervised_explanation=UNSUPERVISED_EXPLANATION,
        project_description=PROJECT_DESCRIPTION,
        plot_exists=PLOT_FILE.exists(),
    )


@app.route("/kmeans_clusters.png")
def kmeans_clusters_plot():
    return send_from_directory(BASE_DIR, PLOT_FILE.name)


if __name__ == "__main__":
    app.run(debug=True)
