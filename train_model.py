import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse

BASE_DIR = Path(__file__).resolve().parent
DATAFILE = BASE_DIR / "Mall_Customers.csv"
ARTIFACTS_FILE = BASE_DIR / "kmeans_artifacts.joblib"
PLOT_FILE = BASE_DIR / "kmeans_clusters.png"


def prepare_features(df):
    working_df = df.copy()
    working_df["Gender"] = working_df["Gender"].map({"Male": 1, "Female": 0})

    feature_columns = [
        "Gender",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)",
    ]
    clean_df = working_df[feature_columns].dropna().copy()
    return clean_df, feature_columns


def draw_cluster_outline(ax, points, edge_color):
    if points.shape[0] < 3:
        return

    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if np.any(eigenvalues <= 0):
        return

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.2 * np.sqrt(eigenvalues)
    ellipse = Ellipse(
        xy=points.mean(axis=0),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor=edge_color,
        linewidth=1.8,
        linestyle="--",
        alpha=0.75,
    )
    ax.add_patch(ellipse)


def visualize_clusters(X_scaled, labels, centroids_scaled):
    pca = PCA(n_components=2)
    projected_points = pca.fit_transform(X_scaled)
    projected_centroids = pca.transform(centroids_scaled)

    fig, ax = plt.subplots(figsize=(11, 7))
    scatter = ax.scatter(
        projected_points[:, 0],
        projected_points[:, 1],
        c=labels,
        cmap="tab10",
        s=68,
        alpha=0.86,
        edgecolors="white",
        linewidths=0.45,
    )

    unique_clusters = sorted(np.unique(labels).tolist())
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_clusters), 2)))
    for index, cluster_id in enumerate(unique_clusters):
        cluster_points = projected_points[labels == cluster_id]
        draw_cluster_outline(ax, cluster_points, cluster_colors[index])

    ax.scatter(
        projected_centroids[:, 0],
        projected_centroids[:, 1],
        s=340,
        c="#e53935",
        marker="X",
        edgecolors="black",
        linewidths=1.2,
        label="Centroids",
        zorder=5,
    )

    for idx, (center_x, center_y) in enumerate(projected_centroids):
        ax.text(center_x + 0.03, center_y + 0.03, f"C{idx}", fontsize=10, fontweight="bold")

    ax.set_title("K-Means Customer Segments (PCA Projection)", fontsize=14, fontweight="bold")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=320, bbox_inches="tight")
    print(f"Cluster plot saved to '{PLOT_FILE}'")
    plt.show()


def main():
    raw_df = pd.read_csv(DATAFILE)
    features_df, feature_columns = prepare_features(raw_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    kmeans = KMeans(n_clusters=5, random_state=67, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels)
    print("Training completed.")
    print(f"Inertia: {kmeans.inertia_:.4f}")
    print(f"Silhouette score: {silhouette:.4f}")

    visualize_clusters(X_scaled, labels, kmeans.cluster_centers_)

    artifacts = {
        "model": kmeans,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "train_features": features_df,
        "train_labels": labels,
        "silhouette_score": float(silhouette),
        "inertia": float(kmeans.inertia_),
        "dataset_name": "Mall Visiting Customer Data",
    }
    joblib.dump(artifacts, ARTIFACTS_FILE)
    print(f"Artifacts saved to '{ARTIFACTS_FILE}'")


if __name__ == "__main__":
    main()
