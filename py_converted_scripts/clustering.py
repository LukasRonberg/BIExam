import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.decomposition import PCA
from yellowbrick.cluster import SilhouetteVisualizer


def load_and_scale_data(csv_path: str, drop_cols: list = None) -> (pd.DataFrame, np.ndarray):
    """
    Load cleaned data CSV and return original df and scaled numeric features array.
    drop_cols: list of columns to drop before scaling (e.g., ['year', 'country_numeric']).
    """
    df = pd.read_csv(csv_path)
    X = df.select_dtypes(include=[np.number]).copy()
    if drop_cols:
        X = X.drop(columns=drop_cols, errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled


def compute_distortions(X_scaled: np.ndarray, k_range: range) -> list:
    """
    Compute distortion (average minimum distance) for each k in k_range.
    """
    distortions = []
    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        model.fit(X_scaled)
        dists = cdist(X_scaled, model.cluster_centers_, 'euclidean')
        distortions.append(dists.min(axis=1).mean())
    return distortions


def plot_elbow(k_range: range, distortions: list):
    """
    Plot elbow method (distortion vs k).
    """
    fig, ax = plt.subplots()
    ax.plot(list(k_range), distortions, 'bx-')
    ax.set_xlabel('Antal clusters k')
    ax.set_ylabel('Distortion (gns. mindsteafstand)')
    ax.set_title('Elbow-metode: Distortion vs k')
    plt.tight_layout()
    return fig


def compute_silhouette_scores(X_scaled: np.ndarray, k_range: range) -> list:
    """
    Compute silhouette scores for each k in k_range.
    """
    scores = []
    for k in k_range:
        model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = model.fit_predict(X_scaled)
        score = metrics.silhouette_score(X_scaled, labels, metric='euclidean', sample_size=len(X_scaled))
        scores.append(score)
    return scores


def plot_silhouette(k_range: range, scores: list):
    """
    Plot silhouette score vs k.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(k_range), scores, 'bx-')
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette-metode for optimal k')
    plt.tight_layout()
    return fig


def train_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    """
    Train final KMeans model with k clusters.
    """
    model = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    model.fit(X_scaled)
    return model


def assign_clusters(df: pd.DataFrame, model: KMeans) -> pd.DataFrame:
    """
    Predict cluster labels and add 'cluster' column to df.
    """
    df_out = df.copy()
    df_out['cluster'] = model.labels_
    return df_out


def inspect_clusters(df: pd.DataFrame, model: KMeans):
    """
    Print chosen k, cluster sizes, and centers (in scaled space).
    Requires both the DataFrame with 'cluster' column and the trained KMeans model.
    """
    """
    Print chosen k, cluster sizes, and centers (in scaled space).
    """
    k = df['cluster'].nunique()
    sizes = df['cluster'].value_counts()
    centers = model.cluster_centers_
    print(f"Valgt k: {k}")
    print("Cluster størrelser:")
    print(sizes)
    print("Cluster-center (skaleret):")
    print(centers)


def compute_pca_projection(X_scaled: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Fit PCA and return 2D projection of X_scaled.
    """
    pca = PCA(n_components=n_components, random_state=42)
    proj = pca.fit_transform(X_scaled)
    return proj, pca


def plot_pca_clusters(proj: np.ndarray, labels: np.ndarray):
    """
    Scatter plot of PCA projection colored by cluster label.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(proj[:,0], proj[:,1], c=labels, cmap='viridis', s=10, alpha=0.6)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA-projektion af clusters')
    plt.tight_layout()
    return fig


def plot_centroids_on_pca(pca: PCA, model: KMeans, ax=None):
    """
    Plot KMeans centroids on top of an existing PCA scatter.
    """
    centers_proj = pca.transform(model.cluster_centers_)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(centers_proj[:,0], centers_proj[:,1], c='red', marker='X', s=100, label='Centroids')
    ax.legend()
    return ax


def plot_decision_boundaries(proj: np.ndarray, model: KMeans, pca: PCA):
    """
    Plot decision boundaries for KMeans in PCA space.
    """
    centers_proj = pca.transform(model.cluster_centers_)
    x_min, x_max = proj[:,0].min()-1, proj[:,0].max()+1
    y_min, y_max = proj[:,1].min()-1, proj[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]
    dists = np.linalg.norm(grid[:,None,:] - centers_proj[None,:,:], axis=2)
    labels_grid = np.argmin(dists, axis=1).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(labels_grid, origin='lower', extent=(x_min,x_max,y_min,y_max), cmap='viridis', alpha=0.3)
    ax.scatter(proj[:,0], proj[:,1], c=model.labels_, s=20, cmap='viridis', edgecolor='white')
    plot_centroids_on_pca(pca, model, ax)
    ax.set_title('Beslutningsgrænser i PCA-rum')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.tight_layout()
    return fig


def silhouette_visualization(X_scaled: np.ndarray, model: KMeans):
    """
    Create a manual silhouette plot and return the matplotlib Figure.
    """
    from sklearn.metrics import silhouette_samples
    labels = model.labels_
    n_clusters = len(np.unique(labels))
    sil_vals = silhouette_samples(X_scaled, labels, metric='euclidean')

    fig, ax = plt.subplots(figsize=(6, 4))
    y_lower = 10
    for i in range(n_clusters):
        ith_vals = sil_vals[labels == i]
        ith_vals.sort()
        size_i = ith_vals.shape[0]
        y_upper = y_lower + size_i
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, ith_vals,
            alpha=0.7
        )
        ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10  # 10 for spacing between clusters

    avg_score = np.mean(sil_vals)
    ax.axvline(avg_score, color='red', linestyle='--')
    ax.set_title(f"Silhouette Plot (k={n_clusters})")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster label")
    plt.tight_layout()
    return fig