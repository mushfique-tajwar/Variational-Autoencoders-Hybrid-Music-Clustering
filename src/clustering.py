import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA
from typing import Dict

sns.set(style="whitegrid")


def kmeans_clusters(X: np.ndarray, n_clusters: int = 5, random_state: int = 42):
    # Drop any rows with NaNs just in case
    if np.isnan(X).any() or np.isinf(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples = X.shape[0]
    n_eff_clusters = max(1, min(n_clusters, n_samples))
    km = KMeans(n_clusters=n_eff_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km


def visualize_embedding(X: np.ndarray, labels: np.ndarray, method: str, out_path: str):
    if method.lower() == 'tsne':
        # Perplexity must be < n_samples; choose a safe value
        n = len(X)
        perp = max(2, min(30, (n - 1) // 3)) if n > 2 else 2
        emb = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=perp, random_state=42).fit_transform(X)
    elif method.lower() == 'umap':
        emb = UMAP(n_components=2, random_state=42).fit_transform(X)
    else:
        raise ValueError("method must be 'tsne' or 'umap'")
    plt.figure(figsize=(7,6))
    sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=labels, palette='tab10', s=20, linewidth=0)
    plt.title(f"{method.upper()} of Latent Space")
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def pca_baseline(X: np.ndarray, n_components: int = 16) -> np.ndarray:
    n_samples, n_features = X.shape
    n_comp_eff = max(1, min(n_components, n_samples, n_features))
    pca = PCA(n_components=n_comp_eff, random_state=42)
    Z = pca.fit_transform(X)
    return Z
