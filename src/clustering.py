from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterResult:
    labels: np.ndarray  # (n,)
    method: str


def kmeans_cluster(Z: np.ndarray, n_clusters: int, seed: int = 42) -> ClusterResult:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    y = km.fit_predict(Z)
    return ClusterResult(labels=y, method=f"kmeans(k={n_clusters})")


def agglomerative_cluster(Z: np.ndarray, n_clusters: int) -> ClusterResult:
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(n_clusters=n_clusters)
    y = model.fit_predict(Z)
    return ClusterResult(labels=y, method=f"agglomerative(k={n_clusters})")


def dbscan_cluster(Z: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> ClusterResult:
    from sklearn.cluster import DBSCAN

    model = DBSCAN(eps=eps, min_samples=min_samples)
    y = model.fit_predict(Z)
    return ClusterResult(labels=y, method=f"dbscan(eps={eps},min={min_samples})")
